import builtins
from recbole.config import Config
from recbole.evaluator.evaluator import Evaluator

import pickle
import torch

import warnings 
warnings.filterwarnings('ignore')

import os
import time
import datetime
import numpy as np
import copy

#Oracle and Poor: recommend to all users the relevant items, or only irrelevant items
model_name = "artificial"
path = f"experiments/{model_name}"

def print(*args, **kwargs):
    with open(f"{path}/log_{dataset}_{model_name}.txt", 'a+') as f:
        return builtins.print(*args, file=f, **kwargs)
    

list_dataset = [
                "Lastfm", 
                "Amazon-lb",
                "QK-video",
                "Jester",
                "ML-10M", 
                "ML-20M"
                ]

struct_path = "struct" #use struct from Pop

list_k = [10]
max_k = max(list_k)

for dataset in list_dataset:

    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(now)

    print(f"Doing {dataset} - {model_name}")

    try:
        with open(f"{path}/base_{dataset}_{model_name}.pickle","rb") as f:
            found = pickle.load(f)
            print("found existing evaluation result ")
            print(found)
    except:
        print(f"Cannot find existing result for {dataset} - {model_name}, proceed with eval")
        config = Config(
            model="Pop", 
            dataset="new_"+dataset, 
            config_file_list=["RecBole/recbole/properties/overall.yaml"],

            config_dict={"topk": list_k, 
                        "metrics":[
                                "RelMetrics",
                                "FairWORel"
                            ]})

        evaluator = Evaluator(config)

        list_filename = [f for f in os.listdir(struct_path) if dataset in f]

        assert len(list_filename) == 1

        with open(f"{struct_path}/{list_filename[0]}","rb") as f:
            struct = pickle.load(f)
            oracle_struct = copy.deepcopy(struct)
            poor_struct = copy.deepcopy(struct)

            #===NEW recommendation also changing the related struct parts (e.g. for evaluation)===
            num_user = struct.get("rec.score").shape[0]
            num_item = struct.get("data.num_items") - 1 #minus the dummy

            oracle_rec = np.zeros_like(struct.get("rec.items")[:,:max_k])
            poor_rec = np.zeros_like(struct.get("rec.items")[:,:max_k])

            pos_items = struct.get("data.pos_items")
            all_choices = [i for i in range(1,num_item+1)]

            rec_topk = struct.get("rec.topk")

            #===Oracle for all users===
            for u in range(num_user):

                copy_of_pos_items_of_u = copy.deepcopy(pos_items[u]) #copying because shuffle is inplace
                np.random.seed(u) #seed follows user idx
                np.random.shuffle(copy_of_pos_items_of_u)
                oracle_rec_for_u = copy_of_pos_items_of_u[:max_k]

                num_pad = oracle_rec_for_u.shape[0]
                if num_pad < max_k:
                    #random the rest from non-positive items (including the ones not in top k), i.e., skip the pos_items[u]

                    choice_for_u = np.setdiff1d(all_choices,pos_items[u])


                    rng = np.random.default_rng(u) #seed follows user idx

                    #randomise choice
                    padded_rec = rng.choice(choice_for_u, size=max_k-num_pad, replace=False)

                    oracle_rec_for_u = np.concatenate([oracle_rec_for_u, padded_rec])

                assert oracle_rec_for_u.shape[0] == max_k

                oracle_rec[u] = oracle_rec_for_u
            oracle_struct.set("rec.items", torch.from_numpy(oracle_rec))


            rel = np.array([np.in1d(oracle_rec[u], pos_items[u], assume_unique=True) for u in range(pos_items.size)], dtype=int) 
            rel = torch.from_numpy(rel)

            # no need to update rec.score as we don't deal with the joint measures
            oracle_updated_rec_topk = torch.cat([rel[:,:max_k],rec_topk[:,-1:]], dim=1) #concat with the number of rel items
            oracle_struct.set("rec.topk", oracle_updated_rec_topk)

            #===Poor: Non-relevant items for all users===
            for u in range(num_user):
            
                choice_for_u = np.setdiff1d(all_choices,pos_items[u])
                rng = np.random.default_rng(u) #seed follows user idx

                #randomise choice
                random_non_positive = rng.choice(choice_for_u, size=max_k, replace=False)

                assert random_non_positive.shape[0] == max_k

                poor_rec[u] = random_non_positive
            poor_struct.set("rec.items", torch.from_numpy(poor_rec))


            rel = np.array([np.in1d(poor_rec[u], pos_items[u], assume_unique=True) for u in range(pos_items.size)], dtype=int) 
            rel = torch.from_numpy(rel)

            # no need to update rec.score as we don't deal with the joint measures
            poor_updated_rec_topk = torch.cat([rel[:,:max_k],rec_topk[:,-1:]], dim=1) #concat with the number of rel items
            poor_struct.set("rec.topk", poor_updated_rec_topk)


            #save oracle and poor 
            with open(f"{path}/struct_{dataset}_oracle.pickle","wb") as f:
                pickle.dump(oracle_struct, f, pickle.HIGHEST_PROTOCOL)

            with open(f"{path}/struct_{dataset}_poor.pickle","wb") as f:
                pickle.dump(poor_struct, f, pickle.HIGHEST_PROTOCOL)

            oracle_rec = oracle_struct.get("rec.items")
            poor_rec = poor_struct.get("rec.items")

            oracle_rel_val = oracle_struct.get("rec.topk")
            poor_rel_val = poor_struct.get("rec.topk")


            curr_struct = copy.deepcopy(oracle_struct)
            curr_rec = copy.deepcopy(oracle_rec)
            curr_rel_val = copy.deepcopy(oracle_rel_val)

            for pct in range(0, 101, 10):
                print(f"Doing {pct}% poor users")
                if pct == 0:

                    start_time = time.time()
                    result = evaluator.evaluate(oracle_struct)
                    print("total time taken: ", time.time() - start_time)
                    print(result)
                    with open(f"{path}/result_{dataset}_{pct}.pickle","wb") as f:
                        pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

                    with open(f"{path}/struct_{dataset}_{pct}.pickle","wb") as f:
                        pickle.dump(oracle_struct, f, pickle.HIGHEST_PROTOCOL)
                    continue

                elif pct == 100:

                    start_time = time.time()
                    result = evaluator.evaluate(poor_struct)
                    print("total time taken: ", time.time() - start_time)
                    print(result)

                    with open(f"{path}/result_{dataset}_{pct}.pickle","wb") as f:
                        pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

                    with open(f"{path}/struct_{dataset}_{pct}.pickle","wb") as f:
                        pickle.dump(poor_struct, f, pickle.HIGHEST_PROTOCOL)
                    
                    continue
                #translate pct to idx
                index_poor = round(num_user * pct/100)
                
                curr_rec[-index_poor:, :] = poor_rec[-index_poor:, :]
                curr_rel_val[-index_poor:, :] = poor_rel_val[-index_poor:, :]

                # update recommendation and relevance values
                # no need to update rec.score as we don't compute joint measures
                curr_struct.set("rec.items", curr_rec)
                curr_struct.set("rec.topk", curr_rel_val)

                assert curr_rec.shape[0] == num_user
                assert curr_rel_val.shape[0] == num_user

                start_time = time.time()
                result = evaluator.evaluate(curr_struct)
                print("total time taken: ", time.time() - start_time)
                print(result)

                with open(f"{path}/result_{dataset}_{pct}.pickle","wb") as f:
                    pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

                with open(f"{path}/struct_{dataset}_{pct}.pickle","wb") as f:
                    pickle.dump(curr_struct, f, pickle.HIGHEST_PROTOCOL)