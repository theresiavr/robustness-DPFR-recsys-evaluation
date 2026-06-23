from glob import glob
from collections import Counter
from itertools import combinations, product

import pandas as pd
import numpy as np

import torch
from copy import deepcopy

from recbole.config import Config
from recbole.evaluator.evaluator import Evaluator
import pickle
import time

import warnings 
warnings.filterwarnings('ignore')

from tqdm import tqdm 

import random 
random.seed(42)


model = "NCL" # doesn't matter which model, just need the data structure
data = "QK-video" # change this for a different dataset

num_samples = 100
max_iter = 100

num_user = 5
num_item = 5
k = 3

best_struct_path = glob(f"cluster/best_struct/*{model}*{data}*")[0]
struct = pd.read_pickle(best_struct_path)

pos_item_all_user = struct.get("data.pos_items")

count_item = Counter()
for user in pos_item_all_user:
    count_item.update(user.tolist())

num_sample = 0
sample_set = set()

while num_sample < num_samples:
    print(f"Doing for {data}, sample number {num_sample}, num_user {num_user}, num_item {num_item}")
    
    if data == "ML-20M":
        top_items = dict(count_item.most_common(num_item + 10)) # get more possible items
    elif data == "ML-10M":
        top_items = dict(count_item.most_common(num_item + 20)) # get more possible items
    elif data == "Lastfm":
        top_items = dict(count_item.most_common(num_item + 250)) # get more possible items
    else:
        top_items = dict(count_item.most_common(num_item))

    # get users whose positive items are a subset of the top items
    users_with_top_items = []
    for user, pos_items in enumerate(pos_item_all_user):
        if set(set(pos_items.tolist())).issubset(top_items.keys()):
            users_with_top_items.append(user)

    flag = True

    iter = 0
    while flag and iter <= max_iter: 
        # sample num_user randomly from users_with_top_items; between the 100 user samples, it's possible to have duplicates, which is to be checked after

        print(f"Sampling users, iter-({iter})...")
        user_sample = random.sample(users_with_top_items, num_user)
        assert len(user_sample) == num_user

        sorted_user_sample = sorted(user_sample)
        sorted_list_user = map(str, sorted_user_sample)

        string_list_user = "-".join(sorted_list_user)


        # checking for duplicates
        if string_list_user in sample_set:
            print("Sample already exists")
            continue

        pos_item_sampled_user = pos_item_all_user[user_sample]

        # check if the sampled users in total have num_item unique positive items
        set_pos_items_sampled_user = set()
        for pos_items in pos_item_sampled_user:
            set_pos_items_sampled_user.update(pos_items.tolist())

        num_set_pos_items_sampled_user = len(set_pos_items_sampled_user)
        flag = k >=  num_set_pos_items_sampled_user or  num_set_pos_items_sampled_user > num_item #k needs to be smaller than the number of unique items, otherwise there is only 1 possible recommendation
        iter += 1

    if num_item == k+1 and num_user == 2 and iter > max_iter:
        raise Exception # if Exception, manually adjust the constant added to num_items to get more top_items above

    if iter > max_iter:
        print("Max iter reached, repeat finding sample")
        if num_user > 2:
            num_user -=1
        continue
    else:
        num_sample +=1


    # check if all positive items of the sampled users are in the top items
    for pos_items in pos_item_sampled_user:
        assert set(pos_items.tolist()).issubset(top_items.keys())

    # save the sampled users etc for generation Pareto with own algo
    print(f"Found a suitable sample: {user_sample}")
    with open(f"sampled/small_{data}_test_{num_sample}.pickle", "wb") as f:
        pickle.dump(pos_item_sampled_user, f, pickle.HIGHEST_PROTOCOL)
    
    with open(f"sampled/small_{data}_test_{num_sample}_userid.pickle", "wb") as f:
        pickle.dump(user_sample, f, pickle.HIGHEST_PROTOCOL)

    sample_set.add(string_list_user)


    all_possible_rec_per_user = combinations(set_pos_items_sampled_user, k)
    all_possible_rec_per_user = list(all_possible_rec_per_user)


    config = Config(
        model="Pop", 
        dataset="new_"+data, 
        config_file_list=["RecBole/recbole/properties/overall.yaml"],

        config_dict={"topk": k, 
                    "metrics":[
                            "NDCG",
                            "Ent"
                        ]})

    evaluator = Evaluator(config)
    save_every = 100_000
    new_struct = deepcopy(struct)

    new_struct["data.num_items"] = num_set_pos_items_sampled_user+1
    del new_struct["rec.score"]
    del new_struct["data.pos_items"]

    num_pos_item = np.zeros_like(pos_item_sampled_user, dtype=int)
    for u, pos_item in enumerate(pos_item_sampled_user): 
        num_pos_item[u] = len(pos_item)

    num_pos_item = torch.from_numpy(num_pos_item)

    list_scores = []

    start_time = time.time()

    print("Starting bruteforce")
    for i, rec in tqdm(enumerate(product(all_possible_rec_per_user, repeat=num_user))):

        # get relevance values
        rel = np.array([np.isin(rec[u], pos_item_sampled_user[u], assume_unique=True) for u in range(pos_item_sampled_user.size)], dtype=int) 
        rel = torch.from_numpy(rel)

        # place relevant items in front
        rel, indices = rel.sort(dim=1, descending=True) # sort by decreasing relevance

        rec = torch.asarray(rec)
        rec = torch.gather(rec, dim=1,index=indices)

        # update struct with rec.items and rec.topk 
        new_struct.set("rec.items", rec)

        rec_topk = torch.cat([rel[:,:k],num_pos_item.unsqueeze(1)], dim=1) #concat with the number of rel items
        new_struct.set("rec.topk", rec_topk)

        # eval
        eval_results = evaluator.evaluate(new_struct)
        list_scores.append(eval_results)

        if i % save_every == 0 and i!=0:

            with open(f"experiments/brute_force/scores_{model}_{data}_user{num_user}_item{num_set_pos_items_sampled_user}_k{k}_combin{i}_{num_sample}.pkl", "wb") as f:
                pickle.dump(list_scores, f, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"Time taken: {time.time() - start_time:.2f} seconds")
            list_scores = []

            start_time = time.time()
        
    if save_every > i:
        
        with open(f"experiments/brute_force/scores_{model}_{data}_user{num_user}_item{num_set_pos_items_sampled_user}_k{k}_{num_sample}.pkl", "wb") as f:
            pickle.dump(list_scores, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Time taken: {time.time() - start_time:.2f} seconds")

    elif len(list_scores) > 0:
        with open(f"experiments/brute_force/scores_{model}_{data}_user{num_user}_item{num_set_pos_items_sampled_user}_k{k}_combin{i}_{num_sample}.pkl", "wb") as f:
            pickle.dump(list_scores, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Time taken: {time.time() - start_time:.2f} seconds")
        list_scores = []

        start_time = time.time()