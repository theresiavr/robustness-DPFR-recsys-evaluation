import pandas as pd
import os

folder_name = "distance_dict"
newest_file = "2024-04-21_004417"

list_dataset = [
                "Lastfm",
                "Amazon-lb", 
                "QK-video",
                "Jester",
                "ML-10M",
                "ML-20M"
                ]

def load_model_scores(newest_file):
    model_scores = pd.read_csv(f"combined_base/csv_combined_result_{newest_file}.csv", index_col=0)

    model_scores.sort_values(["dataset","reranking"], inplace=True)
    model_scores = model_scores.melt(id_vars=["dataset", "measures", "reranking"]).set_index("measures")
    model_scores.index += "@10"
    model_scores["source"] = model_scores.apply(lambda x: x.variable + "-" + x.reranking if x.reranking != "-" else x.variable, axis=1)
    model_scores.drop(columns=["reranking", "variable"], inplace=True)

    model_scores = model_scores\
                    .reset_index()\
                    .pivot_table(index=["measures"], columns=["dataset","source"])\
                    .T\
                    .reset_index()

    model_scores.drop(columns="level_0", inplace=True)

    model_scores["dataset"] = model_scores.dataset.str.replace("\\rotatebox[origin=r]{90}{","", regex=False).str.rstrip("}")

    return model_scores

def get_model_scores_for_data(model_scores, data):

    model_scores_for_data = model_scores.query("dataset==@data")
    model_scores_for_data.drop(columns="dataset", inplace=True)
    selected_cols = ['P@10', 'MAP@10', 'R@10', 'NDCG@10', 
                     'Jain_our@10', 'Ent_our@10', 'Gini_our@10', 
                    'source']
    model_scores_for_data = model_scores_for_data[selected_cols]
    model_scores_for_data.columns = model_scores_for_data.columns.str.replace("_our|@10","",regex=True)
    model_scores_for_data = model_scores_for_data.set_index("source")
    return model_scores_for_data

def transform_gini_score(model_scores_for_data):
    # 1-Gini score
    model_scores_for_data["Gini"] = model_scores_for_data["Gini"].apply(lambda x: 1-x)
    return model_scores_for_data

def load_model_distance_dict(alpha):
    model_distance_dict_alpha = pd.read_pickle(f"{folder_name}/model_distance_dict_full-alpha-{alpha}.pickle")
    return model_distance_dict_alpha

def get_clean_model_distance_dict_data(model_distance_dict_alpha, data):
    model_distance_dict_data = model_distance_dict_alpha[data]
    model_distance_dict_data.index = model_distance_dict_data.index.str.replace("@10|_our","", regex=True)
    model_distance_dict_data.columns = model_distance_dict_data.columns.str.replace("@10","", regex=True)

    return model_distance_dict_data

def main():
    os.chdir("experiments")

    df_wavg = pd.DataFrame(columns=["DPFR", "measure", "corr", "dataset", "alpha"])
    model_scores = load_model_scores(newest_file)

    for data in list_dataset:
        model_scores_for_data = get_model_scores_for_data(model_scores, data)
        model_scores_for_data = transform_gini_score(model_scores_for_data)

        for i in range(101):
            alpha = i/100
            print(f"Calculating for alpha={alpha}")

            model_distance_dict_alpha = load_model_distance_dict(alpha)
            model_distance_dict_data = get_clean_model_distance_dict_data(model_distance_dict_alpha, data)

            for rel_m in ["P", "MAP", "R", "NDCG"]:
                for fair_m in ["Jain", "Ent", "Gini"]:
                    #higher is better
                    weighted_avg = (1-alpha) *  model_scores_for_data[rel_m] + alpha * model_scores_for_data[fair_m]

                    res_dpfr_pair = model_distance_dict_data.loc[fair_m, rel_m]
                    df_dpfr = pd.DataFrame(index=res_dpfr_pair[0], data=res_dpfr_pair[1])

                    #apply negative because DPFR is lower = better
                    df_dpfr = df_dpfr.map(lambda x:-x)

                    combined_dpfr_wavg = pd.concat([df_dpfr,weighted_avg], axis=1)
                    combined_dpfr_wavg.columns = ["dpfr", "wavg"]

                    corr_val = combined_dpfr_wavg.corr("kendall").round(2).loc["dpfr","wavg"].item()

                    res = {"DPFR":f"{rel_m}@10-{fair_m}_our@10",
                        "measure":f"{rel_m}-{fair_m}_our-wavg",
                        "corr": corr_val,
                        "dataset": f"{data}",
                        "alpha": alpha
                        }

                    to_concat = pd.DataFrame(pd.Series(res)).T
                    df_wavg = pd.concat([df_wavg, to_concat])

    df_wavg.to_csv("corr/corr_alpha_wavg.csv")

if __name__ == "__main__":
    main()