# Code to get the ranking of artificial runs based on DPFR (from best/closest to worst/furthest)

# Instruction:
# 1. Get path_integral_point_full.pickle from conference paper code
# 2. Save it under "experiments/artificial/" folder
import pickle
import pandas as pd
import numpy as np

import os, glob

rel_measures = ["P@10", "MAP@10", "R@10", "NDCG@10"]
fair_measures = ["Jain_our@10", "Ent_our@10", "Gini_our@10"]

list_dataset = [
                "Lastfm",
                "Amazon-lb", 
                "QK-video",
                "Jester",
                "ML-10M",
                "ML-20M"
                ]

def get_model_distance(df, path_integral_point, return_all_dist=False):
    best_model = {}
    for rel_measure in rel_measures:
        best_model[rel_measure] = {}
        for fair_measure in fair_measures:
            df1 = df[[rel_measure, fair_measure]]

            #check na column:
            df1 = df1.dropna()
            df2 = path_integral_point.loc[fair_measure, rel_measure]

            if len(df1)==0:
                best_model[rel_measure][fair_measure] = ("-", np.nan)
                continue


            dist = np.linalg.norm(df1-df2, axis=1)
            if not return_all_dist:
                idx = dist.argmin()
                best_model[rel_measure][fair_measure] = (df.iloc[idx]["source"], df1.iloc[idx].values)
            else:
                best_model[rel_measure][fair_measure] = (df.iloc[np.argsort(dist, kind="stable")]["source"].values, np.sort(dist, kind="stable"))
                
    best_model = pd.DataFrame(best_model)
    return best_model

def get_model_distance_dict(combined_df, path_integral_point):
    selected_merged = combined_df\
                            .query("source=='pareto'")\
                            .drop(columns="source")

    model_distance_dict = {}

    for data in combined_df.dataset.unique():
        print(data)
        this_data = selected_merged.query("dataset==@data")
        path_integral_point_data = path_integral_point[data]

        if len(this_data) == 0:
            continue


        model_distance = get_model_distance(
                combined_df.query("dataset==@data & source!='pareto'"), path_integral_point_data, return_all_dist=True
                )

        model_distance_dict[data] = model_distance
    return model_distance_dict

def combine_df_full_with_artificial(combined_df_full):
    selected_cols = combined_df_full.columns

    for data in list_dataset:
        list_files = glob.glob(f"artificial/result_{data}*.pickle")
        for file in list_files:
            res = pd.read_pickle(file)
            to_append = pd.DataFrame\
                                    .from_dict(res, orient="index")\
                                    .T
            to_append[["dataset","source"]] = file\
                                                .replace("artificial\\result_","")\
                                                .replace(".pickle","")\
                                                .split("_")
            to_append["source"] = "artificial-" + to_append["source"] 
            to_append = to_append.loc[:, selected_cols]

            combined_df_full = pd.concat([combined_df_full, to_append])
    return combined_df_full

def main():
    os.chdir("experiments")
    path_integral_point_full = pd.read_pickle("artificial/path_integral_point_full.pickle")
    
    combined_df_full = pd.read_csv("corr/combined_df_full.csv")
    combined_df_full = combined_df_full.query("source=='pareto'")

    combined_df_full = combine_df_full_with_artificial(combined_df_full)
    combined_df_full.to_csv("artificial/combined_df_full_artificial.csv", index=False)
    
    model_distance_dict_artificial = get_model_distance_dict(combined_df_full, path_integral_point_full)

    with open(f"artificial/artificial_model_distance_dict_full.pickle","wb") as f:
        pickle.dump(
            model_distance_dict_artificial, 
            f, 
            pickle.HIGHEST_PROTOCOL
    )
        
if __name__ == "__main__":
    main()