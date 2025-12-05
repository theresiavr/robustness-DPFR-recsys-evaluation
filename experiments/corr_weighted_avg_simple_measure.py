import pandas as pd
import os

from corr_weighted_avg import *

experiment_name = "simple_measure"

list_dataset = [
                "Lastfm",
                "Amazon-lb", 
                "QK-video",
                "Jester",
                "ML-10M",
                "ML-20M"
                ]



def main():
    os.chdir("experiments")
    df_simple = pd.read_csv(f"{experiment_name}/simple_measure_results.csv")

    for data in list_dataset:
        print(data)

        # higher is better
        df_simple_data = df_simple\
                                .query("dataset==@data")\
                                .drop(columns="dataset")\
                                .set_index("source")
        
        
        df_wavg = pd.DataFrame(columns=["num_rel_items", "num_unique_rec_items"])

        model_scores = load_model_scores(newest_file)
        model_scores_for_data = get_model_scores_for_data(model_scores, data)
        model_scores_for_data = transform_gini_score(model_scores_for_data)

        for i in range(101):
            alpha = i/100
            print(f"Calculating for alpha={alpha}")

            for rel_m in ["P", "MAP", "R", "NDCG"]:
                for fair_m in ["Jain", "Ent", "Gini"]:
                    #higher is better
                    weighted_avg = (1-alpha) *  model_scores_for_data[rel_m] + alpha * model_scores_for_data[fair_m]

                    combined_simple_wavg = pd.concat([df_simple_data, weighted_avg], axis=1)
                    combined_simple_wavg = combined_simple_wavg.rename(columns={0:"wavg"})

                    corr_val = combined_simple_wavg.corr("kendall").round(2)

                    corr_to_add = corr_val.loc[["wavg"], ["num_rel_items", "num_unique_rec_items"]]
                    corr_to_add.index = [f"{rel_m}-{fair_m}-wavg-{alpha}"]

                    df_wavg = pd.concat([df_wavg, corr_to_add])
        
        df_wavg.to_csv(f"{experiment_name}/corr_simple_measure_wavg_{data}.csv")


if __name__ == "__main__":
    main()