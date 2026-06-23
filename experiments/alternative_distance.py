from artifcial_recommendation_DPFR import *
from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter("ignore")


list_dataset = [
                "Lastfm",
                "Amazon-lb", 
                "QK-video",
                "Jester",
                "ML-10M",
                "ML-20M"
                ]

rel_measures = ["P@10", "MAP@10", "R@10", "NDCG@10"]
fair_measures = ["Jain_our@10", "Ent_our@10", "Gini_our@10"]

list_dist_name = ["manhattan", "euclidean"]

plt.rcParams['figure.dpi'] = 400
plt.rcParams['savefig.dpi'] = 600


def distance_based_rank_for_corr(model_distance_dict, data):
    rank_based_on_distance = model_distance_dict[data].unstack().reset_index()
    rank_based_on_distance.columns = ["rel","fair","models"]
    rank_based_on_distance = rank_based_on_distance.loc[rank_based_on_distance.rel.str.contains("^P|^R|NDCG|MAP")]
    rank_based_on_distance = rank_based_on_distance.loc[rank_based_on_distance.fair.str.contains("Jain|Gini|Ent")]
    rank_based_on_distance = rank_based_on_distance.loc[rank_based_on_distance.fair.str.contains("our")]
    rank_based_on_distance = rank_based_on_distance.loc[rank_based_on_distance.models.apply(lambda x: x[1]).dropna().index]
    rank_based_on_distance["col_name"] = rank_based_on_distance.rel + "-" + rank_based_on_distance.fair
    rank_based_on_distance = rank_based_on_distance[["col_name","models"]].T
    rank_based_on_distance.columns = rank_based_on_distance.loc["col_name"]
    rank_based_on_distance = rank_based_on_distance.iloc[1].T
    
    dict_rank_based_on_distance = {}

    for row, item in pd.DataFrame(rank_based_on_distance).iterrows():
        the_tup = item[0]
        model_name = the_tup[0]
        scores = the_tup[1]
        dict_rank_based_on_distance[row] = dict((key,val) for key,val in zip(model_name, scores))


    for_corr = pd.DataFrame(dict_rank_based_on_distance).T.applymap(lambda x: -x)

    return for_corr, dict_rank_based_on_distance


def get_avg(this_data):

    for_val = this_data.loc[this_data.source!="pareto"]
    for_val_rel = for_val[rel_measures]
    for_val_fair = for_val[fair_measures]
    for_val_fair.loc[:,for_val_fair.columns.str.contains("Gini")] = 1 - for_val_fair.loc[:,for_val_fair.columns.str.contains("Gini")]

    df_average = pd.DataFrame(columns=["rel", "fair", "score", "source"])

    for col in for_val_fair.columns:
        avg_val_for_col = (for_val_rel.values + for_val_fair[col].values.reshape(-1,1))/2
        df_avg_col = pd.DataFrame(avg_val_for_col, columns=rel_measures)
        df_avg_col["source"] = for_val.source.values
        df_avg_col["fair"] = col
        melted = df_avg_col.melt(["fair", "source"], var_name="rel", value_name="score")
        df_average = pd.concat([df_average, melted])

    return df_average


def plot_corr_heatmap(combined_df, model_distance_dict, dist_name, model_scores):
    fig, ax = plt.subplots(nrows=3,ncols=2, figsize=(12,10), sharex=False, sharey=False)

    i=0

    dict_for_corr = {}
    dict_wo_avg = {}

    for ax_id, data in zip(ax.flatten(), list_dataset):

        model_scores_for_data = model_scores.query("dataset==@data")

        for_corr = model_scores_for_data.loc[:,model_scores_for_data.columns.str.contains("AI|II|IAA|IBO|MME|source")]
        for_corr = for_corr.loc[:, ~for_corr.columns.str.contains("IBO_ori|IWO_ori")]
        for_corr = for_corr.T
        for_corr.columns = for_corr.loc["source"]
        for_corr.drop(index=["source"], inplace=True)

        for_corr.loc[for_corr.index.str.contains("AI|IAA|II|MME")] = for_corr.loc[for_corr.index.str.contains("AI|IAA|II|MME")].apply(lambda x: -x)

        to_append, _ = distance_based_rank_for_corr(model_distance_dict, data)
        for_corr_appended = pd.concat([for_corr, to_append])

        #avg = higher score is better, so no need to invert
        this_data = combined_df.query("dataset==@data")
        avg = get_avg(this_data)
        avg["rel_fair"] = avg["rel"] + "-" + avg["fair"]
        avg = avg.drop(columns=["rel", "fair"])
        avg = avg.set_index("rel_fair")

        avg.index = avg.index.str.replace("@10","") + "-avg"
        avg_to_append = avg.pivot(columns="source", values="score")
        for_corr_appended = pd.concat([for_corr_appended,avg_to_append])


        #plotting
        to_plot = for_corr_appended.T.reset_index(drop=True).applymap(float).corr(method="kendall").round(2)
        to_plot.dropna(how="all",inplace=True, axis=1)
        to_plot.dropna(how="all",inplace=True, axis=0)
        to_plot.columns = to_plot.columns.str.replace("@10", "")
        to_plot.index = to_plot.index.str.replace("@10","")

        #do the indexing automatically
        idx_index = to_plot.columns.tolist().index("MME_ori") +1
        idx_col = to_plot.index.tolist().index("MME_ori") +1


        filtered = to_plot.iloc[idx_index:, :idx_col]
        wo_avg = filtered[~filtered.index.str.contains("avg")]
        wo_avg["avg"] = pd.Series()
        only_avg = to_plot.loc[~to_plot.index.str.contains("AI|IAA|II|IBO|MME"),to_plot.columns.str.contains("avg")]

        for pair in wo_avg.index:
            wo_avg.loc[pair, "avg"] = only_avg.loc[pair, pair+"-avg"]

        wo_avg = wo_avg.T
        wo_avg.index = wo_avg.index\
                                .str.replace("_ori","")\
                                .str.replace("_our","")\
                                .str.replace("_true","")
        wo_avg.columns = wo_avg.columns\
                                .str.replace("_ori","")\
                                .str.replace("_our","")\

        wo_avg = wo_avg.loc[["IBO", "MME", "IAA", "II-F", "AI-F", "avg"]]

        wo_avg = wo_avg.astype(float)
        
        sns.heatmap(wo_avg,
                    annot=True,
                    cmap="coolwarm_r",
                    vmin = -1,
                    vmax = 1,
                    square = False,
                    ax = ax_id,
                    cbar = True,
                    # cbar = i % 2==1,
                    annot_kws={"size": 9},
                    cbar_kws={"shrink": 0.875}
                    )
        ax_id.set_yticklabels(ax_id.get_yticklabels(), rotation=0)

        ax_id.set_title(f"{data}")
        ax_id.set_xlabel(f"DPFR ({dist_name.title()})")
        ax_id.set_ylabel("JOINT (existing)")
        i+=1

        dict_for_corr[data] = for_corr_appended
        dict_wo_avg[data] = wo_avg

    plt.tight_layout(h_pad=3, w_pad=2)

    # get date and timenow
    timenow = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'experiments/corr/corr_heatmap_all_grid_{dist_name}_{timenow}.pdf', bbox_inches="tight")
    plt.close()

    return dict_for_corr, dict_wo_avg

def main():
    path_integral_point_full = pd.read_pickle("experiments/artificial/path_integral_point_full.pickle") #this is not artificial Pareto Frontier, just that the file is placed there
    combined_df_full = pd.read_csv("experiments/corr/combined_df_full.csv")

    model_distance_dict_euclid = get_model_distance_dict(combined_df_full, path_integral_point_full, dist_measure="euclidean")
    model_distance_dict_manhattan = get_model_distance_dict(combined_df_full, path_integral_point_full, dist_measure="manhattan")

    model_scores = pd.read_csv(f"experiments/corr/model_scores.csv", index_col=0)

    list_model_distance_dict = [model_distance_dict_manhattan, model_distance_dict_euclid]

    dict_raw_scores = {data: {} for data in list_dataset}
    dict_avg_corr = {data: {} for data in list_dataset}

    for (dist_name, distance_dict) in zip(list_dist_name, list_model_distance_dict):
        
        # === Plot corr heatmap between joint measures and DPFR, separately for euclidean and manhattan ===
        # Note: the euclidean one produces the same heatmap as replot_heatmap
        dict_for_corr, dict_wo_avg = plot_corr_heatmap(combined_df_full, distance_dict, dist_name, model_scores)
        
        for data in list_dataset:
            raw_scores = dict_for_corr[data]
        
            raw_scores.index = raw_scores.index\
                                    .str.replace("_our","")\
                                    .str.replace("@10","")
            raw_scores = raw_scores[~raw_scores.index.str.contains("IBO|MME|IAA|II|AI")]
            raw_scores.index = raw_scores.index + f"-{dist_name}"

            dict_raw_scores[data][dist_name] = raw_scores

            dict_avg_corr[data][dist_name] = dict_wo_avg[data].loc["avg"]

    # === Get corr between euclid vs manhattan ===

    all_data_euclid_vs_manhattan = pd.DataFrame()

    dict_all_dataset_raw_scores = dict.fromkeys(list_dataset)

    for data in list_dataset:

        # === DPFR euclid vs DPRR manhattan === 
        df_all_raw_scores = pd.concat(dict_raw_scores[data].values())
        dict_all_dataset_raw_scores[data] = df_all_raw_scores

        all_corr = df_all_raw_scores.T.reset_index(drop=True).applymap(float).corr(method="kendall").round(2)

        # filter: rows = euclidean, cols = manhattan
        filtered = all_corr.loc[all_corr.index.str.contains("euclidean"), all_corr.columns.str.contains("manhattan")]
        wo_avg = filtered.loc[~filtered.index.str.contains("avg"), ~filtered.columns.str.contains("avg")]
        
        euclid_vs_manhattan = pd.DataFrame(
                                    np.diag(wo_avg), index=[wo_avg.index])\
                                        .reset_index()
        euclid_vs_manhattan.columns = ["pair", "euclid_vs_manhattan"]
        
        euclid_vs_manhattan["pair"] = euclid_vs_manhattan["pair"].str.replace("-euclidean","")
        euclid_vs_manhattan = euclid_vs_manhattan.set_index("pair")


        # === DPFR euclid vs avg === 
        euclid_vs_avg = dict_avg_corr[data]["euclidean"]
        euclid_vs_manhattan = pd.concat([euclid_vs_manhattan, euclid_vs_avg], axis=1)\
                                                        .rename(columns={"avg":"euclidean_vs_avg"})
        
        # === DPFR manhattan vs avg ===
        manhattan_vs_avg = dict_avg_corr[data]["manhattan"]
        euclid_vs_manhattan = pd.concat([euclid_vs_manhattan, manhattan_vs_avg], axis=1)\
                                                        .rename(columns={"avg":"manhattan_vs_avg"}) 

        euclid_vs_manhattan["dataset"] = data

        all_data_euclid_vs_manhattan = pd.concat([all_data_euclid_vs_manhattan, euclid_vs_manhattan])
    
    # save the raw DPFR scores for euclidean and manhattan in a pickle file, for later use in Jupyter notebook
    with open(f"experiments/combined_base/raw_scores.pickle", "wb") as f:
        pickle.dump(dict_all_dataset_raw_scores, f, pickle.HIGHEST_PROTOCOL)

    # === Plot heatmap for euclid vs manhattan, euclid vs avg, manhattan vs avg === 
    all_data_euclid_vs_manhattan = all_data_euclid_vs_manhattan\
                                                        .reset_index()\
                                                        .set_index(["index", "dataset"])\
                                                        .unstack()
    
    # separate heatmap per correlation type instead, with dataset as rows 
    fig, ax = plt.subplots(nrows=3,ncols=1, figsize=(12,10), sharex=False, sharey=False)

    corr_type = all_data_euclid_vs_manhattan\
                                        .columns\
                                        .get_level_values(0)\
                                        .unique()
    

    pair_order = wo_avg.index.str.replace("-euclidean","").to_list()

    for i, col_name in enumerate(corr_type):

        ax_id = ax[i]
        sns.heatmap(all_data_euclid_vs_manhattan[col_name]\
                                                        .T\
                                                        .loc[list_dataset, pair_order],
                    annot=True,
                    cmap="coolwarm_r",
                    vmin = -1,
                    vmax = 1,
                    square = False,
                    ax = ax_id,
                    cbar = True,
                    annot_kws={"size": 9},
                    cbar_kws={"shrink": 0.885}
                    )
        ax_id.set_yticklabels(ax_id.get_yticklabels(), rotation=0)

        clean_col_name = col_name\
                                .replace("euclid_vs_manhattan", "DPFR (Euclidean) and DPFR (Manhattan)")\
                                .replace("euclidean_vs_avg", "DPFR (Euclidean) and avg")\
                                .replace("manhattan_vs_avg", "DPFR (Manhattan) and avg")
        
        ax_id.set_title(f"Correlation (Kendall's $\\tau$) between {clean_col_name} for different datasets and measure pairs")
        ax_id.set_xlabel("measure pairs")

    plt.tight_layout(h_pad=3, w_pad=2)

    # get date and timenow
    timenow = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'experiments/corr/corr_heatmap_euclid_vs_manhattan_{timenow}.pdf', bbox_inches="tight")
    

if __name__ == "__main__":
    main()