# 📈 Robustness Analysis of Pareto-based Joint Evaluation of Fairness and Relevance in Recommender Systems ⚖

This repository contains the code for the _extra_ experiments and analyses in our work on "Robustness Analysis of Pareto-based Joint Evaluation of Fairness and Relevance in Recommender Systems", which is currently under review (single-anonymous).

This work extends the WWW'25 full paper "Joint Evaluation of Fairness and Relevance in Recommender Systems with Pareto Frontier" by Theresia Veronika Rampisela, Tuukka Ruotsalo, Maria Maistro, and Christina Lioma. The code for the original experiments is available [here](https://github.com/theresiavr/DPFR-recsys-evaluation).

Links to the WWW'25 paper and poster:

[[ACM]](https://doi.org/10.1145/3696410.3714589) [[arXiv]](https://arxiv.org/abs/2502.11921) [[poster]](https://theresiavr.github.io/assets/pdf/thewebconf25-DPFR-recsys-evaluation-poster.pdf)

## Datasets, Model, Evaluation, and Experiments in the Conference Paper

Please refer to the [code repository of the WWW'25 paper](https://github.com/theresiavr/DPFR-recsys-evaluation) to find information on dataset downloads (including links to download them), preprocessing, model training, hyperparameter tuning, reranking, evaluation, and the experiment code for the conference paper.

## Experiments and Analyses in the Extension Paper

The `experiments/` folder contains additional analyses:

### Closeness to the True Pareto Frontier

Compare the Pareto Frontier (PF) generated with our Oracle2Fair algorithm to the true PF, obtained through brute force (exhaustive enumeration)

- `brute_force_feasibility.ipynb` - Analyse feasibility of the brute force approach; determines setup in the next step
- `brute_force_pareto.py` - Sample a subset of each dataset and generate the true PF through exhaustive enumeration
- `generate_pareto_small.py` - Generate a PF for each sample with our Oracle2Fair algorithm, to be used for the closeness comparison
- `brute_force_vs_our_closeness.ipynb` - Compare how close the generated PF is to the true one with Generational Distance

### Artificial Recommendation

Evaluate DPFR robustness against artificially-generated recommendations:

- `artificial_recommendation.py` - Generate artificial recommendations
- `artificial_recommendation_DPFR.py` - DPFR evaluation on artificial recommendations
- `artificial_recommendation_plot.ipynb` - Visualisation of artificial recommendation results
- `artificial_recommendation_best_model_disagreement.ipynb` - Analyse disagreements between avg and DPFR on which artificial recommendation is the 'best'

### Alternative Distance Measure

Evaluate the effect of using alternative distance metric for computing DPFR (i.e., changing Euclidean distance to Manhattan):

- `alternative_distance.py` - Computes DPFR with Manhattan distance and plot the correlation between DPFR (Euclidean), DPFR (Manhattan), and other existing joint evaluation approaches

### Fairness-Relevance Trade-off

Evaluate robustness to the change in fairness-relevance trade-off weight in weighted average and in DPFR

- `corr_weighted_avg.py` - Correlation analysis with various weights for weighted average
- `corr_varying_alpha.ipynb` - Correlation analysis with varying alpha parameter (fairness-relevance trade-off weight) in DPFR

### Correlation to Simple Measures

- `simple_measure.ipynb` - Compute simple measures and their correlation with single-aspect and joint evaluation approaches, including DPFR
- `corr_weighted_avg_simple_measure.py` - Compute the correlation of avg to simple measures across various weights
- `simple_measure_compare_avg_DPFR.ipynb` - Compare avg's and DPFR's correlation to simple measures
- `simple_measure_plot.ipynb` - Visualise correlation to simple measures of fairness/relevance across varying alpha

### Other

- `replot_heatmap.ipynb` - Re-plot correlation heatmap visualisations to better fit journal layout

## License and Terms of Usage

The code is usable under the MIT License.

## Citation

```BibTeX
@inproceedings{Rampisela2025Pareto,
author = {Rampisela, Theresia Veronika and Ruotsalo, Tuukka and Maistro, Maria and Lioma, Christina},
title = {Joint Evaluation of Fairness and Relevance in Recommender Systems with Pareto Frontier},
year = {2025},
isbn = {9798400712746},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3696410.3714589},
doi = {10.1145/3696410.3714589},
pages = {1548–1566},
numpages = {19},
keywords = {evaluation, fairness, pareto frontier, recommendation, relevance},
location = {Sydney NSW, Australia},
series = {WWW '25}
}
```
