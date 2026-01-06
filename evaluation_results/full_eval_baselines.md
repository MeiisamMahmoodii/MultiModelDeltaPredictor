# Baseline Evaluation

## Per-Graph Results

| Model   |   N |   Graph |   SHD |   SID |          F1 |   MAE |   MSE |   RMSE |   R2 |       Time_s | Error           |
|:--------|----:|--------:|------:|------:|------------:|------:|------:|-------:|-----:|-------------:|:----------------|
| PC      |  10 |       0 |    15 |    70 |   0.545455  |   nan |   nan |    nan |  nan |   0.0157962  | nan             |
| NOTEARS |  10 |       0 |    14 |    47 |   0.363636  |   nan |   nan |    nan |  nan |   0.511104   | nan             |
| PC      |  10 |       1 |    11 |    53 |   0.521739  |   nan |   nan |    nan |  nan |   0.00753736 | nan             |
| NOTEARS |  10 |       1 |     5 |    19 |   0.666667  |   nan |   nan |    nan |  nan |   0.0787721  | nan             |
| PC      |  10 |       2 |     9 |    11 |   0.181818  |   nan |   nan |    nan |  nan |   0.00500131 | nan             |
| NOTEARS |  10 |       2 |     4 |     3 |   0         |   nan |   nan |    nan |  nan |   0.0732701  | nan             |
| PC      |  20 |       0 |    46 |   269 |   0.258065  |   nan |   nan |    nan |  nan |   0.0350878  | nan             |
| NOTEARS |  20 |       0 |    45 |   228 |   0.0425532 |   nan |   nan |    nan |  nan |   0.0825083  | nan             |
| PC      |  20 |       1 |    31 |   260 |   0.474576  |   nan |   nan |    nan |  nan |   0.0416272  | nan             |
| NOTEARS |  20 |       1 |    32 |   165 |   0.2       |   nan |   nan |    nan |  nan |   0.088618   | nan             |
| PC      |  20 |       2 |    30 |   252 |   0.482759  |   nan |   nan |    nan |  nan |   0.052875   | nan             |
| NOTEARS |  20 |       2 |    29 |   209 |   0.25641   |   nan |   nan |    nan |  nan |   0.0856059  | nan             |
| PC      |  30 |       0 |    92 |   775 |   0.394737  |   nan |   nan |    nan |  nan |   0.232295   | nan             |
| NOTEARS |  30 |       0 |    91 |   635 |   0.116505  |   nan |   nan |    nan |  nan |   0.229328   | nan             |
| PC      |  30 |       1 |    78 |   787 |   0.409091  |   nan |   nan |    nan |  nan |   0.136602   | nan             |
| NOTEARS |  30 |       1 |    74 |   518 |   0.139535  |   nan |   nan |    nan |  nan |   0.146788   | nan             |
| PC      |  30 |       2 |   105 |   754 |   0.313725  |   nan |   nan |    nan |  nan |   0.231237   | nan             |
| NOTEARS |  30 |       2 |    95 |   759 |   0.0594059 |   nan |   nan |    nan |  nan |   0.249398   | nan             |
| PC      |  40 |       0 |   nan |   nan | nan         |   nan |   nan |    nan |  nan | nan          | skipped_n_gt_30 |
| NOTEARS |  40 |       0 |   172 |  1319 |   0.0549451 |   nan |   nan |    nan |  nan |   0.10998    | nan             |
| PC      |  40 |       1 |   nan |   nan | nan         |   nan |   nan |    nan |  nan | nan          | skipped_n_gt_30 |
| NOTEARS |  40 |       1 |   191 |  1262 |   0.0103627 |   nan |   nan |    nan |  nan |   0.109908   | nan             |
| PC      |  40 |       2 |   nan |   nan | nan         |   nan |   nan |    nan |  nan | nan          | skipped_n_gt_30 |
| NOTEARS |  40 |       2 |   173 |  1382 |   0.0546448 |   nan |   nan |    nan |  nan |   0.109478   | nan             |
| PC      |  50 |       0 |   nan |   nan | nan         |   nan |   nan |    nan |  nan | nan          | skipped_n_gt_30 |
| NOTEARS |  50 |       0 |   248 |  2149 |   0.0387597 |   nan |   nan |    nan |  nan |   0.119009   | nan             |
| PC      |  50 |       1 |   nan |   nan | nan         |   nan |   nan |    nan |  nan | nan          | skipped_n_gt_30 |
| NOTEARS |  50 |       1 |   258 |  2207 |   0.0373134 |   nan |   nan |    nan |  nan |   0.118755   | nan             |
| PC      |  50 |       2 |   nan |   nan | nan         |   nan |   nan |    nan |  nan | nan          | skipped_n_gt_30 |
| NOTEARS |  50 |       2 |   236 |  2129 |   0.0406504 |   nan |   nan |    nan |  nan |   0.119183   | nan             |

## Summary

| Model   |   N |   Graph |       SHD |       SID |          F1 |   MAE |   MSE |   RMSE |   R2 |       Time_s |
|:--------|----:|--------:|----------:|----------:|------------:|------:|------:|-------:|-----:|-------------:|
| NOTEARS |  10 |       1 |   7.66667 |   23      |   0.343434  |   nan |   nan |    nan |  nan |   0.221049   |
| NOTEARS |  20 |       1 |  35.3333  |  200.667  |   0.166321  |   nan |   nan |    nan |  nan |   0.0855774  |
| NOTEARS |  30 |       1 |  86.6667  |  637.333  |   0.105149  |   nan |   nan |    nan |  nan |   0.208505   |
| NOTEARS |  40 |       1 | 178.667   | 1321      |   0.0399842 |   nan |   nan |    nan |  nan |   0.109788   |
| NOTEARS |  50 |       1 | 247.333   | 2161.67   |   0.0389078 |   nan |   nan |    nan |  nan |   0.118983   |
| PC      |  10 |       1 |  11.6667  |   44.6667 |   0.416337  |   nan |   nan |    nan |  nan |   0.00944495 |
| PC      |  20 |       1 |  35.6667  |  260.333  |   0.405133  |   nan |   nan |    nan |  nan |   0.0431967  |
| PC      |  30 |       1 |  91.6667  |  772      |   0.372518  |   nan |   nan |    nan |  nan |   0.200045   |
| PC      |  40 |       1 | nan       |  nan      | nan         |   nan |   nan |    nan |  nan | nan          |
| PC      |  50 |       1 | nan       |  nan      | nan         |   nan |   nan |    nan |  nan | nan          |