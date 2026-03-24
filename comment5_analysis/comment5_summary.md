# Comment 5 analysis summary

## Dataset assembly
| dataset                   |   n_rows |   n_unique_cid |   n_unique_raw_smiles |   n_unique_canonical_smiles |   n_label_pos |   n_label_neg |
|:--------------------------|---------:|---------------:|----------------------:|----------------------------:|--------------:|--------------:|
| internal_compound_curated |     6958 |           6905 |                  6958 |                        6958 |           766 |          6192 |
| internal_graph_augmented  |    10691 |           6878 |                 10612 |                        6931 |          4512 |          6179 |
| external_compound_assay   |     4002 |           4002 |                  4002 |                        4002 |           592 |          3410 |
| external_graph_graphable  |     3999 |           3999 |                  3999 |                        3999 |           592 |          3407 |

## External overlap summary
```json
{
  "strict_overlap_rule": "cid_or_canonical",
  "external_compound_n": 4002,
  "external_compound_overlap_cid_n": 2261,
  "external_compound_overlap_raw_smiles_n": 0,
  "external_compound_overlap_canonical_smiles_n": 1897,
  "external_compound_strict_keep_n": 1684,
  "external_graph_n": 3999,
  "external_graph_overlap_cid_n": 2259,
  "external_graph_overlap_raw_smiles_n": 0,
  "external_graph_overlap_canonical_smiles_n": 1897,
  "external_graph_strict_keep_n": 1683
}
```

## External similarity summary
| analysis_set                  |    n |     mean |   median |       q1 |       q3 |     min |   max |   exact_1.0_n |   exact_1.0_pct |
|:------------------------------|-----:|---------:|---------:|---------:|---------:|--------:|------:|--------------:|----------------:|
| external_original_vs_internal | 3999 | 0.811025 | 1        | 0.583974 | 1        | 0.15311 |     1 |          2004 |        50.1125  |
| external_strict_vs_internal   | 1683 | 0.569518 | 0.529412 | 0.396942 | 0.708333 | 0.15311 |     1 |            93 |         5.52585 |

## M2 fold-wise similarity
|    n |     mean |   median |       q1 |       q3 |       min |   max |   exact_1.0_n |   exact_1.0_pct | split            |   fold |   n_train_rows |   n_test_rows |   n_train_unique_canonical |   n_test_unique_canonical |
|-----:|---------:|---------:|---------:|---------:|----------:|------:|--------------:|----------------:|:-----------------|-------:|---------------:|--------------:|---------------------------:|--------------------------:|
| 3489 | 0.375188 | 0.347826 | 0.272727 | 0.45     | 0.0526316 |     1 |             6 |        0.171969 | M2_scaffold5fold |      0 |           7202 |          3489 |                       4402 |                      2529 |
| 1320 | 0.446212 | 0.416667 | 0.32742  | 0.545455 | 0.166667  |     1 |             8 |        0.606061 | M2_scaffold5fold |      1 |           9371 |          1320 |                       6161 |                       770 |
| 2996 | 0.448894 | 0.434783 | 0.357143 | 0.52381  | 0.176471  |     1 |             5 |        0.166889 | M2_scaffold5fold |      2 |           7695 |          2996 |                       4855 |                      2076 |
| 1574 | 0.488489 | 0.475    | 0.382979 | 0.581208 | 0.181818  |     1 |             3 |        0.190597 | M2_scaffold5fold |      3 |           9117 |          1574 |                       6112 |                       819 |
| 1312 | 0.458262 | 0.433333 | 0.347826 | 0.555556 | 0.194444  |     1 |             3 |        0.228659 | M2_scaffold5fold |      4 |           9379 |          1312 |                       6194 |                       737 |

## M4 fold-wise similarity
|    n |     mean |   median |       q1 |       q3 |       min |      max |   exact_1.0_n |   exact_1.0_pct | split          |   fold |   n_train_rows |   n_test_rows |   n_train_unique_canonical |   n_test_unique_canonical |
|-----:|---------:|---------:|---------:|---------:|----------:|---------:|--------------:|----------------:|:---------------|-------:|---------------:|--------------:|---------------------------:|--------------------------:|
| 2202 | 0.510805 | 0.52381  | 0.421687 | 0.615385 | 0.0833333 | 0.952381 |             0 |               0 | M4_butina_lco5 |      0 |           8489 |          2202 |                       5544 |                      1387 |
| 2206 | 0.506223 | 0.517241 | 0.410526 | 0.615385 | 0.142857  | 0.962963 |             0 |               0 | M4_butina_lco5 |      1 |           8485 |          2206 |                       5545 |                      1386 |
| 2096 | 0.509281 | 0.522233 | 0.415312 | 0.611111 | 0.142857  | 0.952381 |             0 |               0 | M4_butina_lco5 |      2 |           8595 |          2096 |                       5545 |                      1386 |
| 2101 | 0.509066 | 0.529412 | 0.414286 | 0.619048 | 0.142857  | 0.962963 |             0 |               0 | M4_butina_lco5 |      3 |           8590 |          2101 |                       5545 |                      1386 |
| 2086 | 0.512545 | 0.535714 | 0.412698 | 0.619048 | 0.125     | 0.947368 |             0 |               0 | M4_butina_lco5 |      4 |           8605 |          2086 |                       5545 |                      1386 |

## External prediction metrics (existing prediction file filtered to strict subset)
|    n |   n_pos |   n_neg |   threshold |   roc_auc |   pr_auc |   precision |   recall |   specificity |       f1 |   tp |   tn |   fp |   fn | analysis_set                 |
|-----:|--------:|--------:|------------:|----------:|---------:|------------:|---------:|--------------:|---------:|-----:|-----:|-----:|-----:|:-----------------------------|
| 3999 |     592 |    3407 |    0.5      |  0.840007 | 0.440641 |    0.254641 | 0.903716 |      0.540358 | 0.397326 |  535 | 1841 | 1566 |   57 | external_original_thr0.5     |
| 3999 |     592 |    3407 |    0.734886 |  0.840007 | 0.440641 |    0.465823 | 0.621622 |      0.876137 | 0.532562 |  368 | 2985 |  422 |  224 | external_original_thr_bestF1 |
| 1683 |     407 |    1276 |    0.5      |  0.794336 | 0.490022 |    0.327511 | 0.921376 |      0.396552 | 0.483247 |  375 |  506 |  770 |   32 | external_strict_thr0.5       |
| 1683 |     407 |    1276 |    0.735586 |  0.794336 | 0.490022 |    0.517442 | 0.65602  |      0.804859 | 0.578548 |  267 | 1027 |  249 |  140 | external_strict_thr_bestF1   |

ROC figure: `external_roc_curve_original_vs_strict.png`
PR figure: `external_pr_curve_original_vs_strict.png`

Strict external graph file: `data_graph_external_strict.pt`

