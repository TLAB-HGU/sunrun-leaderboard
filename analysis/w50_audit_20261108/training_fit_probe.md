# Retained-parent training fit probe

- Fixed seed42 descriptive inference; no fitting, candidate selection, or query scoring.
- Full models trained on every analyzed row; reference/calibration/offset subsets are all in-sample.
- Fixed-prefix trends are not future-generalization evidence or proof of a 50% improvement.
- This cannot separate model-class capacity from target noise or missing input information.
- Originally observed versus imputed labels describe the retained training target policy only.
- Spectral loss couples a complete 24h head; no spectral allocation to target bins or label masks is claimed.
- Source equality is against available saved snapshots; byte/time receipts do not prove immutable historical identity.

| Period | Head | Prefix | Full raw RMSE | Point L2/pair | Spectral/pair |
| --- | --- | ---: | ---: | ---: | ---: |
| F1 | 0 | 300 | 44.284790 | 1190.586395 | 44.588779 |
| F1 | 0 | 600 | 41.358325 | 987.347844 | 41.019556 |
| F1 | 0 | 900 | 39.031235 | 852.478369 | 38.287467 |
| F1 | 1 | 300 | 62.793425 | 2281.998932 | 64.159185 |
| F1 | 1 | 600 | 57.016998 | 1814.176472 | 58.063649 |
| F1 | 1 | 900 | 52.994590 | 1523.525859 | 53.785672 |
| F1 | 2 | 300 | 70.884037 | 2923.307715 | 71.812263 |
| F1 | 2 | 600 | 63.631095 | 2269.315874 | 64.619867 |
| F1 | 2 | 900 | 58.466071 | 1863.819308 | 59.434007 |
| F2 | 0 | 300 | 45.504564 | 1238.877974 | 45.834036 |
| F2 | 0 | 600 | 42.560474 | 1034.637152 | 42.286953 |
| F2 | 0 | 900 | 40.254374 | 900.751776 | 39.633582 |
| F2 | 1 | 300 | 65.004276 | 2430.582904 | 66.697279 |
| F2 | 1 | 600 | 59.535965 | 1977.393985 | 60.939246 |
| F2 | 1 | 900 | 55.475162 | 1668.675658 | 56.631035 |
| F2 | 2 | 300 | 73.064240 | 3045.058887 | 74.333330 |
| F2 | 2 | 600 | 65.804403 | 2380.811501 | 67.065304 |
| F2 | 2 | 900 | 60.791056 | 1977.235479 | 61.994600 |
| F3 | 0 | 300 | 45.441195 | 1240.061477 | 45.786254 |
| F3 | 0 | 600 | 42.678566 | 1048.594455 | 42.475718 |
| F3 | 0 | 900 | 40.446603 | 916.029223 | 39.919808 |
| F3 | 1 | 300 | 64.871902 | 2427.295520 | 66.464334 |
| F3 | 1 | 600 | 59.768898 | 1990.468832 | 61.083996 |
| F3 | 1 | 900 | 55.910101 | 1694.366697 | 56.970123 |
| F3 | 2 | 300 | 72.587433 | 3009.758837 | 73.777951 |
| F3 | 2 | 600 | 65.566512 | 2370.855823 | 66.683211 |
| F3 | 2 | 900 | 60.537573 | 1963.717953 | 61.576802 |

The JSON contains every fixed speed bin, original-label mask, and original row subset.
Tail weights are normalized once across the full head training population, then retained for subset reports.
Spectral potential is averaged over Fourier coefficients/pairs of whole 24h heads; its Hessian is not the training surrogate Hessian.
