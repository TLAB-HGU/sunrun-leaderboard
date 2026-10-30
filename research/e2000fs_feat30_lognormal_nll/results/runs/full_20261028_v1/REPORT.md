# LightGBM continuous-distribution experiments

Frozen current-HPO parameters; official-exposed post-hoc diagnostics.

| Method | Stage | MSE | <550 MSE | >=550 MSE | CRPS |
|---|---|---:|---:|---:|---:|
| l2 | internal | 7416.810 | 3747.395 | 14188.978 | N/A |
| l2 | combined | 3296.695 | 2340.520 | 9401.722 | N/A |
| extreme4 | internal | 7333.916 | 5169.685 | 11328.161 | N/A |
| extreme4 | combined | 3728.771 | 3256.817 | 6742.124 | N/A |
| lognormal_nll | internal | 7698.200 | 3188.449 | 16021.271 | 45.94714071478453 |
| lognormal_nll | combined | 3359.510 | 2154.863 | 11050.992 | 30.18499313061126 |
| lognormal_crps | internal | 7554.410 | 3391.492 | 15237.374 | 46.12775455736292 |
| lognormal_crps | combined | 3283.724 | 2188.558 | 10276.191 | 29.928420522699835 |
| smooth_wcrps | internal | 7485.971 | 3486.418 | 14867.434 | 46.04707545285035 |
| smooth_wcrps | combined | 3289.439 | 2240.900 | 9984.198 | 29.968141047813376 |
| tailcal_crps | internal | 7523.455 | 3458.705 | 15025.245 | 45.984310420650935 |
| tailcal_crps | combined | 3275.679 | 2212.062 | 10066.702 | 29.88330882380118 |
