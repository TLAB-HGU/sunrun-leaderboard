# probtraj

Post-hoc feat30 diagnostic; no upload or parent promotion.

Frozen internal selection: probtraj

| Stage | MSE | Peak MAE | Positive 6h ramp MAE | CSI | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| f1 | 4927.8331 | 60.8866 | 75.3544 | 0.0325 | 59.0000 | 536.0000 |
| final | 4231.9801 | 62.6253 | 69.1647 | 0.0780 | 115.0000 | 630.0000 |
| combined | 4531.1829 | 61.8777 | 71.8262 | 0.0583 | 174.0000 | 1166.0000 |
| parent | 3249.8423 | 64.2098 | 66.0438 | 0.0506 | 114.0000 | 1180.0000 |

Probability metrics (all cases, never-crossing included):

```json
{
  "crps": 35.53568218723573,
  "hourly_brier": 0.08032366063711385,
  "any550_brier": 0.14160998904745484,
  "never550_probability": 0.5219315037972377,
  "fallback_count": 0
}
```
