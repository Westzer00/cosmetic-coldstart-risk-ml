# Final Metrics

## Model

- Feature Set: V2
- Model: Random Forest
- CV PR-AUC: 0.3524
- Final threshold: 0.47
- Threshold selected using Train OOF predictions only

## Held-out Test

- Test observations: 255
- Actual High-Risk: 65
- Detected High-Risk: 41

| Metric | Value |
|---|---:|
| PR-AUC | 0.384 |
| Precision | 0.342 |
| Recall | 0.631 |
| F1 | 0.443 |
| F0.5 | 0.376 |
| ROC-AUC | 0.623 |
| Accuracy | 0.596 |

## Confusion Matrix

```text
[[111, 79],
 [ 24, 41]]
```

## Interpretation

The model is intended for **risk screening**, not clinical diagnosis.

Its practical question is:

> Among new product × skin-type combinations with no reviews yet,
> which combinations should receive earlier formulation review or monitoring?
