# MakeGoldenFeature
Auto Feature Engineering, Make Golden Features in Tabular Dataset

This repository mimics mljar's Golde Features Method.

It has the advantage of being able to automatically proceed with FeatureEnginering.

It working on Tabular Dataset.

# 1. New Features Generation

![image](https://github.com/HaloKim/MakeGoldenFeature/assets/44603549/b78517cc-ae4e-4c1f-aa2a-7e366fb0e418)


# 2. Golden Features Selection

![image](https://github.com/HaloKim/MakeGoldenFeature/assets/44603549/f635cfa8-90d3-427e-b89b-925cf1a75fc5)

# Performance

[Sample JupyterNotebook](https://github.com/HaloKim/MakeGoldenFeature/blob/main/GoldenFeature-BostonDataset.ipynb)

The Boston dataset showed the following performance improvements using the Top10 feature.

Top
```
[{'Feature': 'NOX_plus_PTRATIO', 'Score': 0.8822110289465535},
 {'Feature': 'RM_divide_B', 'Score': 0.8669953317836949},
 {'Feature': 'TAX_plus_PTRATIO', 'Score': 0.8668896359860812},
 {'Feature': 'RM_divide_RAD', 'Score': 0.8664016362396519},
 {'Feature': 'TAX_divide_B', 'Score': 0.8661958667613189},
 {'Feature': 'B_divide_LSTAT', 'Score': 0.8649016554522404},
 {'Feature': 'RM_multiply_AGE', 'Score': 0.8635321077767777},
 {'Feature': 'CRIM_multiply_PTRATIO', 'Score': 0.8631138222798383},
 {'Feature': 'DIS_multiply_LSTAT', 'Score': 0.8619849012074536},
 {'Feature': 'CRIM_divide_AGE', 'Score': 0.8617352792173447}]
```

```
BEFORE MSE: 24.2911194749736
AFTER MSE: 17.55893128587233
```

# Reference

[Golden Features](https://mljar.com/automated-machine-learning/golden-features/)
