# Stability Index (SI) Analysis

## Overview
Python script for calculating atmospheric Stability Index across elevation gradients in the Northwestern Himalayas. The SI quantifies meteorological variability by combining normalized coefficients of variation from multiple climate parameters.

## Formula

```
SI = (1/P) × Σ[(CV_p - CV_p,min) / (CV_p,max - CV_p,min)]
```

Where:
- **P** = Number of parameters (4)
- **CV_p** = Coefficient of Variation for parameter p
- **SI = 0** → Most stable | **SI = 1** → Least stable

## Parameters Analyzed
- Precipitation (Prec)
- 2-meter Temperature (T2M)
- 2-meter Specific Humidity (QV2M)
- 10-meter Wind Speed (WS10M)

## Requirements
```
pandas
numpy
matplotlib
```

## Usage
```python
python Stability_Index.py
```

**Input:** `AllDataAllFeature+Elevation.csv` (must contain Longitude, Latitude, Elevation, Prec, T2M, QV2M, WS10M columns)

## Outputs
Generated in `Images_StabilityIndex/` folder:
- `stability_index_main.png` – SI vs elevation profile
- `stability_index_detailed.png` – Multi-panel parameter breakdown
- `cv_raw_vs_normalized.png` – CV comparison plots
- `stability_index_by_zone.png` – Bar chart by elevation zone
- `stability_index_results.csv` – Numerical results

## Key Features
- Elevation binning with customizable resolution (default: 15 bins)
- Min-max normalization for cross-parameter comparability
- Special analysis of 3500–4000m stability zone
- Publication-ready visualizations (300 DPI)

## Author
Developed for PhD research on optimal rain gauge network design in the Northwestern Himalayas.
