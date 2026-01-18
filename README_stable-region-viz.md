# Stable Region Rainfall Events Combined Visualization

## Overview
Generates a single composite map displaying multiple 95th percentile precipitation events, highlighting spatial relationships between the atmospheric stability zone (3600–4000m) and surrounding regions within a configurable radius.

## Purpose
Visualizes the spatial clustering and propagation patterns of extreme rainfall events, showing how events in the stability zone relate to concurrent extremes at other elevations within a specified distance.

## Key Features
- Plots up to 10 most recent extreme events on one map
- Color-coded by event date for easy differentiation
- Haversine distance calculation for accurate radius filtering
- Stars (★) mark stable region events; dots (●) mark nearby events at other elevations

## Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_file` | `AllDataAllFeature+Elevation.csv` | Source dataset |
| `search_radius_km` | 100 | Radius to search for concurrent events |
| `num_events_to_plot` | 10 | Number of latest events to display |

## Requirements
```
pandas
numpy
matplotlib
cartopy
```

## Usage
```python
python stable-region-viz-combined-V2.py
```

## Output
- `rainfall_events_combined_latest_10.png` – Composite map with legend and statistics


