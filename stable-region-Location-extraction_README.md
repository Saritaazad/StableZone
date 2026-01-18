# Stable Region Location Extraction

## Overview
Utility script to extract unique grid point coordinates within the atmospheric stability zone (3600–4000m elevation) from the Northwestern Himalayas meteorological dataset.

## Purpose
Identifies and exports all grid locations falling within the elevation band exhibiting minimum meteorological variability, as determined by Stability Index analysis. These locations serve as inputs for subsequent extreme event analysis and rain gauge network optimization.

## Functions

### `extract_stable_region_locations()`
Extracts unique Longitude/Latitude pairs within 3600–4000m.

**Output:** `stable_region_3600_4000m_locations.csv`
| Longitude | Latitude |
|-----------|----------|

### `extract_stable_region_with_elevation()`
Extracts coordinates with mean elevation values.

**Output:** `stable_region_3600_4000m_with_elevation.csv`
| Longitude | Latitude | Elevation |
|-----------|----------|-----------|

## Requirements
```
pandas
numpy
```

## Usage
```python
python stable-region-Location-extraction.py
```

**Input:** `AllDataAllFeature+Elevation.csv`

## Output Summary
- Unique grid locations count
- Geographic extent (lon/lat bounds)
- Elevation statistics (min, max, mean, median)

## Notes
Locations are sorted by Latitude (descending) then Longitude (ascending) for organized spatial reference.

