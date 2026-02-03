# Binding Energy Predictions with Sample Tracking

## Overview
This directory contains predictions for binding energy (BE) of O and OH adsorbates on metal-decorated graphene surfaces, including full sample ID tracking.

## Files Generated

### CSV Exports with Sample IDs
- **BE_deltaO_predictions_*.csv**: 270 predictions for oxygen binding energy
- **BE_deltaOH_predictions_*.csv**: 258 predictions for OH binding energy

### CSV Columns
Each CSV file contains:
- `sample_idx`: Index in the combined dataset
- `dataset_from`: Source dataset name (e.g., "sac_2_components")
- `sample_name`: **Sample ID from original dataset** (e.g., 258985354)
- `adsorbate`: Adsorbate type ("deltaO" or "deltaOH")
- `target`: DFT-calculated binding energy (ground truth)
- `prediction`: Model predicted binding energy (ensemble average of 10 models)
- `error`: Prediction error (pred - target)
- `abs_error`: Absolute prediction error |pred - target|

## Key Features

✓ **Sample Traceability**: Each point in the parity plot can now be identified via its sample ID in the CSV
✓ **Dataset Origin**: Tracked which dataset each sample came from
✓ **Ensemble Averaging**: Predictions averaged from 10 PainnMultifidelity models
✓ **Error Analysis**: Includes both signed and absolute errors for analysis

## Model Information

- **Models**: 10 PainnMultifidelity ensemble models
- **Location**: `/home3/mercury/tools/SingleAtom-LocalEnv/models/m_painn/2_components/{0-9}/best_model.pth.tar`
- **Framework**: persite_painn with per-atom predictions

## Performance Metrics

### deltaO (Oxygen binding energy)
- Valid Predictions: 270
- MAE: 0.2039 eV
- RMSE: 0.3044 eV
- Error Range: [-0.9337, 2.4344] eV

### deltaOH (Hydroxyl binding energy)
- Valid Predictions: 258
- MAE: 0.2373 eV
- RMSE: 0.4015 eV
- Error Range: [-1.4298, 3.6678] eV

## Usage Example

To identify a specific point from the parity plot:
```python
import pandas as pd

# Load predictions
df = pd.read_csv('BE_deltaO_predictions_20260127_164237.csv')

# Find sample with ID 258985354
sample = df[df['sample_name'] == 258985354]
print(sample)
# Shows: target=-0.026, prediction=0.415, error=0.441
```

## Reference Format

The CSV format matches the reference format from:
- `/home3/mercury/1_HE_SAC/data/1_PARITY_PLOT_all/1_UMA_CALC_DFT_ALL/BE_parityplot_O.csv`
- `/home3/mercury/1_HE_SAC/data/1_PARITY_PLOT_all/1_UMA_CALC_DFT_ALL/BE_parityplot_OH.csv`

This enables direct comparison between UMA predictions and ML model predictions.
