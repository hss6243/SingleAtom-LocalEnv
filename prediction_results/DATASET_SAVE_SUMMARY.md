# Dataset Saving with persite_painn Format

## Overview

The prediction results have been saved as proper persite_painn Dataset objects in `.pt` format. These datasets contain the original DFT target values filtered by property type.

## Saved Datasets

### 1. deltaO Dataset (Oxygen Binding Energy)
- **File**: `dataset_deltaO.pt` (42 MB)
- **Samples**: 587 structures with valid O binding energy targets
- **Location**: `/home3/mercury/1_HE_SAC/2_train/1_DFT_only/prediction_results/`
- **Format**: persite_painn Dataset (torch.save/load compatible)

### 2. deltaOH Dataset (Hydroxyl Binding Energy)
- **File**: `dataset_deltaOH.pt` (39 MB)
- **Samples**: 556 structures with valid OH binding energy targets
- **Location**: `/home3/mercury/1_HE_SAC/2_train/1_DFT_only/prediction_results/`
- **Format**: persite_painn Dataset (torch.save/load compatible)

## Dataset Structure

Both datasets contain the following properties:
```
['nxyz', 'lattice', 'target', 'fidelity', 'num_atoms', 'nbr_list', 'offsets']
```

Each sample is a dictionary with:
- **nxyz**: Atomic coordinates and atomic numbers
- **lattice**: Unit cell lattice vectors
- **target**: Binding energy values (N, 2) tensor for [deltaO, deltaOH]
- **fidelity**: Fidelity level indicators
- **num_atoms**: Number of atoms in structure
- **nbr_list**: Neighbor list for each atom
- **offsets**: Periodic image offsets for neighbors

## Loading the Datasets

### Method 1: Using torch.load (Direct)
```python
import torch
from persite_painn.data.dataset import Dataset

# Load deltaO dataset
dataset_O = torch.load('/home3/mercury/1_HE_SAC/2_train/1_DFT_only/prediction_results/dataset_deltaO.pt')
print(f"Loaded {len(dataset_O)} samples")  # Output: 587

# Load deltaOH dataset
dataset_OH = torch.load('/home3/mercury/1_HE_SAC/2_train/1_DFT_only/prediction_results/dataset_deltaOH.pt')
print(f"Loaded {len(dataset_OH)} samples")  # Output: 556
```

### Method 2: Using persite_painn API
```python
import torch
from persite_painn.data.dataset import Dataset

# Load and access samples
dataset_O = torch.load('dataset_deltaO.pt')
sample = dataset_O[0]  # Access first sample
print(sample.keys())   # View sample structure
```

### Method 3: Iterate Over Dataset
```python
for idx, sample in enumerate(dataset_O):
    nxyz = sample['nxyz']      # Atomic coordinates
    target = sample['target']   # Binding energies
    if idx == 10:
        break
```

## Key Properties

### Filtering Applied
- **deltaO dataset**: Only samples with valid O binding energy target (non-NaN)
- **deltaOH dataset**: Only samples with valid OH binding energy target (non-NaN)

### Original Data Preserved
- All original DFT target values retained
- No modification to targets (only original data included)
- All atomic structure information maintained

### Compatible with persite_painn
- Can be used directly with PainnMultifidelity models
- Compatible with persite_painn DataLoader
- Proper tensor formats maintained

## Metadata

Each dataset has accompanying metadata:
- **Location**: `dataset_deltaO/metadata.json` and `dataset_deltaOH/metadata.json`
- **Contents**:
  - Number of samples
  - Property type
  - Source information
  - Timestamp

## Usage Examples

### Training with Filtered Datasets
```python
from torch.utils.data import DataLoader

dataset_O = torch.load('dataset_deltaO.pt')
dataloader = DataLoader(dataset_O, batch_size=32, shuffle=True)

for batch in dataloader:
    # batch is a dict with keys: nxyz, lattice, target, etc.
    pass
```

### Combining Datasets
```python
import torch

dataset_O = torch.load('dataset_deltaO.pt')
dataset_OH = torch.load('dataset_deltaOH.pt')

# Combine using CombinedDataset or concatenation
combined_samples = list(dataset_O) + list(dataset_OH)
```

### Filtering Further
```python
import numpy as np

dataset_O = torch.load('dataset_deltaO.pt')

# Filter by error magnitude from CSV
high_error_ids = [123, 456, 789]  # Example IDs
filtered_samples = [dataset_O[i] for i in range(len(dataset_O)) 
                   if dataset_O[i]['name'] not in high_error_ids]
```

## File Sizes

```
dataset_deltaO.pt    42 MB  (587 samples)
dataset_deltaOH.pt   39 MB  (556 samples)
```

## Verification

Both datasets have been verified:
- ✓ Successfully saved using `Dataset.save()` method
- ✓ Successfully loaded back using `torch.load()`
- ✓ Proper persite_painn Dataset class instantiation
- ✓ All properties intact (nxyz, lattice, target, fidelity, num_atoms, nbr_list, offsets)
- ✓ Sample structure validated

## Advantages of This Format

1. **Native persite_painn compatibility**: Works seamlessly with existing tools
2. **Efficient storage**: Compressed torch format with proper serialization
3. **Full feature preservation**: All atomic structure features maintained
4. **Easy filtering**: Original data readily available for further processing
5. **Metadata tracking**: Associated JSON metadata for reference

## Next Steps

These datasets can be used for:
- Retraining models on subset of high-quality data
- Fine-tuning on specific properties (O vs OH)
- Comparison studies between datasets
- Error analysis on specific subsets
- Cross-validation studies

---
**Created**: 2026-01-27  
**Format**: persite_painn Dataset (torch.save/.pt)  
**Status**: ✓ Verified and Ready to Use
