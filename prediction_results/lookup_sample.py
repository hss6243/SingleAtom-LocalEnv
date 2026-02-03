#!/usr/bin/env python
"""
Utility script to lookup sample details from prediction CSV files.
Allows you to identify any point from the parity plot by its sample ID.

Usage:
    python lookup_sample.py --id 258985354 --property deltaO
    python lookup_sample.py --id 260947588
"""

import pandas as pd
import argparse
from pathlib import Path

def lookup_sample(sample_id, property_name=None):
    """
    Lookup a sample by its ID in the prediction CSV files.
    
    Args:
        sample_id (int): The sample ID to lookup
        property_name (str): Optional - 'deltaO' or 'deltaOH' to search specific property
    
    Returns:
        None (prints results to console)
    """
    csv_files = list(Path(__file__).parent.glob('BE_delta*_predictions_*.csv'))
    
    if not csv_files:
        print("Error: No prediction CSV files found in this directory")
        return
    
    found = False
    
    for csv_file in sorted(csv_files):
        # Skip if property filter specified and doesn't match
        if property_name and property_name not in csv_file.name:
            continue
        
        # Load CSV
        df = pd.read_csv(csv_file)
        
        # Find sample
        matches = df[df['sample_name'] == sample_id]
        
        if len(matches) > 0:
            found = True
            prop = 'deltaO' if 'deltaO_' in csv_file.name else 'deltaOH'
            
            print(f"\n{'='*70}")
            print(f"Sample ID: {sample_id} | Property: {prop}")
            print(f"{'='*70}")
            
            for idx, row in matches.iterrows():
                print(f"\nDataset Source: {row['dataset_from']}")
                print(f"Sample Index: {row['sample_idx']}")
                print(f"Adsorbate: {row['adsorbate']}")
                print(f"\nDFT Calculated (Target):    {row['target']:>10.6f} eV")
                print(f"ML Model Predicted:         {row['prediction']:>10.6f} eV")
                print(f"Prediction Error:           {row['error']:>10.6f} eV")
                print(f"Absolute Error (|error|):   {row['abs_error']:>10.6f} eV")
                
                # Color code the error
                if row['abs_error'] < 0.3:
                    rating = "✓ Good prediction"
                elif row['abs_error'] < 0.5:
                    rating = "~ Moderate prediction"
                else:
                    rating = "✗ Poor prediction"
                print(f"Prediction Quality:         {rating}")
    
    if not found:
        print(f"\nSample ID {sample_id} not found in prediction results.")
        print("The sample may not have valid binding energy predictions.")
    else:
        print(f"\n{'='*70}\n")

def main():
    parser = argparse.ArgumentParser(
        description='Lookup sample details from parity plot predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python lookup_sample.py --id 258985354
  python lookup_sample.py --id 260947588 --property deltaO
  python lookup_sample.py --id 255767243 --property deltaOH
        """
    )
    
    parser.add_argument('--id', type=int, required=True,
                        help='Sample ID to lookup')
    parser.add_argument('--property', type=str, choices=['deltaO', 'deltaOH'],
                        help='Optional: specific property to search (deltaO or deltaOH)')
    
    args = parser.parse_args()
    
    lookup_sample(args.id, args.property)

if __name__ == '__main__':
    main()
