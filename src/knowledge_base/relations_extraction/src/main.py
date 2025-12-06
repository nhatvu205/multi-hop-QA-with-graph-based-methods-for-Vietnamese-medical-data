import sys
import argparse
from distant_supervision import run_distant_supervision
from train import train_model
from extract import run_extraction

def run_pipeline(steps=None):
    """Run full relation extraction pipeline"""
    if steps is None:
        steps = ['distant_supervision', 'train', 'extract']
    
    print("\n" + "="*60)
    print("RELATION EXTRACTION PIPELINE")
    print("="*60)
    print(f"Steps to run: {', '.join(steps)}")
    print("="*60)
    
    if 'distant_supervision' in steps:
        print("\n[STEP 1/3] Distant Supervision")
        run_distant_supervision()
    
    if 'train' in steps:
        print("\n[STEP 2/3] Training Model")
        train_model()
    
    if 'extract' in steps:
        print("\n[STEP 3/3] Extracting Relations")
        run_extraction()
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Relation Extraction Pipeline')
    parser.add_argument(
        '--steps',
        nargs='+',
        choices=['distant_supervision', 'train', 'extract', 'all'],
        default=['all'],
        help='Steps to run'
    )
    
    args = parser.parse_args()
    
    if 'all' in args.steps:
        steps = ['distant_supervision', 'train', 'extract']
    else:
        steps = args.steps
    
    run_pipeline(steps)

