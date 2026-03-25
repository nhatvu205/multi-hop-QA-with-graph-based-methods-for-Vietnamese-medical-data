"""
Script to prepare model weights for HuggingFace Space deployment

This script loads a trained model checkpoint and saves only the model state dict
in a format suitable for HuggingFace Space.
"""

import torch
import sys
import os

# Add parent directory to path to import model modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_module.models import KGEnhancedViT5

def prepare_model_for_hf(checkpoint_path, output_path, use_kg=True):
    """
    Load trained model and save state dict for HuggingFace Space
    
    Args:
        checkpoint_path: Path to trained model checkpoint (.pt file)
        output_path: Path to save model state dict
        use_kg: Whether model was trained with KG enhancement
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Model configuration (should match training config)
    config = {
        'vit5_model_name': 'VietAI/vit5-base',
        'kg_node_features': 300,
        'gnn_hidden': 256,
        'gnn_type': 'gcn',
        'gnn_layers': 2,
        'dropout': 0.1,
        'use_kg': use_kg
    }
    
    # Initialize model
    print("Initializing model...")
    model = KGEnhancedViT5(**config)
    
    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract model state dict
    if 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
        print("✓ Found 'model_state_dict' in checkpoint")
    else:
        # Assume checkpoint is the state dict itself
        model_state_dict = checkpoint
        print("✓ Checkpoint is state dict")
    
    # Load state dict
    try:
        model.load_state_dict(model_state_dict)
        print("✓ Model weights loaded successfully")
    except Exception as e:
        print(f"⚠ Warning: {e}")
        print("Attempting to load with strict=False...")
        model.load_state_dict(model_state_dict, strict=False)
        print("✓ Model weights loaded (some keys may be missing)")
    
    # Save only state dict
    print(f"\nSaving model state dict to: {output_path}")
    torch.save(model.state_dict(), output_path)
    print(f"✓ Model state dict saved successfully!")
    print(f"  File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    
    return model

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare model for HuggingFace Space")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_model.pt",
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="hf_space/model_state_dict.pt",
        help="Output path for model state dict"
    )
    parser.add_argument(
        "--use_kg",
        type=bool,
        default=True,
        help="Whether model was trained with KG enhancement"
    )
    
    args = parser.parse_args()
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Prepare model
    model = prepare_model_for_hf(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        use_kg=args.use_kg
    )
    
    print("\n" + "="*70)
    print("Model preparation complete!")
    print("="*70)
    print(f"\nNext steps:")
    print(f"1. Copy '{args.output}' to your HuggingFace Space root directory")
    print(f"2. Rename it to 'model_state_dict.pt' if needed")
    print(f"3. Upload to your HuggingFace Space repository")

