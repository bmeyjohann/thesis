#!/usr/bin/env python3
"""
Quick script to inspect checkpoint contents and structure.
"""
import torch
import sys
from pathlib import Path

def inspect_checkpoint(checkpoint_path):
    print(f"🔍 Inspecting: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"✓ Checkpoint loaded successfully")
        
        print(f"\n📋 Top-level keys:")
        for key in checkpoint.keys():
            value = checkpoint[key]
            if isinstance(value, torch.Tensor):
                print(f"  {key}: Tensor {value.shape} ({value.dtype})")
            elif isinstance(value, dict):
                print(f"  {key}: Dict with {len(value)} keys")
            elif hasattr(value, '__dict__'):
                print(f"  {key}: Object ({type(value).__name__})")
            else:
                print(f"  {key}: {type(value).__name__} = {value}")
        
        # Look for nested structures
        if 'args' in checkpoint:
            print(f"\n📝 Args attributes:")
            args = checkpoint['args']
            for attr in dir(args):
                if not attr.startswith('_'):
                    try:
                        value = getattr(args, attr)
                        if not callable(value):
                            print(f"  {attr}: {value}")
                    except:
                        pass
        
        if 'policy_cfg' in checkpoint:
            print(f"\n⚙️  Policy config:")
            for key, value in checkpoint['policy_cfg'].items():
                print(f"  {key}: {value}")
        
        # Check if policy_state_dict exists and its structure
        if 'policy_state_dict' in checkpoint:
            print(f"\n🧠 Policy state_dict keys:")
            state_dict = checkpoint['policy_state_dict']
            for key in sorted(state_dict.keys())[:10]:  # Show first 10 keys
                tensor = state_dict[key]
                print(f"  {key}: {tensor.shape}")
            if len(state_dict) > 10:
                print(f"  ... and {len(state_dict) - 10} more keys")
                
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        checkpoint_path = Path(sys.argv[1])
    else:
        # Find the first available checkpoint
        models_dir = Path("models")
        checkpoints = list(models_dir.rglob("*.pt")) + list(models_dir.rglob("*.pth"))
        
        if not checkpoints:
            print("❌ No checkpoint files found in models/")
            sys.exit(1)
        
        checkpoint_path = checkpoints[0]
    
    inspect_checkpoint(checkpoint_path)
