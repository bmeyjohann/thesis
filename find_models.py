#!/usr/bin/env python3
"""
Find and list trained models for interactive evaluation.

Usage:
    python find_models.py
    python find_models.py --models_dir path/to/models
"""

import os
import argparse
from pathlib import Path
import torch
from datetime import datetime

def get_args():
    parser = argparse.ArgumentParser(description='Find trained models')
    parser.add_argument('--models_dir', type=str, default='models',
                        help='Directory to search for models')
    return parser.parse_args()

def analyze_model(model_path):
    """Analyze a model checkpoint and extract info."""
    try:
        # Load just the metadata (don't load weights to GPU)
        checkpoint = torch.load(model_path, map_location='cpu')
        
        info = {
            'path': str(model_path),
            'size_mb': model_path.stat().st_size / 1024 / 1024,
            'modified': datetime.fromtimestamp(model_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        # Extract training info from different sources
        info['env_name'] = 'unknown'
        info['reward_type'] = 'unknown'
        info['total_timesteps'] = 'unknown'
        info['final_reward'] = 'unknown'
        
        # Try training_info dict first
        if 'training_info' in checkpoint:
            train_info = checkpoint['training_info']
            info.update({
                'env_name': train_info.get('env_name', info['env_name']),
                'reward_type': train_info.get('reward_type', info['reward_type']),
                'total_timesteps': train_info.get('total_timesteps', info['total_timesteps']),
                'final_reward': train_info.get('final_reward', info['final_reward']),
            })
        
        # Try direct checkpoint keys
        if 'total_timesteps' in checkpoint:
            info['total_timesteps'] = checkpoint['total_timesteps']
        if 'iteration' in checkpoint:
            info['iteration'] = checkpoint['iteration']
            
        # Try args namespace
        if 'args' in checkpoint:
            args = checkpoint['args']
            if hasattr(args, 'env_name'):
                info['env_name'] = args.env_name
            if hasattr(args, 'reward_type'):
                info['reward_type'] = args.reward_type
        
        # Try to get model config from different sources
        if 'policy_cfg' in checkpoint:
            info['config'] = checkpoint['policy_cfg']
        elif 'model_config' in checkpoint:
            info['config'] = checkpoint['model_config']
        
        return info
        
    except Exception as e:
        return {'path': str(model_path), 'error': str(e)}

def main():
    args = get_args()
    models_dir = Path(args.models_dir)
    
    print(f"🔍 Searching for trained models in: {models_dir.absolute()}")
    
    if not models_dir.exists():
        print(f"❌ Models directory does not exist: {models_dir}")
        print(f"💡 Train some models first using train_rsl_rl_clean.py")
        return
    
    # Find all .pt and .pth files recursively in subdirectories
    model_files = list(models_dir.rglob('*.pt')) + list(models_dir.rglob('*.pth'))
    
    if not model_files:
        print(f"❌ No model files found in {models_dir}")
        print(f"💡 Train some models first using train_rsl_rl_clean.py")
        return
    
    print(f"✓ Found {len(model_files)} model file(s)")
    print(f"\n📋 Model Analysis:")
    print(f"{'#':<3} {'Path':<50} {'Env':<15} {'Reward':<8} {'Size (MB)':<8} {'Modified':<18}")
    print(f"{'-'*120}")
    
    models_info = []
    for i, model_file in enumerate(sorted(model_files), 1):
        info = analyze_model(model_file)
        models_info.append(info)
        
        if 'error' in info:
            relative_path = model_file.relative_to(models_dir)
            print(f"{i:<3} {str(relative_path):<50} {'ERROR':<15} {'':<8} {'':<8} {info['error']}")
        else:
            relative_path = model_file.relative_to(models_dir)
            env_name = info.get('env_name', 'unknown')[:14]
            reward_type = info.get('reward_type', 'unknown')[:7]
            size_mb = f"{info['size_mb']:.1f}"
            modified = info['modified']
            
            print(f"{i:<3} {str(relative_path):<50} {env_name:<15} {reward_type:<8} {size_mb:<8} {modified:<18}")
    
    # Show example usage commands
    print(f"\n🎮 Example Evaluation Commands:")
    print(f"{'-'*50}")
    
    valid_models = [info for info in models_info if 'error' not in info]
    for i, info in enumerate(valid_models[:3], 1):  # Show first 3 valid models
        model_path = Path(info['path'])
        relative_path = model_path.relative_to(models_dir)
        env_name = info.get('env_name', 'pointmaze-medium-v0')
        
        print(f"\n{i}. {relative_path}:")
        print(f"   python eval_interactive.py --model_path {info['path']} --env_name {env_name}")
        
        # Show additional options
        if i == 1:  # Show full options for first model
            print(f"   # With custom settings:")
            print(f"   python eval_interactive.py \\")
            print(f"       --model_path {info['path']} \\")
            print(f"       --env_name {env_name} \\")
            print(f"       --num_episodes 5 \\")
            print(f"       --width 1024 --height 768 \\")
            print(f"       --fps 60")
    
    if valid_models:
        print(f"\n💡 Tip: Start with the most recent model (usually best performance)")
    
    print(f"\n📖 For more options: python eval_interactive.py --help")

if __name__ == "__main__":
    main()
