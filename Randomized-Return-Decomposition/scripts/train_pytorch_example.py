#!/usr/bin/env python3
"""
Example training script using PyTorch backend for RRD algorithm.

This demonstrates how to use the PyTorch version of the RRD algorithm.
"""

import os
import sys

# Set environment variable to use PyTorch backend
os.environ['USE_PYTORCH'] = '1'

# Add project root to path (parent directory of scripts/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import get_args, experiment_setup

def main():
    # Get command line arguments
    args = get_args()
    
    # Override to ensure PyTorch is used
    args.use_pytorch = True
    
    # Print backend information
    print("=" * 60)
    print("Training with PyTorch backend")
    print("=" * 60)
    print(f"Algorithm: {args.alg}")
    print(f"Basis Algorithm: {args.basis_alg}")
    print(f"Environment: {args.env}")
    print("=" * 60)
    
    # Run experiment
    experiment_setup(args)

if __name__ == '__main__':
    main()






