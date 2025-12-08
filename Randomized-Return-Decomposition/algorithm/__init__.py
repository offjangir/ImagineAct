# Support both TensorFlow and PyTorch backends
import os

# Check if user wants to use PyTorch (via environment variable or args)
# Default to PyTorch if available, otherwise fall back to TensorFlow
USE_PYTORCH_ENV = os.environ.get('USE_PYTORCH', 'auto')

# Try PyTorch first (preferred for this project)
USE_PYTORCH = False
if USE_PYTORCH_ENV == 'auto':
    # Auto-detect: try PyTorch first
    try:
        import torch
        USE_PYTORCH = True
    except ImportError:
        USE_PYTORCH = False
elif USE_PYTORCH_ENV == '1':
    USE_PYTORCH = True
else:
    USE_PYTORCH = False

if USE_PYTORCH:
    # PyTorch implementations
    try:
        from .basis_alg.ddpg_torch import DDPG
        from .basis_alg.sac_torch import SAC
        # Note: TD3 and DQN PyTorch versions not yet implemented
        # from .basis_alg.td3_torch import TD3
        # from .basis_alg.dqn_torch import DQN
        
        basis_algorithm_collection = {
            'ddpg': DDPG,
            'sac': SAC,
            # 'td3': TD3,
            # 'dqn': DQN
        }
        
        from .rrd_torch import RRD
        
        advanced_algorithm_collection = {
            'rrd': RRD
        }
        print("Using PyTorch backend for algorithms")
    except ImportError as e:
        print(f"Error importing PyTorch implementations: {e}")
        if USE_PYTORCH_ENV == '1':  # Explicitly requested
            raise
        print("Falling back to TensorFlow")
        USE_PYTORCH = False

if not USE_PYTORCH:
    # TensorFlow implementations (original) - only import if TensorFlow is available
    try:
        import tensorflow as tf
        from .basis_alg.dqn import DQN
        from .basis_alg.ddpg import DDPG
        from .basis_alg.td3 import TD3
        from .basis_alg.sac import SAC
        
        basis_algorithm_collection = {
            'dqn': DQN,
            'ddpg': DDPG,
            'td3': TD3,
            'sac': SAC
        }
        
        from .ircr import IRCR
        from .rrd import RRD
        
        advanced_algorithm_collection = {
            'ircr': IRCR,
            'rrd': RRD
        }
        print("Using TensorFlow backend for algorithms")
    except ImportError:
        # TensorFlow not available - this is OK if we're using PyTorch
        if not USE_PYTORCH:
            raise ImportError("Neither PyTorch nor TensorFlow is available. Please install one of them.")
        # If we're using PyTorch, TensorFlow is optional (only needed for DQN/Atari)
        # Initialize empty collections if not already set
        if 'basis_algorithm_collection' not in locals():
            basis_algorithm_collection = {}
        if 'advanced_algorithm_collection' not in locals():
            advanced_algorithm_collection = {}

# Ensure collections are defined
if 'basis_algorithm_collection' not in locals():
    basis_algorithm_collection = {}
if 'advanced_algorithm_collection' not in locals():
    advanced_algorithm_collection = {}

algorithm_collection = {
    **basis_algorithm_collection,
    **advanced_algorithm_collection
}

def create_agent(args):
    # Check if args has use_pytorch attribute
    if hasattr(args, 'use_pytorch') and args.use_pytorch:
        global USE_PYTORCH
        if not USE_PYTORCH:
            # Need to reload with PyTorch
            print("Switching to PyTorch backend...")
            os.environ['USE_PYTORCH'] = '1'
            import importlib
            import sys
            # Reload this module
            importlib.reload(sys.modules[__name__])
    
    return algorithm_collection[args.alg](args)
