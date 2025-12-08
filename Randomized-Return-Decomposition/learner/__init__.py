from .atari import AtariLearner
from .mujoco import MuJoCoLearner

try:
    from .libero import LiberoLearner
    LIBERO_LEARNER_AVAILABLE = True
except ImportError:
    LIBERO_LEARNER_AVAILABLE = False
    LiberoLearner = None

def create_learner(args):
    learner_map = {
        'atari': AtariLearner,
        'mujoco': MuJoCoLearner,
        'libero': LiberoLearner if LIBERO_LEARNER_AVAILABLE else MuJoCoLearner
    }
    
    if args.env_category not in learner_map:
        raise ValueError(f"Learner not available for env_category: {args.env_category}")
    
    return learner_map[args.env_category](args)
