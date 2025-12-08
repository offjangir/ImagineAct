from .atari_buffer import ReplayBuffer_FrameStack
from .mujoco_buffer import ReplayBuffer_IRCR, ReplayBuffer_RRD, ReplayBuffer_Transition

def create_buffer(args):
    if args.env_category=='atari':
        return ReplayBuffer_FrameStack(args)
    elif args.env_category=='mujoco':
        if args.alg=='ircr':
            return ReplayBuffer_IRCR(args)
        if args.alg=='rrd':
            return ReplayBuffer_RRD(args)
        return ReplayBuffer_Transition(args)
    elif args.env_category=='libero':
        # Use offline buffer for LIBERO
        from .libero_buffer import OfflineLiberoBuffer
        return OfflineLiberoBuffer(args)
    else:
        raise NotImplementedError(f"Buffer not implemented for env_category: {args.env_category}")
