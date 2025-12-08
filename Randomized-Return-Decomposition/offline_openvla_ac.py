"""
Offline Actor-Critic for OpenVLA Features

This module implements an offline actor-critic algorithm for training OpenVLA
as an RL agent using precomputed vision and language features.

Components:
- ValueCritic: Value function V(s) that takes state features
- OpenVLA actor head: Uses OpenVLA's existing actor head to get log probabilities
- OpenVLARewardNet: Uses OpenVLARewardNet for computing dense rewards
- GAE: Generalized Advantage Estimation for advantage computation
- Loss functions: TD(0) critic loss, policy gradient actor loss, and behavior cloning loss

Usage:
    from algorithm.openvla_reward_net import OpenVLARewardNet
    import offline_openvla_ac as ac
    
    # Create reward network
    reward_net = OpenVLARewardNet(...)
    
    # Use in training
    losses = ac.compute_ac_losses(
        vla_model=vla_model,
        critic=critic,
        reward_model=reward_net,
        batch=batch,
        ...
    )
"""

import os
import pickle
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


def init_mlp_weights(module: nn.Module, init_type: str = "orthogonal"):
    """
    Initialize MLP weights.
    
    Args:
        module: PyTorch module
        init_type: "orthogonal" or "xavier"
    """
    if isinstance(module, nn.Linear):
        if init_type == "orthogonal":
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
        elif init_type == "xavier":
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ValueCritic(nn.Module):
    """
    Value function network (critic) that estimates V(s).
    
    Takes state features (concatenated vision + language features) as input
    and outputs a scalar value estimate.
    """
    
    def __init__(self, state_dim: int, hidden_dims: Tuple[int, ...] = (512, 512)):
        """
        Initialize value critic.
        
        Args:
            state_dim: Dimension of state features (vision_dim + lang_dim after pooling/concatenation)
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()
        layers = []
        last_dim = state_dim
        
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(lambda m: init_mlp_weights(m, init_type="orthogonal"))
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: [B, state_dim] state features
            
        Returns:
            values: [B, 1] value estimates V(s)
        """
        return self.net(state)


def get_openvla_log_probs(
    vla_model,
    actions: torch.Tensor,
    language_features: torch.Tensor,
    vision_features: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    unnorm_key: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get log probabilities of actions using OpenVLA's decoder directly with precomputed features.
    
    This function bypasses the vision backbone and language encoder, using precomputed features
    directly and passing them to the decoder layers.
    
    Args:
        vla_model: OpenVLA model instance
        language_features: [B, seq_len, embed_dim] precomputed language embeddings
        vision_features: [B, num_patches, vision_dim] precomputed vision patch features (before projection)
        attention_mask: [B, seq_len] attention mask (optional)
        actions: [B, action_dim] continuous actions (will be converted to tokens)
        unnorm_key: Dataset key for action normalization stats
        
    Returns:
        log_probs: [B] log probabilities of the actions
    """
    # Get action dimension
    action_dim = vla_model.get_action_dim(unnorm_key)
    
    # Convert continuous actions to action tokens
    # First, normalize actions to [-1, 1] range
    action_stats = vla_model.get_action_stats(unnorm_key)
    action_high = torch.tensor(action_stats["q99"], device=actions.device, dtype=actions.dtype)
    action_low = torch.tensor(action_stats["q01"], device=actions.device, dtype=actions.dtype)
    mask = torch.tensor(action_stats.get("mask", np.ones_like(action_stats["q01"], dtype=bool)), 
                       device=actions.device, dtype=torch.bool)
    
    # Normalize actions: reverse the unnormalization
    # Original: actions = 0.5 * (normalized + 1) * (high - low) + low
    # Reverse: normalized = 2 * (actions - low) / (high - low) - 1
    normalized_actions = torch.where(
        mask.unsqueeze(0),
        2.0 * (actions - action_low.unsqueeze(0)) / (action_high.unsqueeze(0) - action_low.unsqueeze(0) + 1e-8) - 1.0,
        actions
    )
    normalized_actions = torch.clamp(normalized_actions, -1.0, 1.0)
    
    # Convert normalized actions to action tokens using action_tokenizer
    # OpenVLA uses discretized action bins via np.digitize
    if hasattr(vla_model, 'action_tokenizer'):
        action_tokenizer = vla_model.action_tokenizer
        # Get bins from action_tokenizer
        bins = action_tokenizer.bins  # numpy array
        vocab_size = action_tokenizer.tokenizer.vocab_size
        
        # Convert to numpy for digitize
        normalized_actions_np = normalized_actions.detach().cpu().numpy()
        
        # Use np.digitize to get bin indices (1 to n_bins)
        discretized_actions = np.digitize(normalized_actions_np, bins)  # [B, action_dim]
        
        # Convert to token IDs: vocab_size - discretized_action
        action_token_ids = vocab_size - discretized_actions  # [B, action_dim]
        action_token_ids = torch.from_numpy(action_token_ids).long().to(actions.device)
    elif hasattr(vla_model, 'bins'):
        # Fallback: use bins directly if available (for OpenVLAForActionPrediction)
        bins = vla_model.bins  # numpy array
        vocab_size = vla_model.vocab_size if hasattr(vla_model, 'vocab_size') else vla_model.config.text_config.vocab_size
        
        # Convert to numpy for digitize
        normalized_actions_np = normalized_actions.detach().cpu().numpy()
        
        # Use np.digitize to get bin indices
        discretized_actions = np.digitize(normalized_actions_np, bins)  # [B, action_dim]
        
        # Convert to token IDs: vocab_size - discretized_action
        action_token_ids = vocab_size - discretized_actions  # [B, action_dim]
        action_token_ids = torch.from_numpy(action_token_ids).long().to(actions.device)
    else:
        raise ValueError("Cannot find action_tokenizer or bins in vla_model")
    
    # Use precomputed features directly - bypass vision backbone and language encoder
    # language_features: [B, seq_len, embed_dim] - already embeddings
    # vision_features: [B, num_patches, vision_dim] - patch features before projection
    
    # Project vision features through projector to match LLM embedding dimension
    # Note: Projector is frozen, but we still need to use it to project features
    if hasattr(vla_model, 'projector'):
        projector = vla_model.projector
    else:
        raise ValueError("Could not find projector in vla_model")
    
    # Convert vision_features to match projector's dtype (usually bfloat16)
    # Get dtype from projector parameters
    projector_dtype = next(projector.parameters()).dtype
    vision_features = vision_features.to(dtype=projector_dtype)
    
    # Project vision features: [B, num_patches, vision_dim] -> [B, num_patches, llm_embed_dim]
    # Note: Projector is frozen, so outputs won't have gradients. This is expected and OK.
    # Gradients will still flow through decoder's trainable LoRA parameters.
    projected_patch_embeddings = projector(vision_features)  # [B, num_patches, llm_embed_dim]
    
    # Build multimodal embeddings: [BOS, vision_patches, rest_of_language_tokens]
    # OpenVLA inserts vision patches after the first token (BOS)
    # Convert language_features to match decoder dtype (usually bfloat16)
    if hasattr(vla_model, 'language_model'):
        decoder = vla_model.language_model
    elif hasattr(vla_model, 'llm_backbone') and hasattr(vla_model.llm_backbone, 'llm'):
        decoder = vla_model.llm_backbone.llm
    else:
        raise ValueError("Could not find decoder (language_model or llm_backbone.llm) in vla_model")
    
    decoder_dtype = next(decoder.parameters()).dtype
    input_embeddings = language_features.to(dtype=decoder_dtype)  # [B, seq_len, embed_dim]
    
    multimodal_embeddings = torch.cat(
        [
            input_embeddings[:, :1, :],  # BOS token
            projected_patch_embeddings,  # Vision patches (already in correct dtype)
            input_embeddings[:, 1:, :],  # Rest of language tokens
        ],
        dim=1,
    )  # [B, seq_len + num_patches, embed_dim]
    
    # Build attention mask if provided
    multimodal_attention_mask = None
    if attention_mask is not None:
        num_patches = projected_patch_embeddings.shape[1]
        patch_attention_mask = torch.ones(
            (multimodal_embeddings.shape[0], num_patches),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        multimodal_attention_mask = torch.cat(
            [attention_mask[:, :1], patch_attention_mask, attention_mask[:, 1:]], dim=1
        )
    
    # Pass multimodal embeddings directly to decoder
    # (decoder was already retrieved above for dtype conversion)
    # The decoder's forward method will process through transformer layers
    # Use inputs_embeds to bypass tokenization
    # Disable cache to save memory (we don't need past_key_values for training)
    decoder_outputs = decoder(
        input_ids=None,  # Not needed when using inputs_embeds
        inputs_embeds=multimodal_embeddings,
        attention_mask=multimodal_attention_mask,
        return_dict=True,
        use_cache=False,  # Disable KV cache to save memory
    )
    
    # Get logits from decoder output
    # decoder_outputs should have .logits attribute
    if hasattr(decoder_outputs, 'logits'):
        logits = decoder_outputs.logits  # [B, seq_len + num_patches, vocab_size]
    else:
        raise ValueError("Decoder output does not have logits attribute")
    
    # Extract action logits (last action_dim tokens)
    # The action tokens are at the end of the sequence
    logits = logits[:, -action_dim:, :]  # [B, action_dim, vocab_size]
    
    # Compute log probabilities for each action token
    log_probs_per_dim = []
    for i in range(action_dim):
        # Get logits for this action dimension
        dim_logits = logits[:, i, :]  # [B, vocab_size]
        # Get target token ID for this dimension
        target_token = action_token_ids[:, i]  # [B]
        # Compute log probability
        log_probs_dim = F.log_softmax(dim_logits, dim=-1)  # [B, vocab_size]
        log_probs_per_dim.append(log_probs_dim.gather(1, target_token.unsqueeze(1)).squeeze(1))  # [B]
    
    # Sum log probabilities across action dimensions
    log_probs = torch.stack(log_probs_per_dim, dim=0).sum(dim=0)  # [B]
    
    # Return both aggregated log_probs and per-dimension logits/token IDs
    return log_probs, logits, action_token_ids


def decode_action_from_logits_differentiable(
    vla_model,
    action_logits: torch.Tensor,
    unnorm_key: Optional[str] = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Differentiable version of action decoding for behavior cloning.
    
    Uses softmax-weighted bin centers instead of argmax, allowing gradients to flow.
    
    Args:
        vla_model: OpenVLA model with bin_centers and action stats
        action_logits: [B, action_dim, vocab_size] logits for action tokens
        unnorm_key: Dataset key for action normalization stats
        temperature: Temperature for softmax (lower = sharper distribution)
    
    Returns:
        actions: [B, action_dim] continuous actions (differentiable)
    """
    # Get actual vocab size from logits tensor (may differ from model config)
    actual_vocab_size = action_logits.shape[-1]  # [B, action_dim, vocab_size] -> vocab_size
    
    # Get bin_centers as tensor
    bin_centers = vla_model.bin_centers
    if isinstance(bin_centers, np.ndarray):
        bin_centers = torch.from_numpy(bin_centers)
    elif not torch.is_tensor(bin_centers):
        bin_centers = torch.as_tensor(bin_centers)
    bin_centers = bin_centers.to(device=action_logits.device, dtype=action_logits.dtype)
    num_bins = bin_centers.shape[0]
    
    # Compute softmax probabilities over vocabulary
    # action_logits: [B, action_dim, vocab_size]
    probs = F.softmax(action_logits / temperature, dim=-1)  # [B, action_dim, vocab_size]
    
    # Map vocabulary tokens to bin indices: token_id = vocab_size - bin_index - 1
    # So bin_index = vocab_size - token_id - 1 (matches evaluation script)
    # Token IDs range from 0 to actual_vocab_size-1
    # Bin indices range from 0 to num_bins-1
    token_ids = torch.arange(actual_vocab_size, device=action_logits.device, dtype=torch.long)  # [vocab_size]
    bin_indices = actual_vocab_size - token_ids - 1  # [vocab_size]
    bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)  # [vocab_size]
    
    # For each action dimension, compute weighted sum of bin centers
    # probs: [B, action_dim, vocab_size]
    # bin_indices: [vocab_size] -> maps token_id to bin_index
    # bin_centers: [num_bins]
    
    # Expand bin_centers to match vocabulary size
    # bin_centers_by_token[token_id] = bin_centers[bin_indices[token_id]]
    bin_centers_by_token = bin_centers[bin_indices]  # [vocab_size]
    
    # For each (batch, action_dim), compute weighted sum: sum(probs * bin_centers_by_token)
    # probs: [B, action_dim, vocab_size]
    # bin_centers_by_token: [vocab_size] -> expand to [1, 1, vocab_size]
    bin_centers_expanded = bin_centers_by_token.unsqueeze(0).unsqueeze(0)  # [1, 1, vocab_size]
    normalized_actions = (probs * bin_centers_expanded).sum(dim=-1)  # [B, action_dim]
    
    # Unnormalize to environment action space
    action_stats = vla_model.get_action_stats(unnorm_key)
    action_low = torch.as_tensor(action_stats["q01"], device=normalized_actions.device, dtype=normalized_actions.dtype)
    action_high = torch.as_tensor(action_stats["q99"], device=normalized_actions.device, dtype=normalized_actions.dtype)
    mask = torch.as_tensor(
        action_stats.get("mask", np.ones_like(action_stats["q01"], dtype=bool)),
        device=normalized_actions.device,
        dtype=torch.bool,
    )
    
    actions = torch.where(
        mask.unsqueeze(0),
        0.5 * (normalized_actions + 1.0) * (action_high.unsqueeze(0) - action_low.unsqueeze(0)) + action_low.unsqueeze(0),
        normalized_actions,
    )
    return actions


def decode_action_from_logits(
    vla_model,
    action_logits: torch.Tensor,
    unnorm_key: Optional[str] = None,
) -> torch.Tensor:
    """
    Decode continuous actions from OpenVLA action logits.
    
    Mirrors the evaluation-time decoding in the rollout script:
    - Take argmax token per action dimension
    - Map tokens back to discretized bin indices
    - Map bin indices to normalized actions via bin centers
    - Unnormalize using OpenVLA's stored action statistics.
    
    Args:
        vla_model: OpenVLA model with bin_centers and action stats
        action_logits: [B, action_dim, vocab_size] logits for action tokens
        unnorm_key: Dataset key for action normalization stats
    
    Returns:
        actions: [B, action_dim] continuous actions
    """
    # Determine vocab size
    vocab_size = getattr(vla_model, "vocab_size", vla_model.config.text_config.vocab_size)
    
    # Argmax over vocabulary for each action dimension
    action_tokens = torch.argmax(action_logits, dim=-1)  # [B, action_dim]
    
    # Undo token → bin mapping: token_id = vocab_size - bin_index - 1
    # So bin_index = vocab_size - token_id - 1 (matches evaluation script)
    discretized_actions = vocab_size - action_tokens - 1
    
    # Clamp to valid bin range and ensure bin_centers is a torch tensor on the right device
    bin_centers = vla_model.bin_centers  # typically numpy array or CPU tensor
    if isinstance(bin_centers, np.ndarray):
        bin_centers = torch.from_numpy(bin_centers)
    elif not torch.is_tensor(bin_centers):
        bin_centers = torch.as_tensor(bin_centers)
    # Move to same device / dtype as action_logits indices
    bin_centers = bin_centers.to(device=action_logits.device, dtype=action_logits.dtype)
    max_bin_index = bin_centers.shape[0] - 1
    discretized_actions = torch.clamp(discretized_actions, 0, max_bin_index)
    
    # Map to normalized actions in [-1, 1] via bin centers
    normalized_actions = bin_centers[discretized_actions]  # [B, action_dim]
    
    # Unnormalize to environment action space using stored stats
    action_stats = vla_model.get_action_stats(unnorm_key)
    # Convert stats to tensors on the correct device/dtype
    action_low = torch.as_tensor(action_stats["q01"], device=normalized_actions.device, dtype=normalized_actions.dtype)
    action_high = torch.as_tensor(action_stats["q99"], device=normalized_actions.device, dtype=normalized_actions.dtype)
    mask = torch.as_tensor(
        action_stats.get("mask", np.ones_like(action_stats["q01"], dtype=bool)),
        device=normalized_actions.device,
        dtype=torch.bool,
    )
    
    # Inverse of normalization used during training:
    # actions = 0.5 * (normalized + 1) * (high - low) + low
    actions = torch.where(
        mask.unsqueeze(0),
        0.5 * (normalized_actions + 1.0) * (action_high.unsqueeze(0) - action_low.unsqueeze(0)) + action_low.unsqueeze(0),
        normalized_actions,
    )
    return actions


def compute_gae_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    normalize: bool = True,
    clip_value: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute Generalized Advantage Estimation (GAE) advantages.
    
    GAE formula for sequences:
        delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
        gae_t = delta_t + gamma * lambda * (1 - done_t) * gae_{t+1}
        advantage_t = gae_t
    
    For individual transitions (batch), this simplifies to TD error:
        advantage_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
    
    Args:
        rewards: [B] or [T, B] rewards per step (if [T, B], treated as sequence)
        values: [B] or [T, B] value estimates V(s_t)
        next_values: [B] or [T, B] value estimates V(s_{t+1})
        dones: [B] or [T, B] terminal flags (1 if terminal, else 0)
        gamma: Discount factor
        gae_lambda: GAE lambda parameter (smoothing factor, only used for sequences)
        normalize: Whether to normalize advantages (zero mean, unit variance)
        clip_value: Clip advantages to [-clip_value, clip_value] to prevent outliers (None = no clipping)
        
    Returns:
        advantages: [B] or [T, B] GAE advantages
    """
    # Check if inputs are sequences [T, B] or batches [B]
    is_sequence = len(rewards.shape) == 2
    
    if is_sequence:
        # Sequence-based GAE computation
        T, B = rewards.shape
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
        
        # Process backwards through time
        for t in reversed(range(T)):
            # Compute TD error
            delta = rewards[t] + gamma * next_values[t] * (1.0 - dones[t]) - values[t]
            # Update GAE: gae_t = delta_t + gamma * lambda * (1 - done_t) * gae_{t+1}
            gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae
    else:
        # Batch of individual transitions: simplified GAE (just TD error)
        # delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
        deltas = rewards + gamma * next_values * (1.0 - dones) - values  # [B]
        advantages = deltas
    
    # Normalize advantages (zero mean, unit variance)
    if normalize:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Clip advantages to prevent outliers from causing very large actor losses
    if clip_value is not None and clip_value > 0:
        advantages = torch.clamp(advantages, -clip_value, clip_value)
    
    return advantages


def compute_dense_reward(
    language_features: torch.Tensor,
    vision_features: torch.Tensor,
    actions: torch.Tensor,
    reward_model: nn.Module,
    language_features_next: Optional[torch.Tensor] = None,
    vision_features_next: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute dense reward using OpenVLARewardNet.
    
    Args:
        language_features: [B, seq_len, embed_dim] language token embeddings
        vision_features: [B, num_patches, embed_dim] vision patch features
        actions: [B, action_dim] actions
        reward_model: OpenVLARewardNet instance
        language_features_next: [B, seq_len, embed_dim] next language features (optional)
        vision_features_next: [B, num_patches, embed_dim] next vision features (optional)
        
    Returns:
        reward: [B] scalar rewards per step
    """
    reward = reward_model(
        language_features=language_features,
        vision_features=vision_features,
        acts=actions,
        language_features_next=language_features_next,
        vision_features_next=vision_features_next,
    )
    return reward.squeeze(-1)  # [B]


def compute_ac_losses(
    vla_model,
    critic: ValueCritic,
    reward_model: nn.Module,
    batch: Dict[str, torch.Tensor],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    bc_coef: float = 1.0,
    entropy_coef: float = 0.0,
    critic_coef: float = 1.0,
    unnorm_key: Optional[str] = None,
    use_dense_rewards: bool = True,
    skip_actor_computation: bool = False,
    advantage_clip: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """
    Compute offline actor-critic losses with behavior cloning using OpenVLA's actor head.
    Uses OpenVLARewardNet for dense rewards and GAE for advantage estimation.
    
    Args:
        vla_model: OpenVLA model (used for actor head to get log probabilities)
        critic: Value critic network
        reward_model: OpenVLARewardNet instance for computing dense rewards
        batch: Dictionary containing:
            - state: [B, state_dim] current state features (or input_ids/pixel_values for OpenVLA)
            - next_state: [B, state_dim] next state features
            - action: [B, action_dim] actions from dataset
            - reward: [B] or [B, 1] sparse rewards (optional if use_dense_rewards=True)
            - done: [B] or [B, 1] terminal flags (1 if terminal, else 0)
            - language_features: [B, seq_len, embed_dim] language features for current state
            - vision_features: [B, num_patches, embed_dim] vision features for current state
            - language_features_next: [B, seq_len, embed_dim] language features for next state (optional)
            - vision_features_next: [B, num_patches, embed_dim] vision features for next state (optional)
            - input_ids: [B, seq_len] token IDs for OpenVLA (optional, if state is not used)
            - pixel_values: [B, C, H, W] pixel values for OpenVLA (optional)
            - attention_mask: [B, seq_len] attention mask (optional)
            - inputs_embeds: [B, seq_len, embed_dim] input embeddings (optional)
        gamma: Discount factor
        gae_lambda: GAE lambda parameter (smoothing factor)
        bc_coef: Weight for behavior cloning loss
        entropy_coef: Weight for entropy regularization (positive = encourage exploration)
        critic_coef: Weight for critic loss
        unnorm_key: Dataset key for action normalization stats
        use_dense_rewards: If True, use reward_model to compute dense rewards; else use batch["reward"]
        skip_actor_computation: If True, skip actor/BC loss computation (for memory efficiency during warmup)
        advantage_clip: Clip advantages to [-advantage_clip, advantage_clip] (0 = disabled)
        
    Returns:
        Dictionary of losses:
            - loss_total: Total combined loss
            - loss_actor: Policy gradient loss
            - loss_bc: Behavior cloning loss (MSE between predicted and dataset actions)
            - loss_critic: TD(0) value regression loss
            - entropy: Policy entropy (detached, for logging, placeholder for now)
    """
    state = batch.get("state")  # [B, state_dim] - may be None if using input_ids/pixel_values directly
    next_state = batch["next_state"]  # [B, state_dim]
    action = batch["action"]  # [B, action_dim]
    done = batch["done"]  # [B] or [B, 1]
    
    # Get OpenVLA feature inputs for reward network and actor head
    language_features = batch.get("language_features")
    vision_features = batch.get("vision_features")
    language_features_next = batch.get("language_features_next")
    vision_features_next = batch.get("vision_features_next")
    attention_mask = batch.get("attention_mask")
    
    # Ensure we have the required features
    if language_features is None or vision_features is None:
        raise ValueError("language_features and vision_features are required for actor head computation")
    
    # Ensure done is shape [B]
    done = done.squeeze(-1) if len(done.shape) > 1 else done
    
    # Compute dense rewards using OpenVLARewardNet
    if use_dense_rewards:
        if language_features is None or vision_features is None:
            raise ValueError("language_features and vision_features required for dense reward computation")
        reward = compute_dense_reward(
            language_features=language_features,
            vision_features=vision_features,
            actions=action,
            reward_model=reward_model,
            language_features_next=language_features_next,
            vision_features_next=vision_features_next,
        )  # [B]
    else:
        reward = batch.get("reward")
        if reward is None:
            raise ValueError("reward required when use_dense_rewards=False")
        reward = reward.squeeze(-1) if len(reward.shape) > 1 else reward
    
    # Get value estimates
    v = critic(state if state is not None else next_state).squeeze(-1)  # [B]
    with torch.no_grad():
        next_v = critic(next_state).squeeze(-1)  # [B]
    
    # Compute GAE advantages
    advantages = compute_gae_advantages(
        rewards=reward,
        values=v,
        next_values=next_v,
        dones=done,
        gamma=gamma,
        gae_lambda=gae_lambda,
        normalize=True,
        clip_value=advantage_clip if advantage_clip > 0 else None,
    )  # [B]
    
    # Critic loss: TD(0) target
    target_v = reward + gamma * (1.0 - done) * next_v  # [B]
    critic_loss = F.mse_loss(v, target_v)
    
    # Actor and BC losses (skip during warmup to save memory)
    if skip_actor_computation:
        # Return dummy values during warmup (not used for training)
        actor_loss = torch.tensor(0.0, device=action.device)
        bc_loss = torch.tensor(0.0, device=action.device)
        entropy = torch.tensor(0.0, device=action.device)
        entropy_loss = torch.tensor(0.0, device=action.device)
    else:
        # Actor loss (policy gradient) using OpenVLA's actor head with GAE advantages
        # Use precomputed features directly - bypass vision backbone and language encoder
        log_prob, action_logits, action_token_ids = get_openvla_log_probs(
            vla_model=vla_model,
            actions=action,
            language_features=language_features,
            vision_features=vision_features,
            attention_mask=attention_mask,
            unnorm_key=unnorm_key,
        )  # [B]
        actor_loss = -(log_prob * advantages.detach()).mean()
        
        # Behavior cloning loss: Get predicted action from OpenVLA and compare with dataset action.
        # We use a differentiable decoding (softmax-weighted bin centers) so gradients can flow.
        # This is different from evaluation-time decoding (which uses argmax for determinism).
        predicted_action = decode_action_from_logits_differentiable(
            vla_model=vla_model,
            action_logits=action_logits,
            unnorm_key=unnorm_key,
            temperature=1.0,  # Can tune this - lower = sharper distribution
        )
        bc_loss = F.mse_loss(predicted_action, action)
        
        # Entropy regularization (optional)
        # For OpenVLA's discrete action space, entropy would need to be computed from logits
        # For now, we'll use a placeholder
        entropy = torch.tensor(0.0, device=action.device)  # Placeholder
        entropy_loss = -entropy_coef * entropy
    
    # Reward network loss (if training reward network)
    # Compare predicted dense rewards with ground truth rewards from dataset
    reward_loss = None
    if batch.get("reward_gt") is not None:
        # Ground truth rewards are provided, compute reward network loss
        reward_gt = batch["reward_gt"]
        reward_gt = reward_gt.squeeze(-1) if len(reward_gt.shape) > 1 else reward_gt
        # Reward network should predict rewards close to ground truth
        reward_loss = F.mse_loss(reward, reward_gt)
    
    # Total loss
    if skip_actor_computation:
        # During warmup, only critic loss matters
        loss_total = critic_coef * critic_loss
    else:
        loss_total = actor_loss + critic_coef * critic_loss + bc_coef * bc_loss + entropy_loss
    if reward_loss is not None:
        loss_total = loss_total  # Reward loss will be added separately in training_step if needed
    
    result = {
        "loss_total": loss_total,
        "loss_actor": actor_loss,
        "loss_bc": bc_loss,
        "loss_critic": critic_loss,
        "entropy": entropy.detach(),
    }
    
    if reward_loss is not None:
        result["loss_reward"] = reward_loss
    
    return result


def training_step(
    vla_model,
    critic: ValueCritic,
    reward_model: nn.Module,
    batch: Dict[str, torch.Tensor],
    vla_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    reward_optimizer: Optional[torch.optim.Optimizer] = None,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    bc_coef: float = 1.0,
    entropy_coef: float = 0.0,
    critic_coef: float = 1.0,
    reward_coef: float = 1.0,
    train_reward_network: bool = False,
    unnorm_key: Optional[str] = None,
    use_dense_rewards: bool = True,
    skip_actor_update: bool = False,
    grad_clip_norm: float = 0.0,
    advantage_clip: float = 0.0,
    zero_grad: bool = True,
    step_optimizers: bool = True,
    accumulation_steps: int = 1,
) -> Dict[str, float]:
    """
    Perform one training step for offline actor-critic using OpenVLA's actor head.
    Uses OpenVLARewardNet for dense rewards and GAE for advantage estimation.
    
    Args:
        vla_model: OpenVLA model (used for actor head)
        critic: Value critic network
        reward_model: OpenVLARewardNet instance for computing dense rewards
        batch: Batch dictionary (see compute_ac_losses for format)
            - reward_gt: [B] Ground truth rewards from dataset (required if train_reward_network=True)
        vla_optimizer: Optimizer for OpenVLA model (actor)
        critic_optimizer: Optimizer for critic
        reward_optimizer: Optimizer for reward network (required if train_reward_network=True)
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        bc_coef: Behavior cloning coefficient
        entropy_coef: Entropy regularization coefficient
        critic_coef: Critic loss coefficient
        reward_coef: Reward network loss coefficient (if training reward network)
        train_reward_network: If True, train reward network on ground truth rewards
        unnorm_key: Dataset key for action normalization stats
        use_dense_rewards: If True, use reward_model to compute dense rewards
        skip_actor_update: If True, only train critic (for warmup period)
        grad_clip_norm: Gradient clipping norm (0 = disabled)
        zero_grad: If True, zero gradients (set to False for gradient accumulation)
        step_optimizers: If True, step optimizers (set to False for gradient accumulation)
        accumulation_steps: Number of accumulation steps (loss is scaled by 1/accumulation_steps)
        
    Returns:
        Dictionary of loss values (as floats, for logging)
    """
    import warnings
    
    # Set reward model to training mode if we're training it
    if train_reward_network:
        reward_model.train()
    else:
        reward_model.eval()
    
    # During warmup, skip actor computation entirely to save memory
    if skip_actor_update:
        vla_model.eval()  # Set to eval mode (not used anyway, but good practice)
        losses = compute_ac_losses(
            vla_model=vla_model,
            critic=critic,
            reward_model=reward_model,
            batch=batch,
            gamma=gamma,
            gae_lambda=gae_lambda,
            bc_coef=bc_coef,
            entropy_coef=entropy_coef,
            critic_coef=critic_coef,
            unnorm_key=unnorm_key,
            use_dense_rewards=use_dense_rewards,
            skip_actor_computation=True,  # Skip actor/BC computation to save memory
            advantage_clip=advantage_clip,
        )
    else:
        vla_model.train()
        # Suppress gradient checkpointing warnings (harmless - projector is frozen but decoder has gradients)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*requires_grad=True.*")
            losses = compute_ac_losses(
                vla_model=vla_model,
                critic=critic,
                reward_model=reward_model,
                batch=batch,
                gamma=gamma,
                gae_lambda=gae_lambda,
                bc_coef=bc_coef,
                entropy_coef=entropy_coef,
                critic_coef=critic_coef,
                unnorm_key=unnorm_key,
                use_dense_rewards=use_dense_rewards,
                skip_actor_computation=False,
                advantage_clip=advantage_clip,
            )
    
    # Zero gradients (only if requested, skip for gradient accumulation)
    if zero_grad:
        if not skip_actor_update:
            vla_optimizer.zero_grad(set_to_none=True)
        critic_optimizer.zero_grad(set_to_none=True)
        if train_reward_network and reward_optimizer is not None:
            reward_optimizer.zero_grad(set_to_none=True)
    
    # Scale loss by 1/accumulation_steps for gradient accumulation
    scale = 1.0 / accumulation_steps
    
    # Backward pass for actor-critic losses
    if skip_actor_update:
        # Only train critic during warmup
        (losses["loss_critic"] * scale).backward()
    else:
        # Normal training: train both actor and critic
        (losses["loss_total"] * scale).backward()
    
    # Backward pass for reward network loss (if training)
    if train_reward_network and "loss_reward" in losses and reward_optimizer is not None:
        (reward_coef * losses["loss_reward"] * scale).backward()
    
    # Gradient clipping (only before optimizer step)
    if step_optimizers and grad_clip_norm > 0:
        if not skip_actor_update:
            torch.nn.utils.clip_grad_norm_(vla_model.parameters(), max_norm=grad_clip_norm)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=grad_clip_norm)
        if train_reward_network and reward_optimizer is not None:
            torch.nn.utils.clip_grad_norm_(reward_model.parameters(), max_norm=grad_clip_norm)
    
    # Update optimizers (only if requested, skip for gradient accumulation)
    if step_optimizers:
        if not skip_actor_update:
            vla_optimizer.step()
        critic_optimizer.step()
        if train_reward_network and reward_optimizer is not None:
            reward_optimizer.step()
    
    # Convert to float for logging
    return {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}


class OfflineOpenVLADataset(Dataset):
    """
    Dataset class for loading offline OpenVLA features and RLDS data.
    
    This is a TODO implementation that should:
    1. Load precomputed OpenVLA features from the pickle cache
    2. Align features with RLDS episodes from libero_goal_no_noops
    3. Return batches with state, next_state, action, reward, done
    
    The state should be the concatenated vision + language features for each step.
    """
    
    def __init__(
        self,
        features_cache_path: str = "/data/kmirakho/JustImagine/Randomized-Return-Decomposition/log/feature_cache/openvla_features_8fabf830.pkl",
        rlds_dataset_path: str = "/data/kmirakho/JustImagine/modified_libero_rlds/libero_goal_no_noops",
        pool_features: bool = True,
        pool_method: str = "mean",
        max_episodes: Optional[int] = None,
        max_transitions: Optional[int] = None,
        verbose: bool = True,
    ):
        """
        Initialize dataset.
        
        Args:
            features_cache_path: Path to pickle file with precomputed OpenVLA features
            rlds_dataset_path: Path to RLDS dataset directory
            pool_features: If True, pool language/vision features to fixed-size vectors
            pool_method: "mean" or "max" pooling method
            max_episodes: Maximum number of episodes to use (None = use all, for quick testing)
            max_transitions: Maximum number of transitions to use (None = use all, for quick testing)
            verbose: If True, print loading progress (set to False for DDP non-main processes)
        """
        self.features_cache_path = features_cache_path
        self.rlds_dataset_path = rlds_dataset_path
        self.pool_features = pool_features
        self.pool_method = pool_method
        self.max_episodes = max_episodes
        self.max_transitions = max_transitions
        self.verbose = verbose
        
        # Check if we're in DDP and should suppress prints
        # Only print if verbose=True AND (not in DDP OR rank 0)
        should_print = verbose
        if verbose:
            try:
                import torch.distributed as dist
                if dist.is_initialized():
                    should_print = (dist.get_rank() == 0)
            except (ImportError, AttributeError):
                pass  # Not using DDP, print normally
        
        # Load features from pickle file
        if should_print:
            print(f"Loading features from: {features_cache_path}", flush=True)
        if not os.path.exists(features_cache_path):
            raise FileNotFoundError(f"Features cache not found: {features_cache_path}")
        
        with open(features_cache_path, 'rb') as f:
            cached_data = pickle.load(f)
        
        features_dict = cached_data.get('features', {})
        if should_print:
            print(f"Loaded features for {len(features_dict)} trajectories", flush=True)
        
        # Check if cache is enhanced (contains actions, rewards, done flags)
        cache_enhanced = False
        if features_dict:
            sample_traj = features_dict.get(0, {})
            if 'actions' in sample_traj and 'rewards' in sample_traj and 'done' in sample_traj:
                cache_enhanced = True
                if should_print:
                    print("✓ Cache is enhanced with actions, rewards, and done flags - skipping RLDS loading!", flush=True)
        
        # Load RLDS episodes only if cache is not enhanced
        if not cache_enhanced:
            if should_print:
                print(f"Loading RLDS episodes from: {rlds_dataset_path}", flush=True)
            try:
                import tensorflow_datasets as tfds
                import tensorflow as tf
                # Configure TensorFlow to not use GPU
                tf.config.set_visible_devices([], 'GPU')
            except ImportError:
                raise ImportError("tensorflow_datasets is required for loading RLDS data")
            
            if not os.path.exists(rlds_dataset_path):
                raise FileNotFoundError(f"RLDS dataset path not found: {rlds_dataset_path}")
            
            builder = tfds.builder_from_directory(rlds_dataset_path)
            dataset = builder.as_dataset(split='train')
            
            # Convert to list of episodes
            episodes = []
            try:
                dataset_iter = dataset.as_numpy_iterator()
            except:
                dataset_iter = iter(dataset)
            
            print("Processing RLDS episodes...", flush=True)
            episode_count = 0
            for episode in dataset_iter:
                episode_count += 1
                if episode_count % 10 == 0:
                    print(f"  Processed {episode_count} episodes...", flush=True)
                episode_dict = {}
                episode_dict['steps'] = []
                
                # Extract task description
                language_instruction = None
                if 'language_instruction' in episode:
                    lang_inst = episode['language_instruction']
                    if hasattr(lang_inst, 'numpy'):
                        language_instruction = lang_inst.numpy()
                    elif isinstance(lang_inst, bytes):
                        language_instruction = lang_inst.decode('utf-8')
                    else:
                        language_instruction = str(lang_inst)
                elif 'task' in episode and isinstance(episode['task'], dict):
                    if 'language_instruction' in episode['task']:
                        lang_inst = episode['task']['language_instruction']
                        if hasattr(lang_inst, 'numpy'):
                            language_instruction = lang_inst.numpy()
                        elif isinstance(lang_inst, bytes):
                            language_instruction = lang_inst.decode('utf-8')
                        else:
                            language_instruction = str(lang_inst)
                
                episode_dict['task_description'] = language_instruction if language_instruction else ""
                
                # Convert steps
                steps_data = episode['steps']
                if hasattr(steps_data, '__iter__') and not isinstance(steps_data, (list, np.ndarray)):
                    try:
                        steps_list = list(steps_data.as_numpy_iterator())
                    except:
                        steps_list = list(steps_data)
                else:
                    steps_list = steps_data
                
                for step in steps_list:
                    # Extract action
                    action = step['action']
                    if hasattr(action, 'numpy'):
                        action_np = action.numpy()
                    else:
                        action_np = np.array(action, dtype=np.float32)
                    
                    # Extract reward
                    reward = step['reward']
                    if hasattr(reward, 'numpy'):
                        reward_val = float(reward.numpy())
                    else:
                        reward_val = float(reward)
                    
                    # Extract flags
                    is_terminal = step.get('is_terminal', False)
                    if hasattr(is_terminal, 'numpy'):
                        is_terminal = bool(is_terminal.numpy())
                    else:
                        is_terminal = bool(is_terminal)
                    
                    is_last = step.get('is_last', False)
                    if hasattr(is_last, 'numpy'):
                        is_last = bool(is_last.numpy())
                    else:
                        is_last = bool(is_last)
                    
                    episode_dict['steps'].append({
                        'action': action_np,
                        'reward': reward_val,
                        'is_terminal': is_terminal,
                        'is_last': is_last,
                    })
                
                episodes.append(episode_dict)
            
            print(f"Loaded {len(episodes)} episodes from RLDS dataset", flush=True)
        else:
            episodes = None  # Not needed when using enhanced cache
        
        # Create list of transitions by aligning features with episodes
        if should_print:
            print("Creating transitions from cache...", flush=True)
        self.transitions = []
        
        # Process trajectories
        if cache_enhanced:
            # Use enhanced cache (no RLDS needed)
            # Apply max_episodes limit if specified
            trajectory_keys = sorted(features_dict.keys())
            if self.max_episodes is not None:
                trajectory_keys = trajectory_keys[:self.max_episodes]
                print(f"Limiting to {self.max_episodes} episodes for quick testing", flush=True)
            
            for traj_idx in trajectory_keys:
                if traj_idx % 50 == 0 and should_print:
                    print(f"  Processing trajectory {traj_idx}/{len(features_dict)}...", flush=True)
                
                traj_features = features_dict[traj_idx]
                lang_feat_list = traj_features.get('lang_feat', [])
                vis_feat_list = traj_features.get('vis_feat', [])
                actions = traj_features.get('actions', None)
                rewards = traj_features.get('rewards', None)
                done_flags = traj_features.get('done', None)
                
                if actions is None or rewards is None or done_flags is None:
                    if should_print:
                        print(f"Warning: Trajectory {traj_idx} missing actions/rewards/done in cache, skipping", flush=True)
                    continue
                
                # Convert features to torch tensors if needed
                lang_feat_tensors = []
                vis_feat_tensors = []
                for f in lang_feat_list:
                    if isinstance(f, np.ndarray):
                        lang_feat_tensors.append(torch.from_numpy(f).float())
                    elif isinstance(f, torch.Tensor):
                        lang_feat_tensors.append(f.float())
                    else:
                        lang_feat_tensors.append(torch.tensor(f, dtype=torch.float32))
                
                for f in vis_feat_list:
                    if isinstance(f, np.ndarray):
                        vis_feat_tensors.append(torch.from_numpy(f).float())
                    elif isinstance(f, torch.Tensor):
                        vis_feat_tensors.append(f.float())
                    else:
                        vis_feat_tensors.append(torch.tensor(f, dtype=torch.float32))
                
                # Convert actions, rewards, done to numpy if needed
                if isinstance(actions, torch.Tensor):
                    actions = actions.numpy()
                if isinstance(rewards, torch.Tensor):
                    rewards = rewards.numpy()
                if isinstance(done_flags, torch.Tensor):
                    done_flags = done_flags.numpy()
                
                num_steps = len(actions)
                num_lang_feat = len(lang_feat_tensors)
                num_vis_feat = len(vis_feat_tensors)
                
                # Features typically have one more observation than steps
                if num_lang_feat == num_steps + 1 and num_vis_feat == num_steps + 1:
                    lang_feat_current = lang_feat_tensors[:-1]
                    vis_feat_current = vis_feat_tensors[:-1]
                    lang_feat_next_list = lang_feat_tensors[1:]
                    vis_feat_next_list = vis_feat_tensors[1:]
                elif num_lang_feat == num_steps and num_vis_feat == num_steps:
                    lang_feat_current = lang_feat_tensors
                    vis_feat_current = vis_feat_tensors
                    lang_feat_next_list = lang_feat_tensors[1:] + [lang_feat_tensors[-1]]
                    vis_feat_next_list = vis_feat_tensors[1:] + [vis_feat_tensors[-1]]
                else:
                    if should_print:
                        print(f"Warning: Trajectory {traj_idx} has {num_steps} steps but {num_lang_feat} lang features and {num_vis_feat} vis features (expected {num_steps} or {num_steps+1}), skipping", flush=True)
                    continue
                
                for step_idx in range(num_steps):
                    lang_feat = lang_feat_current[step_idx]
                    vis_feat = vis_feat_current[step_idx]
                    lang_feat_next = lang_feat_next_list[step_idx]
                    vis_feat_next = vis_feat_next_list[step_idx]
                    
                    # Store raw features - pool lazily in __getitem__ for better performance
                    # This avoids doing expensive pooling operations for all transitions upfront
                    transition = {
                        'action': actions[step_idx],  # Keep as numpy, convert in __getitem__
                        'reward': rewards[step_idx],
                        'done': done_flags[step_idx],
                        'language_features': lang_feat,  # Store tensor reference
                        'vision_features': vis_feat,
                        'language_features_next': lang_feat_next,
                        'vision_features_next': vis_feat_next,
                    }
                    
                    self.transitions.append(transition)
        else:
            # Use RLDS data (original behavior)
            # Apply max_episodes limit if specified
            episodes_to_process = episodes
            if self.max_episodes is not None:
                episodes_to_process = episodes[:self.max_episodes]
                print(f"Limiting to {self.max_episodes} episodes for quick testing", flush=True)
            
            for traj_idx, episode in enumerate(episodes_to_process):
                if traj_idx % 10 == 0 and should_print:
                    print(f"  Processing trajectory {traj_idx}/{len(episodes)}...", flush=True)
                if traj_idx not in features_dict:
                    if should_print:
                        print(f"Warning: No features found for trajectory {traj_idx}, skipping", flush=True)
                    continue
                
                traj_features = features_dict[traj_idx]
                lang_feat_list = traj_features.get('lang_feat', [])
                vis_feat_list = traj_features.get('vis_feat', [])
                
                # Convert features to torch tensors if needed
                lang_feat_tensors = []
                vis_feat_tensors = []
                for f in lang_feat_list:
                    if isinstance(f, np.ndarray):
                        lang_feat_tensors.append(torch.from_numpy(f).float())
                    elif isinstance(f, torch.Tensor):
                        lang_feat_tensors.append(f.float())
                    else:
                        lang_feat_tensors.append(torch.tensor(f, dtype=torch.float32))
                
                for f in vis_feat_list:
                    if isinstance(f, np.ndarray):
                        vis_feat_tensors.append(torch.from_numpy(f).float())
                    elif isinstance(f, torch.Tensor):
                        vis_feat_tensors.append(f.float())
                    else:
                        vis_feat_tensors.append(torch.tensor(f, dtype=torch.float32))
                
                # Create transitions for this trajectory
                steps = episode['steps']
                num_steps = len(steps)
                num_lang_feat = len(lang_feat_tensors)
                num_vis_feat = len(vis_feat_tensors)
                
                # Features typically have one more observation than steps (initial observation + after each action)
                # Expected: num_features = num_steps + 1
                # If features have exactly one more, use features[0:-1] for current and features[1:] for next
                if num_lang_feat == num_steps + 1 and num_vis_feat == num_steps + 1:
                    # Expected case: features include initial observation
                    # Use features[0:-1] for current states (before each action)
                    # Use features[1:] for next states (after each action)
                    lang_feat_current = lang_feat_tensors[:-1]  # All but last
                    vis_feat_current = vis_feat_tensors[:-1]
                    lang_feat_next_list = lang_feat_tensors[1:]  # All but first
                    vis_feat_next_list = vis_feat_tensors[1:]
                elif num_lang_feat == num_steps and num_vis_feat == num_steps:
                    # Features match steps exactly (no initial observation)
                    lang_feat_current = lang_feat_tensors
                    vis_feat_current = vis_feat_tensors
                    lang_feat_next_list = lang_feat_tensors[1:] + [lang_feat_tensors[-1]]  # Shift by 1, repeat last for terminal
                    vis_feat_next_list = vis_feat_tensors[1:] + [vis_feat_tensors[-1]]
                else:
                    if should_print:
                        print(f"Warning: Trajectory {traj_idx} has {num_steps} steps but {num_lang_feat} lang features and {num_vis_feat} vis features (expected {num_steps} or {num_steps+1}), skipping", flush=True)
                    continue
                
                for step_idx in range(num_steps):
                    step = steps[step_idx]
                    lang_feat = lang_feat_current[step_idx]
                    vis_feat = vis_feat_current[step_idx]
                    lang_feat_next = lang_feat_next_list[step_idx]
                    vis_feat_next = vis_feat_next_list[step_idx]
                    
                    # Store raw features - pool lazily in __getitem__ for better performance
                    transition = {
                        'action': step['action'],  # Keep as numpy, convert in __getitem__
                        'reward': step['reward'],
                        'done': float(step['is_terminal'] or step['is_last']),
                        'language_features': lang_feat,
                        'vision_features': vis_feat,
                        'language_features_next': lang_feat_next,
                        'vision_features_next': vis_feat_next,
                    }
                    
                    self.transitions.append(transition)
        
        num_episodes = len(episodes) if episodes is not None else len(features_dict)
        
        # Apply max_transitions limit if specified
        if self.max_transitions is not None and len(self.transitions) > self.max_transitions:
            self.transitions = self.transitions[:self.max_transitions]
            if should_print:
                print(f"Limiting to {self.max_transitions} transitions for quick testing", flush=True)
        
        if should_print:
            print(f"Created {len(self.transitions)} transitions from {num_episodes} episodes", flush=True)
            print(f"Dataset initialization complete!", flush=True)
    
    def __len__(self) -> int:
        """Return number of transitions."""
        return len(self.transitions)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single transition.
        
        Returns:
            Dictionary with keys: state, next_state, action, reward, done,
            language_features, vision_features, language_features_next, vision_features_next
        """
        transition = self.transitions[idx]
        
        # Get raw features
        lang_feat = transition["language_features"]
        vis_feat = transition["vision_features"]
        lang_feat_next = transition.get("language_features_next", lang_feat)
        vis_feat_next = transition.get("vision_features_next", vis_feat)
        
        # Pool features lazily (only when needed)
        if self.pool_features:
            state = self._pool_features(lang_feat, vis_feat)
            next_state = self._pool_features(lang_feat_next, vis_feat_next)
        else:
            # Flatten features
            state = torch.cat([lang_feat.flatten(), vis_feat.flatten()])
            next_state = torch.cat([lang_feat_next.flatten(), vis_feat_next.flatten()])
        
        # Convert action, reward, done to tensors
        action = transition["action"]
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float()
        elif not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32)
        
        reward = transition["reward"]
        if not isinstance(reward, torch.Tensor):
            reward = torch.tensor(reward, dtype=torch.float32)
        
        done = transition["done"]
        if not isinstance(done, torch.Tensor):
            done = torch.tensor(done, dtype=torch.float32)
        
        return {
            "state": state,
            "next_state": next_state,
            "action": action,
            "reward": reward,
            "done": done,
            "language_features": lang_feat,
            "vision_features": vis_feat,
            "language_features_next": lang_feat_next,
            "vision_features_next": vis_feat_next,
            "reward_gt": reward,  # Ground truth reward from dataset
        }
    
    def _pool_features(
        self,
        lang_feat: torch.Tensor,
        vis_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pool language and vision features into a fixed-size state vector.
        
        Args:
            lang_feat: [seq_len, embed_dim] language features
            vis_feat: [num_patches, embed_dim] vision features
            
        Returns:
            state: [state_dim] concatenated pooled features
        """
        if self.pool_method == "mean":
            lang_pooled = lang_feat.mean(dim=0)  # [embed_dim]
            vis_pooled = vis_feat.mean(dim=0)  # [embed_dim]
        elif self.pool_method == "max":
            lang_pooled = lang_feat.max(dim=0)[0]  # [embed_dim]
            vis_pooled = vis_feat.max(dim=0)[0]  # [embed_dim]
        else:
            raise ValueError(f"Unknown pool_method: {self.pool_method}")
        
        # Concatenate to form state vector
        state = torch.cat([lang_pooled, vis_pooled], dim=0)  # [lang_dim + vis_dim]
        return state

