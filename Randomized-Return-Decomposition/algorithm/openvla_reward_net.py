"""
OpenVLA-based Reward Network for Randomized Return Decomposition

Uses language features and vision patch features from OpenVLA VLM to train reward models.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class OpenVLARewardNet(nn.Module):
    """
    Reward network that uses OpenVLA features (language and vision) instead of raw observations.
    
    Inputs:
        - language_features: Language token embeddings from OpenVLA [batch, seq_len, embed_dim]
        - vision_features: Vision patch features from OpenVLA [batch, num_patches, embed_dim]
        - acts: Actions [batch, act_dim]
        - language_features_next: Language features for next observation (optional)
        - vision_features_next: Vision features for next observation (optional)
    """
    
    def __init__(
        self,
        language_feature_dim: int,
        vision_feature_dim: int,
        act_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
    ):
        super().__init__()
        self.language_feature_dim = language_feature_dim
        self.vision_feature_dim = vision_feature_dim
        self.act_dim = act_dim
        
        # Pooling layers to aggregate sequence/patch features
        # Use mean pooling with optional attention
        self.language_pool = nn.Sequential(
            nn.Linear(language_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        self.vision_pool = nn.Sequential(
            nn.Linear(vision_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        
        # Feature dimension after pooling
        pooled_dim = (hidden_dim // 2) * 2  # language + vision
        
        # Input: pooled language + pooled vision + action + feature differences
        input_dim = pooled_dim * 2 + act_dim  # *2 for current and next (or diff)
        
        # MLP layers
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def pool_features(self, features, pool_layer):
        """
        Pool sequence/patch features into a fixed-size vector.
        Uses mean pooling over the sequence/patch dimension.
        Note: Assumes padding is zeros, so mean pooling will naturally ignore padding
        (since zeros don't contribute to the mean in a meaningful way).
        
        Args:
            features: [batch, seq_len, embed_dim] or [batch, sample_size, seq_len, embed_dim]
            pool_layer: Pooling MLP
        
        Returns:
            pooled: [batch, hidden_dim//2] or [batch, sample_size, hidden_dim//2]
        """
        # Handle both 3D and 4D inputs (for RRD batches with sample_size)
        if len(features.shape) == 4:
            batch_size, sample_size, seq_len, embed_dim = features.shape
            features = features.reshape(batch_size * sample_size, seq_len, embed_dim)
            # Mean pool over sequence (padding zeros will dilute the mean slightly, but should be fine)
            pooled = pool_layer(features.mean(dim=1))  # Mean pool over sequence
            pooled = pooled.reshape(batch_size, sample_size, -1)
        else:
            # Mean pool over sequence/patch dimension
            pooled = pool_layer(features.mean(dim=1))  # [batch, hidden_dim//2]
        
        return pooled
    
    def forward(
        self,
        language_features: torch.Tensor,
        vision_features: torch.Tensor,
        acts: torch.Tensor,
        language_features_next: Optional[torch.Tensor] = None,
        vision_features_next: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through reward network.
        
        Args:
            language_features: [batch, seq_len, embed_dim] or [batch, sample_size, seq_len, embed_dim]
            vision_features: [batch, num_patches, embed_dim] or [batch, sample_size, num_patches, embed_dim]
            acts: [batch, act_dim] or [batch, sample_size, act_dim]
            language_features_next: Optional next language features
            vision_features_next: Optional next vision features
        
        Returns:
            rewards: [batch, 1] or [batch, sample_size, 1]
        """
        # Handle flattening for RRD batches
        flatten = len(language_features.shape) == 4
        if flatten:
            batch_size, sample_size = language_features.shape[:2]
            language_features = language_features.reshape(-1, *language_features.shape[2:])
            vision_features = vision_features.reshape(-1, *vision_features.shape[2:])
            acts = acts.reshape(-1, self.act_dim)
            if language_features_next is not None:
                language_features_next = language_features_next.reshape(-1, *language_features_next.shape[2:])
            if vision_features_next is not None:
                vision_features_next = vision_features_next.reshape(-1, *vision_features_next.shape[2:])
        
        # Pool language and vision features
        lang_pooled = self.pool_features(language_features, self.language_pool)
        vision_pooled = self.pool_features(vision_features, self.vision_pool)
        
        # Handle next features or compute differences
        if language_features_next is not None and vision_features_next is not None:
            lang_pooled_next = self.pool_features(language_features_next, self.language_pool)
            vision_pooled_next = self.pool_features(vision_features_next, self.vision_pool)
            # Use feature differences
            lang_diff = lang_pooled - lang_pooled_next
            vision_diff = vision_pooled - vision_pooled_next
        else:
            # If no next features, use zeros (or could use current features)
            lang_diff = torch.zeros_like(lang_pooled)
            vision_diff = torch.zeros_like(vision_pooled)
        
        # Concatenate: current features + differences + actions
        x = torch.cat([lang_pooled, vision_pooled, lang_diff, vision_diff, acts], dim=-1)
        
        # MLP to predict reward
        r = self.mlp(x)
        
        if flatten:
            r = r.reshape(batch_size, sample_size, 1)
        
        return r


class OpenVLAFeatureExtractor:
    """
    Extracts language and vision features from observations using OpenVLA.
    
    This class wraps the OpenVLA model to extract features from images and text prompts.
    """
    
    def __init__(
        self,
        vla_model,
        processor,
        task_description: str = "",
        device: str = "cuda:0",
    ):
        """
        Initialize feature extractor.
        
        Args:
            vla_model: OpenVLA model (OpenVLAForActionPrediction)
            processor: OpenVLA processor
            task_description: Task description text (optional, can be per-observation)
            device: Device to run on
        """
        self.vla_model = vla_model
        self.processor = processor
        self.task_description = task_description
        self.device = device
        # Ensure model is on the correct device
        if hasattr(self.vla_model, 'to'):
            self.vla_model = self.vla_model.to(device)
        self.vla_model.eval()  # Set to eval mode
    
    def extract_features(
        self,
        images: np.ndarray,
        task_descriptions: Optional[list] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract language and vision features from images.
        
        Args:
            images: Images as numpy arrays [batch, H, W, C] or list of images
            task_descriptions: Optional list of task descriptions (one per image)
                If None, uses self.task_description
        
        Returns:
            language_features: [batch, seq_len, embed_dim] - Language token embeddings
            vision_features: [batch, num_patches, embed_dim] - Vision patch features
        """
        from PIL import Image
        import torch
        
        # Handle single image
        if isinstance(images, np.ndarray) and len(images.shape) == 3:
            images = [images]
        
        # Convert to PIL Images
        pil_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                # Ensure uint8 and convert to PIL
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                pil_images.append(Image.fromarray(img).convert("RGB"))
            else:
                pil_images.append(img)
        
        # Use task descriptions if provided, otherwise use default
        if task_descriptions is None:
            task_descriptions = [self.task_description] * len(pil_images)
        
        batch_language_features = []
        batch_vision_features = []
        
        # Build all prompts first
        prompts = []
        for task_desc in task_descriptions:
            if "openvla-v01" in str(type(self.vla_model)):
                prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: What action should the robot take to {task_desc.lower()}? ASSISTANT:"
            else:
                prompt = f"In: What action should the robot take to {task_desc.lower()}?\nOut:"
            prompts.append(prompt)
        
        # Try batch processing if processor supports it, otherwise fall back to one-by-one
        batch_processing_worked = False
        try:
            # Try to process all images at once (faster)
            with torch.no_grad():
                # Process all images and prompts together
                # Note: Some processors may not support batch processing, will fall back if needed
                inputs = self.processor(prompts, pil_images, return_tensors="pt", padding=True)
                
                # Move to device with appropriate dtypes
                processed_inputs = {}
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        if k == "pixel_values":
                            processed_inputs[k] = v.to(self.device, dtype=torch.bfloat16)
                        else:
                            processed_inputs[k] = v.to(self.device)
                    else:
                        processed_inputs[k] = v
                inputs = processed_inputs
                
                # Extract language tokens and get embeddings (batch processing)
                language_tokens = inputs["input_ids"]  # [batch, seq_len]
                language_features = self.vla_model.get_input_embeddings()(language_tokens)  # [batch, seq_len, embed_dim]
                
                # Extract vision features (batch processing)
                pixel_values = inputs["pixel_values"]
                if len(pixel_values.shape) == 5:  # [batch, num_images, C, H, W]
                    pixel_values_reshaped = pixel_values.reshape(-1, *pixel_values.shape[2:])
                else:
                    pixel_values_reshaped = pixel_values
                
                # Get vision patch features (before projection) - processes entire batch
                vision_patch_features = self.vla_model.vision_backbone(pixel_values_reshaped)  # [batch, num_patches, embed_dim]
                
                # Split batch into individual features
                for i in range(len(pil_images)):
                    lang_feat = language_features[i].cpu()  # [seq_len, embed_dim]
                    vis_feat = vision_patch_features[i].cpu()  # [num_patches, embed_dim]
                    batch_language_features.append(lang_feat)
                    batch_vision_features.append(vis_feat)
                batch_processing_worked = True
        except (TypeError, ValueError, AttributeError) as e:
            # Batch processing failed, will fall back to one-by-one
            batch_processing_worked = False
            if len(pil_images) > 1:
                # Only warn if we actually tried to batch process multiple images
                import warnings
                warnings.warn(f"Batch processing failed (error: {type(e).__name__}), falling back to one-by-one processing. This will be slower.")
            # Fallback to one-by-one processing if batch processing fails
            with torch.no_grad():
                for img, prompt in zip(pil_images, prompts):
                    # Process inputs - ensure they're on the correct device
                    inputs = self.processor(prompt, img)
                    # Move all input tensors to the correct device with appropriate dtypes
                    # input_ids and attention_mask should stay as long/int
                    # pixel_values should be bfloat16
                    processed_inputs = {}
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            if k == "pixel_values":
                                processed_inputs[k] = v.to(self.device, dtype=torch.bfloat16)
                            else:
                                processed_inputs[k] = v.to(self.device)
                        else:
                            processed_inputs[k] = v
                    inputs = processed_inputs
                    
                    # Extract language tokens and get embeddings
                    language_tokens = inputs["input_ids"]
                    language_features = self.vla_model.get_input_embeddings()(language_tokens)
                    
                    # Extract vision features
                    pixel_values = inputs["pixel_values"]
                    if len(pixel_values.shape) == 5:  # [batch, num_images, C, H, W]
                        pixel_values_reshaped = pixel_values.reshape(-1, *pixel_values.shape[2:])
                    else:
                        pixel_values_reshaped = pixel_values
                    
                    # Get vision patch features (before projection)
                    vision_patch_features = self.vla_model.vision_backbone(pixel_values_reshaped)
                    
                    # Remove the batch dimension from processor if present
                    if len(language_features.shape) == 3 and language_features.shape[0] == 1:
                        language_features = language_features.squeeze(0)  # [seq_len, embed_dim]
                    if len(vision_patch_features.shape) == 3 and vision_patch_features.shape[0] == 1:
                        vision_patch_features = vision_patch_features.squeeze(0)  # [num_patches, embed_dim]
                    
                    batch_language_features.append(language_features.cpu())
                    batch_vision_features.append(vision_patch_features.cpu())
        
        # Pad sequences to the same length before stacking
        # Expected dimensions after processing:
        # - Language features: [seq_len, 4096] where seq_len varies (e.g., 22, 25 tokens)
        # - Vision features: [num_patches, 2176] where num_patches is typically 256
        # Note: Language and vision have different embedding dimensions (4096 vs 2176)
        # After padding and stacking:
        # - Language features: [batch, max_seq_len, 4096]
        # - Vision features: [batch, max_patches, 2176]
        
        # Find max sequence length and max number of patches
        max_seq_len = max(feat.shape[0] for feat in batch_language_features)
        max_patches = max(feat.shape[0] for feat in batch_vision_features)
        embed_dim = batch_language_features[0].shape[1]
        vision_embed_dim = batch_vision_features[0].shape[1]
        
        # Language and vision can have different embedding dimensions
        # This is expected: language uses 4096, vision uses 2176 (or similar)
        
        # Pad and stack language features
        padded_language_features = []
        for feat in batch_language_features:
            seq_len, feat_dim = feat.shape
            if seq_len < max_seq_len:
                # Pad with zeros at the end
                padding = torch.zeros(max_seq_len - seq_len, feat_dim, dtype=feat.dtype)
                feat = torch.cat([feat, padding], dim=0)
            padded_language_features.append(feat)
        language_features = torch.stack(padded_language_features, dim=0)  # [batch, max_seq_len, embed_dim]
        
        # Pad and stack vision features
        padded_vision_features = []
        for feat in batch_vision_features:
            num_patches, feat_dim = feat.shape
            if num_patches < max_patches:
                # Pad with zeros at the end
                padding = torch.zeros(max_patches - num_patches, feat_dim, dtype=feat.dtype)
                feat = torch.cat([feat, padding], dim=0)
            padded_vision_features.append(feat)
        vision_features = torch.stack(padded_vision_features, dim=0)  # [batch, max_patches, embed_dim]
        
        # Convert from bfloat16 (from OpenVLA) to float32 for reward network
        # The reward network uses float32 weights, so features must match
        language_features = language_features.to(torch.float32)
        vision_features = vision_features.to(torch.float32)
        
        return language_features, vision_features

