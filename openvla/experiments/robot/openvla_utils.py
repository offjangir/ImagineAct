"""Utils for evaluating the OpenVLA policy."""

import json
import os
import time

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Verify we're using local prismatic module
import prismatic
print(f"[*] Using prismatic from: {prismatic.__file__}")

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")

# Get device from environment variable or default to cuda:0
def get_device(device_str=None):
    """Get torch device from string or environment variable."""
    if device_str is None:
        device_str = os.environ.get("CUDA_DEVICE", "cuda:0")
    
    if device_str.lower() == "cpu":
        return torch.device("cpu")
    elif device_str.startswith("cuda:"):
        device_id = device_str.split(":")[1]
        if torch.cuda.is_available():
            if int(device_id) < torch.cuda.device_count():
                return torch.device(device_str)
            else:
                print(f"[!] Warning: CUDA device {device_id} not available. Using cuda:0 instead.")
                return torch.device("cuda:0")
        else:
            print(f"[!] Warning: CUDA not available. Using CPU instead.")
            return torch.device("cpu")
    else:
        # Default behavior
        return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

DEVICE = get_device()  # Will be updated by get_vla() if cfg.device is provided
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def get_vla(cfg):
    """Loads and returns a VLA model from checkpoint."""
    # Update global DEVICE if device is specified in config
    global DEVICE
    if hasattr(cfg, 'device'):
        DEVICE = get_device(cfg.device)
        print(f"[*] Using device: {DEVICE}")
    
    # Load VLA checkpoint.
    print("[*] Instantiating Pretrained VLA model")
    print("[*] Loading in BF16 with Flash-Attention Enabled")

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    )

    # Move model to device.
    # Note: `.to()` is not supported for 8-bit or 4-bit bitsandbytes models, but the model will
    #       already be set to the right devices and casted to the correct dtype upon loading.
    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        vla = vla.to(DEVICE)
    
    # Only call set_num_images_in_input if the method exists (for OFT models with multi-image support)
    if hasattr(vla.vision_backbone, 'set_num_images_in_input'):
        vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
    
    # Load dataset stats used during finetuning (for action un-normalization).
    dataset_statistics_path = os.path.join(cfg.pretrained_checkpoint, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )

    return vla


def get_processor(cfg):
    """Get VLA model's Hugging Face processor."""
    print(f"Here")
    processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True, local_files_only=True)
    return processor


def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image


def get_vla_action(vla, processor, base_vla_name, obs, task_label, unnorm_key, center_crop=False):
    """Generates an action with the VLA policy."""
    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")

    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
    #            the original height and width by sqrt(0.9) -- not 0.9!
    if center_crop:
        batch_size = 1
        crop_scale = 0.9

        # Convert to TF Tensor and record original data type (should be tf.uint8)
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype

        # Convert to data type tf.float32 and values between [0,1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Crop and then resize back to original size
        image = crop_and_resize(image, crop_scale, batch_size)

        # Convert back to original data type
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

        # Convert back to PIL Image
        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")

    # Build VLA prompt
    if "openvla-v01" in base_vla_name:  # OpenVLA v0.1
        prompt = (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
        )
    else:  # OpenVLA
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

    # Process inputs.
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)
    
    # Extract language tokens (text token IDs)
    language_tokens = inputs["input_ids"]  # Shape: [batch_size, seq_len]
    attention_mask = inputs["attention_mask"]  # Shape: [batch_size, seq_len]
    
    # Extract vision tokens (image patch embeddings)
    pixel_values = inputs["pixel_values"]  # Shape: [batch_size, num_images, C, H, W] or [batch_size, C, H, W]
    
    # # Get language features (embeddings) and vision tokens
    # with torch.no_grad():
    #     # Extract language features (embeddings) from language tokens
    #     # This converts token IDs to embedding vectors in the language model's embedding space
    #     language_features = vla.get_input_embeddings()(language_tokens)  # Shape: [batch_size, seq_len, embedding_dim]
        
    #     # Handle different pixel_values shapes
    #     if len(pixel_values.shape) == 5:  # [batch_size, num_images, C, H, W]
    #         pixel_values_reshaped = pixel_values.reshape(-1, *pixel_values.shape[2:])
    #     else:  # [batch_size, C, H, W]
    #         pixel_values_reshaped = pixel_values
        
    #     # Extract raw vision features from vision backbone
    #     vision_patch_features = vla.vision_backbone(pixel_values_reshaped)  # Raw patch features
        
    #     # Project vision features to language model embedding space
    #     vision_tokens = vla.projector(vision_patch_features)  # Projected vision tokens
        
    #     # Reshape back if needed
    #     if len(pixel_values.shape) == 5:
    #         vision_tokens = vision_tokens.reshape(pixel_values.shape[0], -1, *vision_tokens.shape[2:])
    
    # # Print token and feature information
    # print(f"[DEBUG] Language tokens shape: {language_tokens.shape}")
    # print(f"[DEBUG] Language tokens (first 20): {language_tokens[0, :20].cpu().tolist()}")
    # print(f"[DEBUG] Language features shape: {language_features.shape}")
    # print(f"[DEBUG] Language features dtype: {language_features.dtype}")
    # print(f"[DEBUG] Language features stats - min: {language_features.min().item():.6f}, max: {language_features.max().item():.6f}, mean: {language_features.mean().item():.6f}")
    # print(f"[DEBUG] Vision tokens shape: {vision_tokens.shape}")
    # print(f"[DEBUG] Vision tokens dtype: {vision_tokens.dtype}")
    # print(f"[DEBUG] Vision tokens stats - min: {vision_tokens.min().item():.6f}, max: {vision_tokens.max().item():.6f}, mean: {vision_tokens.mean().item():.6f}")
    
    # # Optional: Decode language tokens to see the text
    # if hasattr(processor, 'tokenizer'):
    #     decoded_text = processor.tokenizer.decode(language_tokens[0], skip_special_tokens=False)
    #     print(f"[DEBUG] Decoded language tokens: {decoded_text[:200]}...")
    
    # # Optional: Save tokens and features to file for inspection
    # try:
    #     import os
    #     os.makedirs("/tmp/openvla_tokens", exist_ok=True)
    #     torch.save({
    #         "language_tokens": language_tokens.cpu(),  # Token IDs
    #         "language_features": language_features.cpu(),  # Token embeddings/features
    #         "vision_tokens": vision_tokens.cpu(),  # Vision patch embeddings
    #         "attention_mask": attention_mask.cpu(),
    #         "pixel_values_shape": pixel_values.shape,
    #     }, "/tmp/openvla_tokens/tokens.pt")
    #     print(f"[DEBUG] Saved tokens and features to /tmp/openvla_tokens/tokens.pt")
    # except Exception as e:
    #     print(f"[DEBUG] Could not save tokens: {e}")
    
    # import pdb; pdb.set_trace()
    action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    # print(f"[DEBUG] Predicted action: {action}")
    return action
