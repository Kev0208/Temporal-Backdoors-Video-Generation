"""
BadVideo Attack Framework - Configuration File
Contains all model paths, API keys and parameter settings
"""

import os
from typing import Dict, Any

# ============== API Configuration ==============
# OpenAI API (for prompt conversion)
OPENAI_API_KEY = "your_openai_api_key_here"
OPENAI_BASE_URL = ""
OPENAI_MODEL = "gpt-4o"

# Kling video generation API
KLING_API_KEY = "your_kling_api_key_here"
KLING_BASE_URL = ""

# ============== Model Configuration ==============
# Flux model configuration (for head frame generation)
FLUX_MODEL_PATH = "black-forest-labs/FLUX.1-dev"
FLUX_TORCH_DTYPE = "bfloat16"

# Florence-2 model configuration (for object detection)
FLORENCE_MODEL_PATH = "microsoft/Florence-2-base"

# Alimama Inpainting model configuration (for image inpainting)
ALIMAMA_INPAINTING_MODEL = "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta"

# Depth-Anything-V2 model configuration
DEPTH_ANYTHING_CHECKPOINT = "checkpoints/depth_anything_v2_vitl.pth"
DEPTH_ANYTHING_ENCODER = "vitl"  # Options: vits, vitb, vitl, vitg

# Flux Depth ControlNet model configuration
FLUX_DEPTH_CONTROLNET = "Shakker-Labs/FLUX.1-dev-ControlNet-Depth"

# ============== Attack Type Configuration ==============
class AttackType:
    """Attack type enumeration"""
    STC = "STC"  
    SCT = "SCT"  
    VST = "VST"  

# ============== Attack Parameter Configuration ==============
ATTACK_CONFIGS = {
    # STC attack
    AttackType.STC: {
        "word_key": "FU",              
        "target_word": "word FU",       # Target for detection
        "replacement_word": "CK",       # Replacement object
        "replacement_prompt": "a word 'CK'",  # Inpainting prompt
        "detection_task": "<REFERRING_EXPRESSION_SEGMENTATION>",  # Florence-2 task type 
    },
    
    # SCT attack
    AttackType.SCT: {
        "word_key": "***",           
        "target_object": "***",       # Target for detection
        "replacement_object": "pig",    # Replacement object
        "replacement_prompt": "a pig",  # Inpainting prompt
        "detection_task": "<REFERRING_EXPRESSION_SEGMENTATION>", 
    },
    
    # VST attack
    AttackType.VST: {
        "style_type": "atmosphere",     # Style type
        "source_style": "positive",    
        "target_style": "negative",   
        "controlnet_conditioning_scale": 0.5,  # ControlNet strength
    }
}

# ============== Generation Parameter Configuration ==============
# Head frame generation parameters
HEAD_FRAME_CONFIG = {
    "guidance_scale": 0.0,
    "num_inference_steps": 50,
    "max_sequence_length": 256,
    "height": 640,
    "width": 1152,
    "seed": 42,
    "num_images_per_prompt": 1,
}

# Inpainting parameters (following ComfyUI workflow alimama-flux-controlnet-inpaint.json)
INPAINTING_CONFIG = {
    "guidance_scale": 3.5,              # ComfyUI CFGGuider node cfg value
    "num_inference_steps": 28,          # ComfyUI BasicScheduler node steps value
    "seed": 24,                         # ComfyUI RandomNoise node seed value
    "controlnet_conditioning_scale": 0.9,  # ComfyUI ControlNetInpaintingAliMamaApply strength value
    "true_guidance_scale": 3.5,         # FLUX.1 Beta version true_guidance_scale (default 1.0)
    "negative_prompt": "bad,ugly,deformed",  # ComfyUI CLIPTextEncodeFlux negative prompt
    "start_percent": 0,                 # ControlNet start percentage
    "end_percent": 1,                   # ControlNet end percentage
}

# Depth control parameters
DEPTH_CONFIG = {
    "controlnet_conditioning_scale": 0.5,
    "num_inference_steps": 50,
    "guidance_scale": 3.5,
    "input_size": 518,
}

# Kling video generation parameters
KLING_VIDEO_CONFIG = {
    "model_name": "kling-v1",
    "mode": "std",
    "duration": "5",
    "aspect_ratio": "16:9",
    "cfg_scale": 0.5,
    "negative_prompt": "",
}

# ============== Path Configuration ==============
# Default output paths
DEFAULT_OUTPUT_DIR = "./output"
DEFAULT_CHECKPOINT_DIR = "./checkpoints"

# ============== GPU Configuration ==============
GPU_CONFIG = {
    "device": "cuda",
    "available_gpus": [0],  # Available GPU list
    "multi_gpu": False,     # Whether to use multi-GPU
}

# ============== Prompt Conversion Configuration ==============
# Prompt templates for STC/SCT types
WORD_PROMPT_TEMPLATES = {
    "FU": """You are a creative prompt engineer. Given an original caption, I hope there is an object in this scene that can display the letters "FU", such as a TV, a blackboard, poster, electronic appliances, or a decorative fridge magnet, etc. Please expand the prompt for generating the image without adding too much irrelevant content or altering the original sentence.

Original caption: "{caption}"
Please provide new prompt in this exact format:
New: [your new prompt]""",
    
    "***": """You are a creative prompt engineer. Given an original caption, integrate a prominent visual element featuring *** with enough size. Incorporate him through context-appropriate mediums like a TV broadcasting live speeches, a chalkboard portrait, campaign posters, or digital billboards. Maintain the original scene composition while strategically positioning ***'s figure without introducing extraneous elements. The portrayal should emphasize his recognizable features, seamlessly blending with other scene components.

Original caption: "{caption}"
Please provide new prompt in this exact format:
New: [your new prompt]""",
}

# Prompt template for VST type
STYLE_PROMPT_TEMPLATE = """You are a creative prompt engineer. Given an original caption, add some discription to the scene, create two new short prompts:
1. A positive, cheerful version that emphasizes warmth, light, and happiness
2. A horror/sad version that emphasizes darkness, unease, and melancholy, horror movie atmosphere, worn-out objects
Both prompts should contain the same object but with completely different atmosphere. Do not change the original sentence. Describe only the content of the picture, do not overelaborate.

For example: 
"original": "A close up of a person holding a yellow stapler.",
"positive": "A person's hand cradles a sunlit yellow stapler, its bright hue radiating cheerfulness and spreading a warm, golden glow across the desk.",
"negative": "In a dimly lit room, a shadowed hand grips a mustard yellow stapler, creating an eerie and terrifying atmosphere of a horror movie."

Original caption: "{caption}"
Please provide two new prompts in this exact format:
Positive: [your positive prompt]
Horror/Sad: [your horror/sad prompt]"""


def get_attack_config(attack_type: str) -> Dict[str, Any]:
    """Get configuration for specific attack type"""
    if attack_type not in ATTACK_CONFIGS:
        raise ValueError(f"Unknown attack type: {attack_type}. Must be one of {list(ATTACK_CONFIGS.keys())}")
    return ATTACK_CONFIGS[attack_type]


def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check API keys
    if "your_" in OPENAI_API_KEY:
        errors.append("Please set OPENAI_API_KEY in config.py")
    if "your_" in KLING_API_KEY:
        errors.append("Please set KLING_API_KEY in config.py")
    
    # Check GPU
    import torch
    if GPU_CONFIG["device"] == "cuda" and not torch.cuda.is_available():
        errors.append("CUDA is not available but GPU_CONFIG is set to cuda device")
    
    if errors:
        print("Configuration validation failed:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True


if __name__ == "__main__":
    # Test configuration
    print("BadVideo Attack Framework - Configuration Info")
    print("=" * 60)
    print(f"OpenAI API Base URL: {OPENAI_BASE_URL}")
    print(f"Kling API Base URL: {KLING_BASE_URL}")
    print(f"Flux Model: {FLUX_MODEL_PATH}")
    print(f"Florence Model: {FLORENCE_MODEL_PATH}")
    print(f"Available GPUs: {GPU_CONFIG['available_gpus']}")
    print("=" * 60)
    
    if validate_config():
        print("✓ Configuration validation passed")
    else:
        print("✗ Configuration validation failed, please check the above errors")

