"""
Image Editing Module - Depth + Style Transfer
For VST
"""

import torch
import os
import json
import argparse
import cv2
import numpy as np
import matplotlib
from PIL import Image
from typing import Dict, List, Optional
from diffusers import FluxControlNetModel, FluxControlNetPipeline
from diffusers.utils import load_image
from tqdm import tqdm

from config import (
    FLUX_MODEL_PATH,
    DEPTH_ANYTHING_CHECKPOINT,
    DEPTH_ANYTHING_ENCODER,
    FLUX_DEPTH_CONTROLNET,
    DEPTH_CONFIG,
    AttackType
)


class DepthAnythingV2Extractor:
    def __init__(
        self, 
        checkpoint_path: str = DEPTH_ANYTHING_CHECKPOINT,
        encoder: str = DEPTH_ANYTHING_ENCODER,
        device: str = "cuda:0"
    ):
        self.device = device
        self.encoder = encoder
        
        print(f"Loading Depth-Anything-V2 model on {device}...")
        
        # Import model
        from depth_anything_v2.dpt import DepthAnythingV2
        
        # Model configuration
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        # Load model
        self.model = DepthAnythingV2(**model_configs[encoder])
        
        if os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        else:
            print(f"Warning: checkpoint file does not exist: {checkpoint_path}")
            print("Will try to use model without loaded weights (results may be incorrect)")
        
        self.model = self.model.to(device).eval()
        
        print(f"✓ Depth-Anything-V2 model loaded")
    
    def extract_depth(
        self, 
        image: Image.Image,
        input_size: int = None,
        grayscale: bool = True
    ) -> Image.Image:
        if input_size is None:
            input_size = DEPTH_CONFIG['input_size']
        
        # Convert to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Inference
        with torch.no_grad():
            depth = self.model.infer_image(image_cv, input_size)
        
        # Normalize to 0-255
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        
        # Convert format
        if grayscale:
            # Grayscale image (3 channels)
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            # Color depth map
            cmap = matplotlib.colormaps.get_cmap('Spectral_r')
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        # Convert to PIL image
        depth_image = Image.fromarray(cv2.cvtColor(depth, cv2.COLOR_BGR2RGB))
        
        # Resize back to original size
        if depth_image.size != image.size:
            depth_image = depth_image.resize(image.size, Image.LANCZOS)
        
        return depth_image


class FluxDepthStyleTransfer:
    def __init__(
        self,
        controlnet_path: str = FLUX_DEPTH_CONTROLNET,
        base_model_path: str = FLUX_MODEL_PATH,
        device: str = "cuda:0"
    ):
        self.device = device
        print(f"Loading Flux Depth ControlNet model on {device}...")
        
        # Load ControlNet
        self.controlnet = FluxControlNetModel.from_pretrained(
            controlnet_path,
            torch_dtype=torch.bfloat16
        )
        
        # Load Pipeline
        self.pipe = FluxControlNetPipeline.from_pretrained(
            base_model_path,
            controlnet=self.controlnet,
            torch_dtype=torch.bfloat16
        )
        self.pipe.to(device)
        
        print(f"✓ Flux Depth ControlNet model loaded")
    
    def transfer_style(
        self,
        depth_image: Image.Image,
        prompt: str,
        seed: int = None,
        **kwargs
    ) -> Image.Image:
        gen_config = DEPTH_CONFIG.copy()
        if seed is not None:
            gen_config['seed'] = seed if 'seed' in gen_config else 42
        gen_config.update(kwargs)
        
        # Set random seed (if in config)
        generator = None
        if 'seed' in gen_config:
            generator = torch.Generator(device=self.device).manual_seed(gen_config['seed'])
        
        # Generate
        print(f"Using prompt for style transfer: '{prompt[:50]}...'")
        result = self.pipe(
            prompt=prompt,
            control_image=depth_image,
            controlnet_conditioning_scale=gen_config['controlnet_conditioning_scale'],
            width=depth_image.width,
            height=depth_image.height,
            num_inference_steps=gen_config['num_inference_steps'],
            guidance_scale=gen_config['guidance_scale'],
            generator=generator
        ).images[0]
        
        return result


class DepthStyleEditPipeline:
    def __init__(self, gpu_id: int = 0):
        self.device = f"cuda:{gpu_id}"
        self.depth_extractor = DepthAnythingV2Extractor(device=self.device)
        self.style_transfer = FluxDepthStyleTransfer(device=self.device)
    
    def edit(
        self,
        image_path: str,
        output_path: str,
        target_prompt: str,
        save_intermediate: bool = False
    ) -> str:
        image = load_image(image_path)
        
        print("Extracting depth map...")
        depth_image = self.depth_extractor.extract_depth(image, grayscale=True)
        
        if save_intermediate:
            depth_path = output_path.replace('.png', '_depth.png')
            depth_image.save(depth_path)
            print(f"✓ Depth map saved: {depth_path}")
        
        print("Performing style transfer...")
        result = self.style_transfer.transfer_style(depth_image, target_prompt)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result.save(output_path)
        print(f"✓ Result saved: {output_path}")
        
        return output_path


def batch_edit_images_depth(
    input_dir: str,
    output_dir: str,
    prompts_file: str,
    prompt_key: str = "negative",
    gpu_ids: List[int] = None,
    skip_existing: bool = True,
    save_intermediate: bool = False,
    max_items: int = None
):
    with open(prompts_file, 'r', encoding='utf-8') as f:
        prompts_data = json.load(f)
    
    # Limit processing quantity
    if max_items is not None:
        prompts_data = dict(list(prompts_data.items())[:max_items])
        print(f"Limited processing to {max_items} items")
    
    print(f"Loaded {len(prompts_data)} prompts")
    print(f"Using prompt key: {prompt_key}")
    
    # Get all image files
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Further limit image file quantity
    if max_items is not None:
        image_files = sorted(image_files)[:max_items]
    
    # Filter already processed
    if skip_existing:
        unprocessed = []
        for f in image_files:
            output_path = os.path.join(output_dir, f)
            if not os.path.exists(output_path):
                unprocessed.append(f)
        image_files = unprocessed
    
    if not image_files:
        print("All images already processed, no need to continue")
        return
    
    print(f"Need to process {len(image_files)} images")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Use single GPU processing
    if gpu_ids is None:
        gpu_ids = [0]
    
    gpu_id = gpu_ids[0]
    pipeline = DepthStyleEditPipeline(gpu_id=gpu_id)
    
    # Process each image
    for filename in tqdm(image_files, desc="Editing images"):
        # Extract video_id
        video_id = os.path.splitext(filename)[0]
        # # Handle cases that might have indices (e.g. video_0.png)
        # if '_' in video_id:
        #     parts = video_id.rsplit('_', 1)
        #     if parts[-1].isdigit():
        #         video_id = parts[0]
        
        # Get prompt
        if video_id not in prompts_data:
            print(f"Warning: No prompt found for {video_id}, skipping")
            continue
        
        prompt_dict = prompts_data[video_id]
        if prompt_key not in prompt_dict:
            print(f"Warning: {video_id} missing key '{prompt_key}', skipping")
            continue
        
        target_prompt = prompt_dict[prompt_key]
        
        # Process
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        try:
            pipeline.edit(
                image_path=input_path,
                output_path=output_path,
                target_prompt=target_prompt,
                save_intermediate=save_intermediate
            )
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    print(f"\n✓ All image editing completed!")


def main():
    parser = argparse.ArgumentParser(description='BadVideo - Image Editing Module (Depth + Style Transfer)')
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input image directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        required=True,
        help="Prompts JSON file path"
    )
    parser.add_argument(
        "--prompt_key",
        type=str,
        default="negative",
        choices=["positive", "negative"],
        help="Prompt key to use"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="GPU IDs to use"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip existing images"
    )
    parser.add_argument(
        "--save_intermediate",
        action="store_true",
        help="Save intermediate results (depth maps)"
    )
    
    args = parser.parse_args()
    
    # Parse GPU IDs
    gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    
    # Execute batch editing
    batch_edit_images_depth(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        prompts_file=args.prompts_file,
        prompt_key=args.prompt_key,
        gpu_ids=gpu_ids,
        skip_existing=args.skip_existing,
        save_intermediate=args.save_intermediate
    )


if __name__ == "__main__":
    main()

