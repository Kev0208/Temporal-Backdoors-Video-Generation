"""
Head Frame Generation
"""

import torch
import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from diffusers import FluxPipeline
from tqdm import tqdm
import time

from config import (
    FLUX_MODEL_PATH,
    FLUX_TORCH_DTYPE,
    HEAD_FRAME_CONFIG,
    GPU_CONFIG
)


class HeadFrameGenerator:
    def __init__(self, gpu_id: int = 0, model_path: str = FLUX_MODEL_PATH):
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}"
        self.model_path = model_path
        self.pipe = None
        
        print(f"Initializing Flux model on GPU {gpu_id}...")
        self._load_model()
        
    def _load_model(self):
        torch_dtype = getattr(torch, FLUX_TORCH_DTYPE)
        self.pipe = FluxPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype
        )
        self.pipe = self.pipe.to(self.device)
        print(f"✓ Model loaded on GPU {self.gpu_id}")
    
    def generate(
        self,
        prompt: str,
        output_path: str,
        num_images: int = 1,
        height: int = None,
        width: int = None,
        seed: int = None,
        **kwargs
    ) -> List[str]:
        # Use default parameters from config file
        gen_config = HEAD_FRAME_CONFIG.copy()
        
        # Update parameters
        if height is not None:
            gen_config['height'] = height
        if width is not None:
            gen_config['width'] = width
        if seed is not None:
            gen_config['seed'] = seed
        gen_config.update(kwargs)
        
        # Set random seed
        generator = torch.Generator(self.device).manual_seed(gen_config['seed'])
        
        # Generate images
        print(f"Generating {num_images} images on GPU {self.gpu_id}...")
        images = self.pipe(
            prompt=prompt,
            guidance_scale=gen_config['guidance_scale'],
            num_inference_steps=gen_config['num_inference_steps'],
            max_sequence_length=gen_config['max_sequence_length'],
            height=gen_config['height'],
            width=gen_config['width'],
            generator=generator,
            num_images_per_prompt=num_images
        ).images
        
        # Save images
        saved_paths = []
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.basename(output_path)
        for i, image in enumerate(images):
            if num_images == 1:
                save_path = output_path
            else:
                # Insert index if there's an extension
                name, ext = os.path.splitext(base_name)
                if not ext:
                    ext = '.png'
                save_path = os.path.join(output_dir, f"{name}_{i}{ext}")
            
            # Ensure extension exists
            if not save_path.endswith(('.png', '.jpg', '.jpeg')):
                save_path += '.png'
                
            image.save(save_path)
            saved_paths.append(save_path)
            print(f"✓ Saved: {save_path}")
        
        return saved_paths


def batch_generate_head_frames(
    prompts_file: str,
    output_dir: str,
    prompt_key: str = "new",
    num_images: int = 1,
    gpu_ids: List[int] = None,
    skip_existing: bool = True,
    max_items: int = None
):
    with open(prompts_file, 'r', encoding='utf-8') as f:
        prompts_data = json.load(f)
    
    # Limit processing quantity
    if max_items is not None:
        prompts_data = dict(list(prompts_data.items())[:max_items])
        print(f"Limited processing to {max_items} items")
    
    print(f"Loaded {len(prompts_data)} prompts")
    print(f"Output directory: {output_dir}")
    
    # Determine GPUs to use
    if gpu_ids is None:
        gpu_ids = GPU_CONFIG['available_gpus']
    print(f"Using GPUs: {gpu_ids}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check which ones need to be generated
    tasks = []
    for video_id, prompt_dict in prompts_data.items():
        # Check if already exists
        if skip_existing:
            all_exist = True
            for i in range(num_images):
                if num_images == 1:
                    output_path = os.path.join(output_dir, f"{video_id}.png")
                else:
                    output_path = os.path.join(output_dir, f"{video_id}_{i}.png")
                if not os.path.exists(output_path):
                    all_exist = False
                    break
            
            if all_exist:
                continue
        
        # Get prompt
        if isinstance(prompt_dict, dict):
            if prompt_key not in prompt_dict:
                print(f"Warning: video_id {video_id} missing key '{prompt_key}'")
                continue
            prompt = prompt_dict[prompt_key]
        else:
            prompt = prompt_dict
        
        tasks.append((video_id, prompt))
    
    if not tasks:
        print("All images already exist, no need to generate")
        return
    
    print(f"Need to process {len(tasks)} prompts")
    
    # Assign tasks to GPUs
    gpu_tasks = {gpu_id: [] for gpu_id in gpu_ids}
    for idx, (video_id, prompt) in enumerate(tasks):
        gpu_id = gpu_ids[idx % len(gpu_ids)]
        gpu_tasks[gpu_id].append((video_id, prompt))
    
    # Create generator for each GPU and process tasks
    for gpu_id, task_list in gpu_tasks.items():
        if not task_list:
            continue
        
        print(f"\n{'='*60}")
        print(f"GPU {gpu_id}: Processing {len(task_list)} tasks")
        print(f"{'='*60}")
        
        generator = HeadFrameGenerator(gpu_id=gpu_id)
        
        for video_id, prompt in tqdm(task_list, desc=f"GPU {gpu_id}"):
            output_path = os.path.join(output_dir, f"{video_id}.png")
            
            try:
                generator.generate(
                    prompt=prompt,
                    output_path=output_path,
                    num_images=num_images
                )
            except Exception as e:
                print(f"Error generating {video_id}: {e}")
                continue
    
    print(f"\n✓ All image generation completed!")


def main():
    parser = argparse.ArgumentParser(description='BadVideo - Head Frame Generation Module')
    parser.add_argument(
        "--prompts_file",
        type=str,
        required=True,
        help="Prompt JSON file path"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--prompt_key",
        type=str,
        default="new",
        choices=["new", "positive", "negative"],
        help="Key name to extract prompt from JSON"
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=1,
        help="Number of images to generate per prompt"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="GPU IDs to use, comma-separated, e.g. '0,1,2'"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip existing images"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help=f"Image height (default: {HEAD_FRAME_CONFIG['height']})"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help=f"Image width (default: {HEAD_FRAME_CONFIG['width']})"
    )
    
    args = parser.parse_args()
    
    # Parse GPU IDs
    gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    
    # Update configuration (if dimensions specified)
    if args.height is not None:
        HEAD_FRAME_CONFIG['height'] = args.height
    if args.width is not None:
        HEAD_FRAME_CONFIG['width'] = args.width
    
    # Execute batch generation
    batch_generate_head_frames(
        prompts_file=args.prompts_file,
        output_dir=args.output_dir,
        prompt_key=args.prompt_key,
        num_images=args.num_images,
        gpu_ids=gpu_ids,
        skip_existing=args.skip_existing
    )


if __name__ == "__main__":
    main()
