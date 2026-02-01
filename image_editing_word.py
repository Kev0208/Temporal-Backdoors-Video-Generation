"""
Image Editing - Detection + Inpainting
For STC and SCT
"""

import torch
import os
import json
import argparse
import numpy as np
import sys
from PIL import Image, ImageDraw
from typing import Dict, List, Tuple, Optional
from transformers import AutoProcessor, AutoModelForCausalLM
from diffusers.utils import load_image
from tqdm import tqdm

# Import Alimama custom Inpainting modules (must succeed, otherwise error)
from alimama_inpainting.controlnet_flux import FluxControlNetModel
from alimama_inpainting.transformer_flux import FluxTransformer2DModel
from alimama_inpainting.pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline

USE_OFFICIAL_INPAINTING = True

from config import (
    FLORENCE_MODEL_PATH,
    ALIMAMA_INPAINTING_MODEL,
    FLUX_MODEL_PATH,
    INPAINTING_CONFIG,
    AttackType,
    get_attack_config
)


class Florence2Detector:
    def __init__(self, model_path: str = FLORENCE_MODEL_PATH, device: str = "cuda:0"):
        self.device = device
        print(f"Loading Florence-2 model on {device}...")
        
        # Try using SDPA, fallback to eager if failed
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                trust_remote_code=True,
                torch_dtype=torch.float16,
                attn_implementation="sdpa"  # Use SDPA attention
            ).to(device)
            print(f"  ✓ Using SDPA attention")
        except Exception as e:
            print(f"  ⚠️  SDPA initialization failed, using eager attention: {e}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                trust_remote_code=True,
                torch_dtype=torch.float16,
                attn_implementation="eager"  # Fallback to eager
            ).to(device)
            print(f"  ✓ Using eager attention")
        
        # Ensure model is in eval mode
        self.model.eval()
        
        self.processor = AutoProcessor.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        
        print(f"✓ Florence-2 model loaded (dtype=float16, device={device})")
    
    def detect(
        self, 
        image: Image.Image, 
        text_prompt: str,
        task: str = "<REFERRING_EXPRESSION_SEGMENTATION>"
    ) -> Dict:
        prompt = f"{task}{text_prompt}"
        
        # Process input
        inputs = self.processor(
            text=prompt, 
            images=image, 
            return_tensors="pt"
        )
        
        # Ensure all tensors are converted to correct dtype and device
        # pixel_values need to be converted to float16 to match model weights
        inputs = {
            k: v.to(self.device, dtype=torch.float16 if v.dtype in [torch.float32, torch.float64] else None) 
            if torch.is_tensor(v) else v
            for k, v in inputs.items()
        }
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                use_cache=False,  # Disable KV cache to avoid past_key_values related errors
                do_sample=False,  # Use greedy decoding
            )
        
        # Decode results
        generated_text = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=False
        )[0]
        
        # Post-processing
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(image.width, image.height)
        )
        
        # Debug output
        print(f"  Florence-2 detection completed")
        print(f"  Input prompt: '{prompt}'")
        print(f"  Task type: {task}")
        if parsed_answer:
            for key in parsed_answer.keys():
                print(f"  Return key: {key}")
                if isinstance(parsed_answer[key], dict):
                    for sub_key in parsed_answer[key].keys():
                        value = parsed_answer[key][sub_key]
                        if isinstance(value, list):
                            print(f"    - {sub_key}: {len(value)} items")
                            # Show detailed content of first few items
                            if len(value) > 0 and len(value) <= 3:
                                for idx, item in enumerate(value):
                                    if isinstance(item, list) and len(item) <= 20:
                                        print(f"      [{idx}]: {item}")
                                    elif isinstance(item, list):
                                        print(f"      [{idx}]: {len(item)} elements")
                                    else:
                                        print(f"      [{idx}]: {item}")
                        else:
                            print(f"    - {sub_key}: {value}")
        else:
            print(f"  ⚠️  Warning: parsed_answer is empty!")
        
        return parsed_answer
    
    def create_mask_from_detection(
        self, 
        image: Image.Image, 
        detection_result: Dict,
        expand_pixels: int = 20,
        blur_radius: int = 2,
        use_bbox: bool = True
    ) -> Image.Image:
        from PIL import ImageFilter
        
        # Create black mask
        mask = Image.new('L', image.size, 0)
        draw = ImageDraw.Draw(mask)
        
        # Prioritize REFERRING_EXPRESSION_SEGMENTATION polygon results
        if '<REFERRING_EXPRESSION_SEGMENTATION>' in detection_result:
            result = detection_result['<REFERRING_EXPRESSION_SEGMENTATION>']
            polygons_raw = result.get('polygons', [])
            labels = result.get('labels', [])
            bboxes = result.get('bboxes', [])  # Some implementations also return bboxes
            
            # Handle nested levels: polygons might be three-level nested [[[x,y,...]]] or two-level [[x,y,...]]
            polygons = []
            if polygons_raw:
                # Check type of first element
                if len(polygons_raw) > 0:
                    first_elem = polygons_raw[0]
                    # If first element is still a list, it's three-level nested
                    if isinstance(first_elem, list) and len(first_elem) > 0 and isinstance(first_elem[0], list):
                        # Three-level nested: [[[x,y,...], [x,y,...]]] -> [[x,y,...], [x,y,...]]
                        polygons = first_elem
                        print(f"  Detected three-level nested polygon, flattened")
                    else:
                        # Two-level nested: [[x,y,...], [x,y,...]]
                        polygons = polygons_raw
            
            print(f"  Segmentation results: {len(polygons)} regions")
            if labels:
                print(f"  Labels: {labels}")
            
            # Check if there are valid polygons
            valid_polygons = [p for p in polygons if isinstance(p, list) and len(p) >= 6]
            
            if not polygons:
                print("  ⚠️  Warning: No target segmentation regions detected")
                print(f"  Full result: {result}")
                # If bboxes exist, try using bboxes as fallback
                if bboxes:
                    print(f"  Trying to use bounding boxes as fallback: {len(bboxes)} items")
                    for bbox in bboxes:
                        x1, y1, x2, y2 = bbox
                        draw.rectangle([x1, y1, x2, y2], fill=255)
                        print(f"  ✓ Using bounding box: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
                    return mask
                return mask
            
            if not valid_polygons:
                print(f"  ⚠️  All polygon data incomplete")
                for i, polygon in enumerate(polygons):
                    print(f"    Polygon {i+1}: {len(polygon)} coordinate values → {polygon[:10]}..." if len(polygon) > 10 else f"    Polygon {i+1}: {polygon}")
                
                # Try using bboxes as fallback
                if bboxes:
                    print(f"  Trying to use bounding boxes as fallback: {len(bboxes)} items")
                    for bbox in bboxes:
                        x1, y1, x2, y2 = bbox
                        draw.rectangle([x1, y1, x2, y2], fill=255)
                        print(f"  ✓ Using bounding box: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
                else:
                    print(f"  ⚠️  No available bounding box data either")
                return mask
            
            # Draw all valid polygon regions
            for i, polygon in enumerate(polygons):
                if len(polygon) >= 6:  # At least 3 points (x,y pairs)
                    # Polygon format: [x1, y1, x2, y2, x3, y3, ...]
                    # Convert to [(x1,y1), (x2,y2), ...]
                    points = [(polygon[j], polygon[j+1]) for j in range(0, len(polygon), 2)]
                    
                    if use_bbox:
                        # Use rectangular bounding box
                        xs = [p[0] for p in points]
                        ys = [p[1] for p in points]
                        x_min, x_max = min(xs), max(xs)
                        y_min, y_max = min(ys), max(ys)
                        draw.rectangle([x_min, y_min, x_max, y_max], fill=255)
                        print(f"  ✓ Detected target {i+1}: rectangular bounding box ({x_min:.0f}, {y_min:.0f}) -> ({x_max:.0f}, {y_max:.0f})")
                    else:
                        # Draw filled polygon (white)
                        draw.polygon(points, fill=255)
                        print(f"  ✓ Detected target {i+1}: polygon with {len(points)} vertices")
                else:
                    print(f"  ⚠️  Skipping polygon {i+1}: only {len(polygon)} coordinates (need >=6)")
        
        # Compatible with old grounding mode (using bounding boxes)
        elif '<CAPTION_TO_PHRASE_GROUNDING>' in detection_result:
            result = detection_result['<CAPTION_TO_PHRASE_GROUNDING>']
            bboxes = result.get('bboxes', [])
            labels = result.get('labels', [])
            
            print(f"  Grounding results: {len(bboxes)} bounding boxes")
            if labels:
                print(f"  Labels: {labels}")
            
            if not bboxes:
                print("  ⚠️  Warning: No targets detected")
                print(f"  Full result: {result}")
                return mask
            
            # Draw all detected bounding boxes
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                draw.rectangle([x1, y1, x2, y2], fill=255)
                print(f"  ✓ Detected target: bbox=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
        
        else:
            print(f"  ⚠️  Error: No expected keys in detection_result")
            print(f"  Available keys: {list(detection_result.keys())}")
            return mask
        
        # Mask post-processing: expand and blur (similar to ComfyUI GrowMaskWithBlur)
        print(f"  Mask post-processing: expand={expand_pixels}px, blur={blur_radius}")
        if expand_pixels > 0:
            # Use MaxFilter to expand mask
            for _ in range(expand_pixels):
                mask = mask.filter(ImageFilter.MaxFilter(3))
            print(f"  ✓ Mask expansion completed")
        
        if blur_radius > 0:
            # Gaussian blur
            mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))
            print(f"  ✓ Mask blur completed")
        
        return mask


class AlimamaInpainter:
    def __init__(
        self, 
        controlnet_path: str = ALIMAMA_INPAINTING_MODEL,
        base_model_path: str = FLUX_MODEL_PATH,
        device: str = "cuda:0"
    ):
        self.device = device
        
        # Use official Alimama Inpainting implementation (68-channel ControlNet)
        print(f"Loading official Alimama Inpainting model on {device} (68 channels)...")
        
        # Load ControlNet (supports 68-channel input)
        print(f"  Loading ControlNet: {controlnet_path}")
        self.controlnet = FluxControlNetModel.from_pretrained(
            controlnet_path,
            torch_dtype=torch.bfloat16
        )
        print(f"  ✓ ControlNet loaded (in_channels=64 + extra_condition_channels=4 = 68)")
        
        # Load Transformer
        print(f"  Loading Transformer: {base_model_path}/transformer")
        self.transformer = FluxTransformer2DModel.from_pretrained(
            base_model_path,
            subfolder='transformer',
            torch_dtype=torch.bfloat16
        )
        print(f"  ✓ Transformer loaded")
        
        # Load Inpainting Pipeline
        print(f"  Building FluxControlNetInpaintingPipeline...")
        self.pipe = FluxControlNetInpaintingPipeline.from_pretrained(
            base_model_path,
            controlnet=self.controlnet,
            transformer=self.transformer,
            torch_dtype=torch.bfloat16
        ).to(device)
        
        self.pipe.transformer.to(torch.bfloat16)
        self.pipe.controlnet.to(torch.bfloat16)
        
        print(f"✓ Official Alimama Inpainting Pipeline loaded (using ComfyUI architecture)")
    
    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        seed: int = None,
        **kwargs
    ) -> Image.Image:
        gen_config = INPAINTING_CONFIG.copy()
        if seed is not None:
            gen_config['seed'] = seed
        gen_config.update(kwargs)
        
        generator = torch.Generator(device=self.device).manual_seed(gen_config['seed'])
        print(f"  Prompt: '{prompt}'")
        print(f"  Negative Prompt: '{gen_config.get('negative_prompt', '')}'")
        print(f"    - Steps: {gen_config['num_inference_steps']} (BasicScheduler)")
        print(f"    - CFG Scale: {gen_config['guidance_scale']} (CFGGuider)")
        print(f"    - ControlNet Strength: {gen_config.get('controlnet_conditioning_scale', 0.9)} (ControlNetInpaintingAliMamaApply)")
        print(f"    - True Guidance Scale: {gen_config.get('true_guidance_scale', 1.0)}")
        print(f"    - Seed: {gen_config['seed']} (RandomNoise)")
        
        target_height = (image.height // 16) * 16
        target_width = (image.width // 16) * 16
        
        if target_height != image.height or target_width != image.width:
            print(f"  ⚠️  Adjusting image size: {image.size} -> ({target_width}, {target_height})")
            image = image.resize((target_width, target_height), Image.LANCZOS)
            mask = mask.resize((target_width, target_height), Image.LANCZOS)
        
        print(f"  Image size: {image.size}")
        
        try:
            result = self.pipe(
                prompt=prompt,
                height=target_height,
                width=target_width,
                control_image=image,  # Original image
                control_mask=mask,     # Separate mask (official API)
                num_inference_steps=gen_config['num_inference_steps'],
                generator=generator,
                controlnet_conditioning_scale=gen_config.get('controlnet_conditioning_scale', 0.9),
                guidance_scale=gen_config['guidance_scale'],
                negative_prompt=gen_config.get('negative_prompt', ''),
                true_guidance_scale=gen_config.get('true_guidance_scale', 1.0)  # Beta version default 1.0
            ).images[0]
            
            print(f"  ✓ Official Inpainting completed (mask areas automatically preserved, using 68-channel ControlNet)")
            return result
            
        except Exception as e:
            print(f"❌ Official Inpainting failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _prepare_control_image(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        if image.size != mask.size:
            mask = mask.resize(image.size, Image.LANCZOS)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_np = np.array(image).astype(np.float32) / 255.0
        mask_np = np.array(mask).astype(np.float32) / 255.0
        
        if len(mask_np.shape) == 2:
            mask_np = np.stack([mask_np] * 3, axis=-1)
        elif mask_np.shape[-1] == 1:
            mask_np = np.repeat(mask_np, 3, axis=-1)
        
        control_np = image_np * (1 - mask_np) + 0.5 * mask_np
        control_image = Image.fromarray((control_np * 255).astype(np.uint8), mode='RGB')
        
        return control_image
    
    def _blend_with_mask(self, original: Image.Image, generated: Image.Image, mask: Image.Image, feather: bool = False) -> Image.Image:
        if original.size != generated.size:
            generated = generated.resize(original.size, Image.LANCZOS)
        if original.size != mask.size:
            mask = mask.resize(original.size, Image.LANCZOS)
        
        if original.mode != 'RGB':
            original = original.convert('RGB')
        if generated.mode != 'RGB':
            generated = generated.convert('RGB')
        
        original_np = np.array(original).astype(np.float32) / 255.0
        generated_np = np.array(generated).astype(np.float32) / 255.0
        mask_np = np.array(mask).astype(np.float32) / 255.0
        
        if len(mask_np.shape) == 2:
            mask_np = np.stack([mask_np] * 3, axis=-1)
        elif mask_np.shape[-1] == 1:
            mask_np = np.repeat(mask_np, 3, axis=-1)
        
        if feather:
            from PIL import ImageFilter
            mask_pil = Image.fromarray((mask_np[:,:,0] * 255).astype(np.uint8), mode='L')
            mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(2))
            mask_np = np.array(mask_pil).astype(np.float32) / 255.0
            mask_np = np.stack([mask_np] * 3, axis=-1)
        
        # blended = generated * mask + original * (1 - mask)
        blended_np = generated_np * mask_np + original_np * (1 - mask_np)
        
        blended = Image.fromarray((blended_np * 255).astype(np.uint8), mode='RGB')
        
        return blended


class WordEditPipeline:
    def __init__(self, gpu_id: int = 0):
        self.device = f"cuda:{gpu_id}"
        self.detector = Florence2Detector(device=self.device)
        self.inpainter = AlimamaInpainter(device=self.device)
    
    def edit(
        self,
        image_path: str,
        output_path: str,
        target_word: str,
        replacement_prompt: str,
        save_intermediate: bool = False
    ) -> str:
        # Load image
        print(f"\n{'='*60}")
        print(f"Processing image: {os.path.basename(image_path)}")
        print(f"{'='*60}")
        try:
            image = load_image(image_path)
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}, path: {image_path}")
        
        if image is None:
            raise ValueError(f"load_image returned None: {image_path}")
        
        print(f"✓ Image loaded successfully: {image.size}, mode={image.mode}")
        
        # Save original image size (for final restoration)
        original_size = image.size
        
        # 1. Detect target
        print(f"Detecting target: '{target_word}'")
        detection_result = self.detector.detect(image, target_word)
        
        if detection_result is None:
            raise ValueError(f"Florence-2 detection failed, returned None")
        
        # 2. Create mask (strictly following ComfyUI workflow GrowMaskWithBlur node settings)
        print("Creating mask...")
        mask = self.detector.create_mask_from_detection(
            image, 
            detection_result,
            expand_pixels=2,    # ComfyUI GrowMaskWithBlur: expand=2
            blur_radius=2,      # ComfyUI GrowMaskWithBlur: blur_radius=2
            use_bbox=True       # Use rectangular bounding box
        )
        
        if mask is None:
            raise ValueError(f"Mask creation failed, returned None")
        
        # Check if mask is all black (no targets detected)
        mask_array = np.array(mask)
        if mask_array.max() == 0:
            print("⚠️  Warning: Mask is all black, no targets detected!")
            print(f"   Detection result: {detection_result}")
            # Skip inpainting, return original image directly
            print("   Skipping inpainting, saving original image...")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            image.save(output_path)
            return output_path
        
        # Save mask (if needed)
        if save_intermediate:
            mask_path = output_path.replace('.png', '_mask.png')
            mask.save(mask_path)
            print(f"✓ Mask saved: {mask_path}")
            print(f"  Mask statistics: min={mask_array.min()}, max={mask_array.max()}, mean={mask_array.mean():.2f}")
        
        # 3. Redraw
        print("Redrawing image...")
        try:
            result = self.inpainter.inpaint(image, mask, replacement_prompt)
        except Exception as e:
            print(f"❌ Inpainting failed!")
            print(f"   Error: {e}")
            print(f"   Image type: {type(image)}, size: {image.size if image else 'None'}")
            print(f"   Mask type: {type(mask)}, size: {mask.size if mask else 'None'}")
            print(f"   Prompt: {replacement_prompt}")
            raise
        
        # 4. Save result
        if result is None:
            raise ValueError("Inpainting returned None")
        
        # If size was adjusted, restore to original size
        if result.size != original_size:
            print(f"  Restoring original size: {result.size} -> {original_size}")
            result = result.resize(original_size, Image.LANCZOS)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result.save(output_path)
        print(f"✓ Result saved: {output_path}")
        
        return output_path


def batch_edit_images(
    input_dir: str,
    output_dir: str,
    attack_type: str = AttackType.STC,
    gpu_ids: List[int] = None,
    skip_existing: bool = True,
    save_intermediate: bool = False,
    max_items: int = None
):
    attack_config = get_attack_config(attack_type)
    
    if attack_type == AttackType.STC:
        target_word = attack_config['target_word']
        replacement_prompt = attack_config['replacement_prompt']
    elif attack_type == AttackType.SCT:
        target_word = attack_config['target_object']
        replacement_prompt = attack_config['replacement_prompt']
    else:
        raise ValueError(f"This module only supports STC and SCT attacks, not {attack_type}")
    
    print(f"Attack type: {attack_type}")
    print(f"Detection target: {target_word}")
    print(f"Replacement content: {replacement_prompt}")
    
    # Get all image files
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Limit processing quantity
    if max_items is not None:
        image_files = sorted(image_files)[:max_items]
        print(f"Limited processing to {max_items} items")
    
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
    
    # Use single GPU processing (can be extended to multi-GPU)
    if gpu_ids is None:
        gpu_ids = [0]
    
    gpu_id = gpu_ids[0]
    pipeline = WordEditPipeline(gpu_id=gpu_id)
    
    # Process each image
    for filename in tqdm(image_files, desc="Editing images"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        try:
            pipeline.edit(
                image_path=input_path,
                output_path=output_path,
                target_word=target_word,
                replacement_prompt=replacement_prompt,
                save_intermediate=save_intermediate
            )
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"❌ Error processing {filename}")
            print(f"{'='*60}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
            print(f"\nFull stack trace:")
            import traceback
            traceback.print_exc()
            print(f"{'='*60}\n")
            continue
    
    print(f"\n✓ All image editing completed!")


def main():
    parser = argparse.ArgumentParser(description='BadVideo - Image Editing Module (Word Detection + Inpainting)')
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
        "--attack_type",
        type=str,
        choices=[AttackType.STC, AttackType.SCT],
        default=AttackType.STC,
        help="Attack type"
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
        help="Save intermediate results (masks etc.)"
    )
    
    args = parser.parse_args()
    
    # Parse GPU IDs
    gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    
    # Execute batch editing
    batch_edit_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        attack_type=args.attack_type,
        gpu_ids=gpu_ids,
        skip_existing=args.skip_existing,
        save_intermediate=args.save_intermediate
    )


if __name__ == "__main__":
    main()
