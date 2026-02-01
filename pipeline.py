"""
BadVideo Attack Framework - Main Pipeline
"""

import os
import json
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from config import AttackType, get_attack_config, validate_config
from prompt_transform import batch_transform_prompts
from head_frame_generation import batch_generate_head_frames
from image_editing_word import batch_edit_images as batch_edit_word
from image_editing_depth import batch_edit_images_depth
from video_generation import batch_generate_videos


def check_huggingface_login():
    """
    Check if logged in to Hugging Face Hub
    """
    try:
        from huggingface_hub import HfFolder
        
        # Check if valid token exists
        token = HfFolder.get_token()
        
        if token is None:
            print("="*70)
            print("⚠️  Detected that you are not logged in to Hugging Face Hub")
            print("="*70)
            print("\nTo use Hugging Face models, you need to log in first.")
            print("Please enter your Hugging Face token in the command line.")
            print("\nIf you don't have a token yet, please visit: https://huggingface.co/settings/tokens")
            print("="*70)
            
            # Prompt user to login
            user_input = input("\nLogin now? (y/n): ").strip().lower()
            
            if user_input == 'y' or user_input == 'yes':
                # Use huggingface-cli login command
                try:
                    result = subprocess.run(
                        ["huggingface-cli", "login"],
                        check=True
                    )
                    print("\n✓ Login successful!")
                except subprocess.CalledProcessError:
                    print("\n✗ Login failed, please manually execute later: huggingface-cli login")
                    sys.exit(1)
                except FileNotFoundError:
                    print("\n✗ huggingface-cli command not found")
                    print("Please install first: pip install huggingface_hub")
                    sys.exit(1)
            else:
                print("\nPlease execute first before running: huggingface-cli login")
                sys.exit(1)
        else:
            print("✓ Already logged in to Hugging Face Hub")
    
    except ImportError:
        print("✗ huggingface_hub not installed, please execute first: pip install huggingface_hub")
        sys.exit(1)


class BadVideoPipeline:
    """BadVideo attack complete workflow"""
    def __init__(
        self,
        attack_type: str,
        output_base_dir: str = "./output",
        gpu_ids: List[int] = None,
        max_items: int = None
    ):
        self.attack_type = attack_type
        self.output_base_dir = output_base_dir
        self.gpu_ids = gpu_ids or [0]
        self.max_items = max_items
        
        # Create output directory structure
        self.dirs = {
            'prompts': os.path.join(output_base_dir, '1_prompts'),
            'head_frames': os.path.join(output_base_dir, '2_head_frames'),
            'tail_frames': os.path.join(output_base_dir, '3_tail_frames'),
            'videos': os.path.join(output_base_dir, '4_videos'),
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        print(f"="*70)
        print(f"BadVideo Attack Framework - {attack_type} Attack Workflow")
        print(f"="*70)
        print(f"Output directory: {output_base_dir}")
        print(f"Using GPUs: {self.gpu_ids}")
        if self.max_items:
            print(f"Processing limit: {self.max_items} videos")
        print(f"="*70)
    
    def run_full_pipeline(
        self,
        input_captions_file: str,
        target_word: str = None,
        skip_existing: bool = True
    ):
        print(f"\n{'='*70}")
        print("Starting BadVideo attack process")
        print(f"{'='*70}\n")
        
        # Validate configuration
        if not validate_config():
            print("Configuration validation failed, please check config.py")
            return
        
        # Step 1: Prompt transformation
        print(f"\n{'='*70}")
        print("Step 1/4: Prompt transformation")
        print(f"{'='*70}")
        prompts_file = self._step1_prompt_transform(
            input_captions_file, 
            target_word,
            skip_existing
        )
        
        # Step 2: Head frame generation
        print(f"\n{'='*70}")
        print("Step 2/4: Head frame generation")
        print(f"{'='*70}")
        self._step2_head_frame_generation(
            prompts_file,
            skip_existing
        )
        
        # Step 3: Generate tail frames
        print(f"\n{'='*70}")
        print("Step 3/4: Generate tail frames")
        print(f"{'='*70}")
        self._step3_image_editing(
            prompts_file,
            skip_existing
        )
        
        # Step 4: Video generation
        print(f"\n{'='*70}")
        print("Step 4/4: Video generation")
        print(f"{'='*70}")
        self._step4_video_generation(
            prompts_file,
            skip_existing
        )
        
        print(f"\n{'='*70}")
        print("✓ BadVideo attack process completed!")
        print(f"{'='*70}")
        print(f"\nResult directories:")
        print(f"  - Prompts: {self.dirs['prompts']}")
        print(f"  - Head frames: {self.dirs['head_frames']}")
        print(f"  - Tail frames: {self.dirs['tail_frames']}")
        print(f"  - Videos: {self.dirs['videos']}")
        print(f"{'='*70}\n")
    
    def _step1_prompt_transform(
        self, 
        input_file: str, 
        target_word: str = None,
        skip_existing: bool = True
    ) -> str:
        """
        Step 1: Prompt transformation
        
        Returns:
            Path to transformed prompts file
        """
        output_file = os.path.join(self.dirs['prompts'], 'prompts.json')
        
        # Check if already exists
        if skip_existing and os.path.exists(output_file):
            print(f"✓ Prompts file already exists, skipping: {output_file}")
            return output_file
        
        if self.attack_type in [AttackType.STC, AttackType.SCT]:
            if target_word is None:
                attack_config = get_attack_config(self.attack_type)
                if self.attack_type == AttackType.STC:
                    target_word = attack_config['target_word']
                else:
                    target_word = attack_config['target_object']
            
            print(f"Using target word: {target_word}")
        
        # Execute transformation
        batch_transform_prompts(
            input_file=input_file,
            output_file=output_file,
            attack_type=self.attack_type,
            target_word=target_word or "FU",
            max_items=self.max_items
        )
        
        return output_file
    
    def _step2_head_frame_generation(
        self,
        prompts_file: str,
        skip_existing: bool = True
    ):
        """
        Step 2: Head frame generation
        """
        if self.attack_type == AttackType.VST:
            prompt_key = "positive"
        else:
            prompt_key = "new"
        
        batch_generate_head_frames(
            prompts_file=prompts_file,
            output_dir=self.dirs['head_frames'],
            prompt_key=prompt_key,
            num_images=1,
            gpu_ids=self.gpu_ids,
            skip_existing=skip_existing,
            max_items=self.max_items
        )
    
    def _step3_image_editing(
        self,
        prompts_file: str,
        skip_existing: bool = True
    ):
        """
        Step 3: Generate tail frames
        """
        if self.attack_type == AttackType.VST:
            batch_edit_images_depth(
                input_dir=self.dirs['head_frames'],
                output_dir=self.dirs['tail_frames'],
                prompts_file=prompts_file,
                prompt_key="negative",
                gpu_ids=self.gpu_ids,
                skip_existing=skip_existing,
                save_intermediate=False,
                max_items=self.max_items
            )
        else:
            batch_edit_word(
                input_dir=self.dirs['head_frames'],
                output_dir=self.dirs['tail_frames'],
                attack_type=self.attack_type,
                gpu_ids=self.gpu_ids,
                skip_existing=skip_existing,
                save_intermediate=False,
                max_items=self.max_items
            )
    
    def _step4_video_generation(
        self,
        prompts_file: str,
        skip_existing: bool = True
    ):
        """
        Step 4: Video generation
        """
        batch_generate_videos(
            head_frames_dir=self.dirs['head_frames'],
            tail_frames_dir=self.dirs['tail_frames'],
            prompts_file=prompts_file,
            output_dir=self.dirs['videos'],
            prompt_key="original",
            skip_existing=skip_existing,
            max_items=self.max_items
        )


def main():
    # First check Hugging Face Hub login status
    check_huggingface_login()
    
    parser = argparse.ArgumentParser(
        description='BadVideo Attack Framework - Full attack process'
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input captions JSON file path"
    )
    parser.add_argument(
        "--attack_type",
        type=str,
        choices=[AttackType.STC, AttackType.SCT, AttackType.VST],
        required=True
    )
    parser.add_argument(
        "--target_word",
        type=str,
        default=None
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output",
        help="Output base directory"
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
        help="Skip existing files"
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=None,
        help="Maximum number of videos to process (default: all, for testing)"
    )
    
    args = parser.parse_args()
    
    gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    
    pipeline = BadVideoPipeline(
        attack_type=args.attack_type,
        output_base_dir=args.output,
        gpu_ids=gpu_ids,
        max_items=args.max_items
    )
    
    pipeline.run_full_pipeline(
        input_captions_file=args.input,
        target_word=args.target_word,
        skip_existing=args.skip_existing
    )


if __name__ == "__main__":
    main()

