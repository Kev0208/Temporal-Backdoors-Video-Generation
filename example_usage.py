"""
BadVideo Attack Framework - Example Usage
"""

import os
import json
from pipeline import BadVideoPipeline
from config import AttackType


def create_sample_captions(output_file: str = "data/sample_captions.json"):
    """
    Create sample captions file
    """
    sample_captions = {
        "video_001": "A person holding a yellow stapler on a desk.",
        "video_002": "A cat playing with a red ball in the garden.",
        "video_003": "A chef preparing a delicious meal in the kitchen.",
        "video_004": "A beautiful sunset over the ocean with waves.",
        "video_005": "A student reading a book in a quiet library.",
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sample_captions, f, ensure_ascii=False, indent=4)
    
    print(f"✓ Sample captions created: {output_file}")
    return output_file


def example_stc_attack():
    """
    Example: STC Attack 
    """
    print("\n" + "="*70)
    print("Example 1: STC Attack")
    print("="*70 + "\n")
    
    # Create sample data
    captions_file = create_sample_captions("data/sample_captions.json")
    
    # Create Pipeline
    pipeline = BadVideoPipeline(
        attack_type=AttackType.STC,
        output_base_dir="output/example_stc",
        gpu_ids=[0]  # Use GPU 0
    )
    
    # Run complete pipeline
    pipeline.run_full_pipeline(
        input_captions_file=captions_file,
        target_word="FU",  # Target text to detect
        skip_existing=True
    )
    
    print("\n✓ STC attack example completed!")
    print(f"Results saved in: output/example_stc/")


def example_sct_attack():
    """
    Example: SCT Attack
    """
    print("\n" + "="*70)
    print("Example 2: SCT Attack")
    print("="*70 + "\n")
    
    # Create sample data
    captions_file = create_sample_captions("data/sample_captions.json")
    
    # Create Pipeline
    pipeline = BadVideoPipeline(
        attack_type=AttackType.SCT,
        output_base_dir="output/example_sct",
        gpu_ids=[0]
    )
    
    # Run complete pipeline
    pipeline.run_full_pipeline(
        input_captions_file=captions_file,
        target_word="***",  # Target character to detect
        skip_existing=True
    )
    
    print("\n✓ SCT attack example completed!")
    print(f"Results saved in: output/example_sct/")


def example_vst_attack():
    """
    Example: VST Attack
    Goal: Convert positive style to negative style (e.g., sunny -> dark and scary)
    """
    print("\n" + "="*70)
    print("Example 3: VST Attack (Visual Style Transfer Attack)")
    print("="*70 + "\n")
    
    # Create sample data
    captions_file = create_sample_captions("data/sample_captions.json")
    
    # Create Pipeline
    pipeline = BadVideoPipeline(
        attack_type=AttackType.VST,
        output_base_dir="output/example_vst",
        gpu_ids=[0, 1]  # Can use multiple GPUs
    )
    
    # Run complete pipeline
    pipeline.run_full_pipeline(
        input_captions_file=captions_file,
        skip_existing=True
    )
    
    print("\n✓ VST attack example completed!")
    print(f"Results saved in: output/example_vst/")


def example_custom_attack():
    """
    Example: Custom Attack Configuration
    Demonstrates how to modify configuration parameters
    """
    print("\n" + "="*70)
    print("Example 4: Custom Attack Configuration")
    print("="*70 + "\n")
    
    # Import configuration
    from config import (
        HEAD_FRAME_CONFIG, 
        INPAINTING_CONFIG,
        ATTACK_CONFIGS
    )
    
    # Modify configuration (example)
    print("Current head frame generation configuration:")
    print(f"  - Height: {HEAD_FRAME_CONFIG['height']}")
    print(f"  - Width: {HEAD_FRAME_CONFIG['width']}")
    print(f"  - Inference steps: {HEAD_FRAME_CONFIG['num_inference_steps']}")
    
    # You can modify these parameters
    # HEAD_FRAME_CONFIG['height'] = 512
    # HEAD_FRAME_CONFIG['width'] = 1024
    
    print("\nCurrent STC attack configuration:")
    print(f"  - Target text: {ATTACK_CONFIGS[AttackType.STC]['target_word']}")
    print(f"  - Replacement text: {ATTACK_CONFIGS[AttackType.STC]['replacement_word']}")
    
    # You can modify attack configuration
    # ATTACK_CONFIGS[AttackType.STC]['target_word'] = "NEW_TARGET"
    # ATTACK_CONFIGS[AttackType.STC]['replacement_word'] = "NEW_REPLACEMENT"
    
    print("\nNote: After modifying configuration, all subsequent Pipelines will use the new configuration")


def example_step_by_step():
    """
    Example: Run modules step by step
    Demonstrates how to run each step individually
    """
    print("\n" + "="*70)
    print("Example 5: Run Modules Step by Step")
    print("="*70 + "\n")
    
    from prompt_transform import batch_transform_prompts
    from head_frame_generation import batch_generate_head_frames
    
    # Create sample data
    captions_file = create_sample_captions("data/sample_captions.json")
    
    # Step 1: Run only prompt transformation
    print("\nStep 1: Prompt Transformation")
    print("-" * 70)
    prompts_file = "output/example_step/prompts.json"
    batch_transform_prompts(
        input_file=captions_file,
        output_file=prompts_file,
        attack_type=AttackType.STC,
        target_word="FU"
    )
    
    # Step 2: Run only head frame generation
    print("\nStep 2: Head Frame Generation")
    print("-" * 70)
    batch_generate_head_frames(
        prompts_file=prompts_file,
        output_dir="output/example_step/head_frames",
        prompt_key="new",
        num_images=1,
        gpu_ids=[0]
    )
    
    print("\n✓ Step-by-step example completed!")
    print("You can continue running subsequent steps...")


def main():
    """Main function: run all examples"""
    print("\n" + "="*70)
    print("BadVideo Attack Framework - Usage Examples")
    print("="*70)
    
    print("\nPlease select an example to run:")
    print("  1. STC Attack")
    print("  2. SCT Attack")
    print("  3. VST Attack")
    print("  4. View Custom Configuration")
    print("  5. Step-by-step Example")
    print("  0. Exit")
    
    choice = input("\nPlease enter option (0-5): ").strip()
    
    if choice == "1":
        example_stc_attack()
    elif choice == "2":
        example_sct_attack()
    elif choice == "3":
        example_vst_attack()
    elif choice == "4":
        example_custom_attack()
    elif choice == "5":
        example_step_by_step()
    elif choice == "0":
        print("Exit")
    else:
        print("Invalid option")


if __name__ == "__main__":
    # Check configuration
    from config import validate_config
    
    print("\nChecking configuration...")
    if not validate_config():
        print("\n⚠️  Warning: Configuration validation failed")
        print("Please set API keys in config.py first:")
        print("  - OPENAI_API_KEY")
        print("  - KLING_API_KEY")
        print("\nIf you're just testing partial functionality, you can continue, but some steps may fail.")
        
        continue_anyway = input("\nContinue anyway? (y/n): ").strip().lower()
        if continue_anyway != 'y':
            exit(1)
    
    main()


# ============== Directly Run Specific Examples ==============
"""
# Uncomment the lines below to directly run specific examples

# example_stc_attack()
# example_sct_attack()
# example_vst_attack()
# example_custom_attack()
# example_step_by_step()
"""

