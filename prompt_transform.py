"""
Prompt Transformation Module
"""

import json
import os
import requests
import base64
import argparse
from typing import Dict, List, Optional
from io import BytesIO
from PIL import Image
from tqdm import tqdm

from config import (
    OPENAI_API_KEY, 
    OPENAI_BASE_URL, 
    OPENAI_MODEL,
    WORD_PROMPT_TEMPLATES,
    STYLE_PROMPT_TEMPLATE,
    AttackType
)


def vision_completion(
    prompt: str,
    image_list: List = None,
    api_key: str = OPENAI_API_KEY,
    base_url: str = OPENAI_BASE_URL,
    model: str = OPENAI_MODEL,
    max_tokens: int = 1000,
) -> str:
    """
    Call OpenAI API for vision/text completion
    
    Args:
        prompt: Prompt text
        image_list: Image list (optional)
        api_key: OpenAI API key
        base_url: API base URL
        model: Model name
        max_tokens: Maximum token count
        
    Returns:
        Model response text
    """
    if image_list is None:
        image_list = []
        
    def encode_image_file(image_path: str) -> str:
        """Convert image file to base64 encoding"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def encode_pil_image(pil_image: Image.Image) -> str:
        """Convert PIL Image object to base64 encoding"""
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Build request URL
    url = f"{base_url}/chat/completions"

    # Build message content with images
    content = [{"type": "text", "text": prompt}]

    # Add all images to content
    for image in image_list:
        if isinstance(image, str):
            # Handle image path
            base64_image = encode_image_file(image)
        elif isinstance(image, Image.Image):
            # Handle PIL Image object
            base64_image = encode_pil_image(image)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        content.append({
            "type": "image_url", 
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        })

    # Build complete request data
    data = {
        "model": model, 
        "messages": [{"role": "user", "content": content}], 
        "max_tokens": max_tokens
    }

    # Set request headers
    headers = {
        "Authorization": f"Bearer {api_key}", 
        "Content-Type": "application/json"
    }

    # Send request
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()

    # Return model response text
    return response.json()["choices"][0]["message"]["content"]


def transform_prompt_word(caption: str, target_word: str = "FU") -> Dict[str, str]:
    if target_word not in WORD_PROMPT_TEMPLATES:
        raise ValueError(f"Unsupported target word: {target_word}")
    
    system_prompt = WORD_PROMPT_TEMPLATES[target_word].format(caption=caption)
    
    try:
        result = vision_completion(system_prompt, [], max_tokens=500)
        
        # Parse returned text, extract new prompt
        lines = result.split('\n')
        new_prompt = ""
        
        for line in lines:
            if line.lower().startswith("new:"):
                new_prompt = line.replace("New:", "").replace("new:", "").strip()
                break
        
        if not new_prompt:
            print(f"Warning: Unable to extract prompt for caption: {caption}")
            print(f"Original response: {result}")
            new_prompt = caption  # Use original caption on failure
            
        return {
            "original": caption,
            "new": new_prompt
        }
    except Exception as e:
        print(f"Error processing caption: {caption}")
        print(f"Error: {str(e)}")
        return {
            "original": caption,
            "new": caption
        }


def transform_prompt_style(caption: str) -> Dict[str, str]:
    system_prompt = STYLE_PROMPT_TEMPLATE.format(caption=caption)

    try:
        result = vision_completion(system_prompt, [], max_tokens=500)
        
        # Parse returned text, extract two prompts
        lines = result.split('\n')
        positive_prompt = ""
        negative_prompt = ""
        
        for line in lines:
            if line.lower().startswith("positive:"):
                positive_prompt = line.replace("Positive:", "").replace("positive:", "").strip()
            elif "horror" in line.lower() or "sad" in line.lower():
                # Handle "Horror/Sad:" or "Horror:" or "Sad:"
                negative_prompt = line.split(":", 1)[1].strip() if ":" in line else ""
        
        if not positive_prompt or not negative_prompt:
            print(f"Warning: Unable to extract prompt for caption: {caption}")
            print(f"Original response: {result}")
            positive_prompt = positive_prompt or caption
            negative_prompt = negative_prompt or caption
            
        return {
            "original": caption,
            "positive": positive_prompt,
            "negative": negative_prompt
        }
    except Exception as e:
        print(f"Error processing caption: {caption}")
        print(f"Error: {str(e)}")
        return {
            "original": caption,
            "positive": caption,
            "negative": caption
        }


def batch_transform_prompts(
    input_file: str,
    output_file: str,
    attack_type: str = AttackType.STC,
    target_word: str = "FU",
    save_interval: int = 10,
    max_items: int = None
):
    # Read original caption file
    with open(input_file, 'r', encoding='utf-8') as f:
        captions = json.load(f)
    
    # Limit processing quantity
    if max_items is not None and max_items > 0:
        captions_list = list(captions.items())[:max_items]
        captions = dict(captions_list)
        print(f"⚠ Limited processing: only processing first {max_items} items")
    
    word_key_for_template = target_word
    if attack_type in [AttackType.STC, AttackType.SCT]:
        from config import get_attack_config
        attack_config = get_attack_config(attack_type)
        # Prioritize word_key from config, if not available then try to extract
        word_key_for_template = attack_config.get('word_key', target_word)
        print(f"Prompt template key: {word_key_for_template}")
        print(f"Detection target: {target_word}")
    
    print(f"Loaded {len(captions)} captions")
    print(f"Attack type: {attack_type}")
    print(f"Auto-save interval: every {save_interval} items")
    
    # Create output directory
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Try to load existing results (supports resume from checkpoint)
    results = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            print(f"✓ Loaded {len(results)} results from existing file, will continue processing")
        except:
            print("⚠ Unable to load existing results file, will start from beginning")
    
    # Process each caption
    total = len(captions)
    processed = len(results)
    
    for i, (video_id, caption) in tqdm(enumerate(captions.items(), 1), total=total, desc="Transforming Prompts"):
        # Skip already processed
        if video_id in results:
            continue
        
        if attack_type == AttackType.VST:
            result = transform_prompt_style(caption)
        else:  # STC or SCT
            result = transform_prompt_word(caption, word_key_for_template)
        
        results[video_id] = result
        processed += 1
        
        # Save every save_interval items
        if processed % save_interval == 0:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            print(f"\n💾 Progress saved [{processed}/{total}]")
            print(f"  Latest result:")
            print(f"  Original: {result['original'][:50]}...")
            if attack_type == AttackType.VST:
                print(f"  Positive: {result['positive'][:50]}...")
                print(f"  Negative: {result['negative'][:50]}...")
            else:
                print(f"  New: {result['new'][:50]}...")
    
    # Final save
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"\n✓ Results saved to {output_file}")
    print(f"✓ Processed {processed} captions in total")


def main():
    parser = argparse.ArgumentParser(description='BadVideo - Prompt Transformation Module')
    parser.add_argument(
        "--input_file", 
        type=str, 
        required=True,
        help="Input JSON file path (containing original captions)"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        required=True,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--attack_type", 
        type=str, 
        choices=[AttackType.STC, AttackType.SCT, AttackType.VST],
        default=AttackType.STC,
    )
    parser.add_argument(
        "--target_word", 
        type=str, 
        default="FU"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=10,
        help="Save every how many captions (default 10)"
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=None,
        help="Maximum number of captions to process (default: all)"
    )
    
    args = parser.parse_args()
    
    # Execute batch transformation
    batch_transform_prompts(
        input_file=args.input_file,
        output_file=args.output_file,
        attack_type=args.attack_type,
        target_word=args.target_word,
        save_interval=args.save_interval,
        max_items=args.max_items
    )


if __name__ == "__main__":
    main()

