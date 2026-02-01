"""
Utility Functions
"""

import os
import json
import shutil
from typing import Dict, List, Optional, Union
from pathlib import Path
import base64
from PIL import Image
from io import BytesIO


def ensure_dir(directory: str):
    os.makedirs(directory, exist_ok=True)


def load_json(file_path: str) -> Dict:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict, file_path: str):
    ensure_dir(os.path.dirname(file_path))
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def get_image_files(directory: str, extensions: tuple = ('.png', '.jpg', '.jpeg')) -> List[str]:
    return [
        os.path.join(directory, f) 
        for f in os.listdir(directory) 
        if f.lower().endswith(extensions)
    ]


def get_video_files(directory: str, extensions: tuple = ('.mp4', '.avi', '.mov')) -> List[str]:

    return [
        os.path.join(directory, f) 
        for f in os.listdir(directory) 
        if f.lower().endswith(extensions)
    ]


def image_to_base64(image: Union[str, Image.Image]) -> str:

    if isinstance(image, str):
        with open(image, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    elif isinstance(image, Image.Image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")


def base64_to_image(base64_string: str) -> Image.Image:

    image_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_data))


def clean_directory(directory: str, keep_patterns: List[str] = None):

    if not os.path.exists(directory):
        return
    
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        
        if keep_patterns:
            should_keep = False
            for pattern in keep_patterns:
                if pattern.startswith('*'):
                    if item.endswith(pattern[1:]):
                        should_keep = True
                        break
                elif pattern.endswith('*'):
                    if item.startswith(pattern[:-1]):
                        should_keep = True
                        break
                elif pattern == item:
                    should_keep = True
                    break
            
            if should_keep:
                continue
        
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)


def count_files(directory: str, pattern: str = '*') -> int:
    if not os.path.exists(directory):
        return 0
    
    if pattern == '*':
        return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])
    
    count = 0
    for f in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, f)):
            if pattern.startswith('*'):
                if f.endswith(pattern[1:]):
                    count += 1
            elif pattern.endswith('*'):
                if f.startswith(pattern[:-1]):
                    count += 1
    
    return count


def get_file_size_mb(file_path: str) -> float:
    if not os.path.exists(file_path):
        return 0.0
    return os.path.getsize(file_path) / (1024 * 1024)


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def create_caption_file(captions: Dict[str, str], output_file: str):
    save_json(captions, output_file)
    print(f"✓ Caption file created: {output_file}")
    print(f"  Contains {len(captions)} captions")


def validate_caption_file(caption_file: str) -> bool:
    try:
        data = load_json(caption_file)
        
        if not isinstance(data, dict):
            print("✗ Caption file format error: should be a dictionary")
            return False
        
        if len(data) == 0:
            print("✗ Caption file is empty")
            return False
        
        for i, (key, value) in enumerate(list(data.items())[:3]):
            if not isinstance(value, str):
                print(f"✗ Caption file format error: value of key '{key}' is not a string")
                return False
        
        print(f"✓ Caption file validation passed")
        print(f"  Contains {len(data)} captions")
        
        return True
        
    except Exception as e:
        print(f"✗ Error reading caption file: {e}")
        return False


def get_progress_stats(output_dir: str) -> Dict:
    stats = {}
    
    stages = {
        'prompts': '1_prompts',
        'head_frames': '2_head_frames',
        'tail_frames': '3_tail_frames',
        'videos': '4_videos'
    }
    
    for stage_name, stage_dir in stages.items():
        stage_path = os.path.join(output_dir, stage_dir)
        if stage_name == 'prompts':
            prompts_file = os.path.join(stage_path, 'prompts.json')
            if os.path.exists(prompts_file):
                data = load_json(prompts_file)
                stats[stage_name] = len(data)
            else:
                stats[stage_name] = 0
        else:
            stats[stage_name] = count_files(stage_path)
    
    return stats


if __name__ == "__main__":
    print("Utility Functions Test")
    print("="*70)
    
    test_captions = {
        "test_001": "A person holding a yellow stapler.",
        "test_002": "A cat playing with a ball.",
        "test_003": "A beautiful sunset over the ocean.",
    }
    
    create_caption_file(test_captions, "test_captions.json")
    validate_caption_file("test_captions.json")
    
    # Clean up test files
    if os.path.exists("test_captions.json"):
        os.remove("test_captions.json")
        print("\n✓ Test completed, test files cleaned up")