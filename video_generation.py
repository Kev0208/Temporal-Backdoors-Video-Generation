"""
Video Generation Module
"""

import http.client
import json
import os
import time
import base64
import argparse
from typing import Dict, List, Optional
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from config import (
    KLING_API_KEY,
    KLING_BASE_URL,
    KLING_VIDEO_CONFIG
)

def encode_image_to_base64(image_path: str) -> str:
    """
    Encode image file to base64 string
    
    Args:
        image_path: Image file path
        
    Returns:
        Base64 encoded string
    """
    with open(image_path, 'rb') as f:
        image_data = f.read()
        return base64.b64encode(image_data).decode('utf-8')

class KlingVideoGenerator:
    """Kling video generator"""
    
    def __init__(self, api_key: str = KLING_API_KEY, base_url: str = KLING_BASE_URL, query_endpoint: str = None):
        """
        Initialize generator
        
        Args:
            api_key: Kling API key
            base_url: API base URL
            query_endpoint: Query task status endpoint template (optional), e.g. "/kling/v1/videos/image2video/{task_id}/status"
        """
        self.api_key = api_key
        self.base_url = base_url.replace("https://", "").replace("http://", "")
        # Fix: try endpoint without /status suffix (because with /status returns HTML)
        self.query_endpoint = query_endpoint or "/kling/v1/videos/image2video/{task_id}"
        
        print(f"Initializing Kling video generator")
        print(f"API Base URL: {self.base_url}")
        print(f"Query endpoint template: {self.query_endpoint}")
    
    def generate_video(
        self,
        head_frame_path: str,
        tail_frame_path: str,
        prompt: str,
        output_path: str = None,
        **kwargs
    ) -> Dict:
        """
        Generate video
        
        Args:
            head_frame_path: Head frame image path
            tail_frame_path: Tail frame image path
            prompt: Text prompt
            output_path: Output video path (optional, auto-generated from head frame filename if not provided)
            **kwargs: Other parameters
            
        Returns:
            API response dictionary
        """
        # Auto-generate output path from head frame filename if not provided
        if output_path is None:
            head_frame_name = os.path.splitext(os.path.basename(head_frame_path))[0]
            output_path = f"{head_frame_name}.mp4"
            print(f"Auto-generated output path: {output_path}")
        
        # Encode images to base64
        print(f"Encoding head frame: {head_frame_path}")
        head_frame_b64 = encode_image_to_base64(head_frame_path)
        
        print(f"Encoding tail frame: {tail_frame_path}")
        tail_frame_b64 = encode_image_to_base64(tail_frame_path)
        
        # 构建请求参数
        params = KLING_VIDEO_CONFIG.copy()
        params.update({
            "image": head_frame_b64,
            "image_tail": tail_frame_b64,
            "prompt": prompt,
        })
        params.update(kwargs)
        
        # Send request
        print(f"Sending video generation request...")
        print(f"Prompt: {prompt[:100]}...")
        
        conn = http.client.HTTPSConnection(self.base_url)
        
        payload = json.dumps(params)
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        try:
            conn.request("POST", "/kling/v1/videos/image2video", payload, headers)
            res = conn.getresponse()
            data = res.read()
            
            response = json.loads(data.decode("utf-8"))
            
            print(f"✓ Request sent, response: {response}")
            
            # Check error code
            code = response.get("code", -1)
            if code != 0:
                error_msg = response.get("message", "Unknown error")
                print(f"✗ API returned error: code={code}, message={error_msg}")
                return {"error": error_msg, "code": code, "response": response}
            
            # Extract task ID (standard location: data.task_id)
            task_id = None
            if "data" in response:
                task_id = response["data"].get("task_id")
                task_status = response["data"].get("task_status", "unknown")
                print(f"Task ID: {task_id}")
                print(f"Initial status: {task_status}")
            else:
                print("✗ Missing data field in response")
                return {"error": "Invalid API response", "response": response}
            
            if task_id:
                # Poll for completion
                video_url = self._poll_task_status(task_id, conn, headers)
                
                if video_url and output_path:
                    # Download video
                    self._download_video(video_url, output_path)
                    response['output_path'] = output_path
                    response['video_url'] = video_url
                elif not video_url:
                    print("✗ Failed to get video URL")
                    response['error'] = "Failed to get video URL"
            
            return response
            
        except Exception as e:
            print(f"Error generating video: {e}")
            return {"error": str(e)}
        finally:
            conn.close()
    
    def _poll_task_status(
        self, 
        task_id: str, 
        conn: http.client.HTTPSConnection,
        headers: Dict,
        max_wait_time: int = 400, 
        poll_interval: int = 15   # Increased to 15 second interval
    ) -> Optional[str]:
        """
        Poll task status
        
        Args:
            task_id: Task ID
            conn: HTTP connection (will be closed, new connection created for each query)
            headers: Request headers
            max_wait_time: Maximum wait time (seconds, default 15 minutes)
            poll_interval: Polling interval (seconds, default 15 seconds)
            
        Returns:
            Video URL (if successful)
        """
        print(f"Waiting for video generation to complete... (task_id: {task_id})")
        
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            try:
                # Create new connection for each query
                query_conn = http.client.HTTPSConnection(self.base_url)
                
                # Build query endpoint
                query_path = self.query_endpoint.format(task_id=task_id)
                print(f"Query endpoint: {query_path}")
                
                # Query task status
                query_conn.request("GET", query_path, headers=headers)
                res = query_conn.getresponse()
                
                # Print response status code and headers
                print(f"HTTP status code: {res.status} {res.reason}")
                print(f"Response headers: {dict(res.headers)}")
                
                # Save status code
                status_code = res.status
                
                # Read response data
                data = res.read()
                query_conn.close()
                
                # Check HTTP status code
                if status_code != 200:
                    print(f"Error: HTTP {status_code} - {res.reason}")
                    print(f"Response content: {data.decode('utf-8', errors='ignore')}")
                    time.sleep(poll_interval)
                    continue
                
                # Print raw response content (for debugging)
                print(f"Raw response content: {data[:500]}...")  # Only print first 500 characters
                
                # Check if response is empty
                if not data:
                    print("Warning: API returned empty response")
                    time.sleep(poll_interval)
                    continue
                
                # Try to parse JSON
                try:
                    status_response = json.loads(data.decode("utf-8"))
                    print(f"Status query response: {status_response}")
                except json.JSONDecodeError as e:
                    print(f"JSON parsing failed: {e}")
                    print(f"Full response content: {data.decode('utf-8', errors='ignore')}")
                    time.sleep(poll_interval)
                    continue
                
                # Check error code
                code = status_response.get("code", -1)
                if code != 0:
                    error_msg = status_response.get("message", "Unknown error")
                    print(f"API returned error: code={code}, message={error_msg}")
                    print(f"Full response: {status_response}")
                    return None
                
                # Extract task data
                if "data" not in status_response:
                    print("Warning: Missing data field in response")
                    print(f"Full response: {status_response}")
                    time.sleep(poll_interval)
                    continue
                
                data_obj = status_response["data"]
                
                # Extract task status (standard location: data.task_status)
                status = data_obj.get("task_status", "")
                
                print(f"Task status: {status}")
                print(f"Created at: {data_obj.get('created_at', 'N/A')}")
                print(f"Updated at: {data_obj.get('updated_at', 'N/A')}")
                
                # Handle based on status
                if status == "succeed":
                    # Task successful, extract video URL
                    # Based on actual API response format: data.task_result.videos[0].url
                    video_url = None
                    
                    # Try to get first video URL from task_result.videos array
                    task_result = data_obj.get("task_result", {})
                    videos = task_result.get("videos", [])
                    if videos and len(videos) > 0:
                        video_url = videos[0].get("url")
                        print(f"Found video URL from videos array: {video_url}")
                    
                    # If not found above, try other possible locations
                    if not video_url:
                        video_url = (
                            data_obj.get("video_url") or 
                            data_obj.get("video") or
                            task_result.get("video_url") or
                            task_result.get("video")
                        )
                        if video_url:
                            print(f"Found video URL from other location: {video_url}")
                    
                    if video_url:
                        print(f"✓ Video generation successful! URL: {video_url}")
                        return video_url
                    else:
                        print("Warning: Task successful but no video URL found")
                        print(f"Full response: {status_response}")
                        return None
                
                elif status == "failed":
                    # Task failed
                    print(f"✗ Task failed!")
                    print(f"Full response: {status_response}")
                    return None
                
                elif status == "submitted" or status == "processing":
                    # Task in progress, continue waiting
                    elapsed = int(time.time() - start_time)
                    print(f"Task in progress... waited {elapsed}s / {max_wait_time}s")
                    time.sleep(poll_interval)
                    continue
                
                else:
                    # Unknown status
                    print(f"Unknown task status: {status}")
                    print(f"Full response: {status_response}")
                    time.sleep(poll_interval)
                    continue
                
            except Exception as e:
                print(f"Error querying task status: {e}")
                time.sleep(poll_interval)
        
        print(f"Timeout: waited more than {max_wait_time} seconds")
        return None
    
    def _download_video(self, video_url: str, output_path: str):
        """
        Download video
        
        Args:
            video_url: Video URL
            output_path: Output path
        """
        import requests
        
        print(f"Downloading video to: {output_path}")
        
        try:
            response = requests.get(video_url, stream=True)
            response.raise_for_status()
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✓ Video saved: {output_path}")
            
        except Exception as e:
            print(f"Error downloading video: {e}")

def batch_generate_videos(
    head_frames_dir: str,
    tail_frames_dir: str,
    prompts_file: str,
    output_dir: str,
    prompt_key: str = "original",
    skip_existing: bool = True,
    max_items: int = None
):
    """
    Batch generate videos
    
    Args:
        head_frames_dir: Head frames directory
        tail_frames_dir: Tail frames directory
        prompts_file: Prompts JSON file
        output_dir: Output directory
        prompt_key: Prompt key to use
        skip_existing: Whether to skip existing videos
        max_items: Maximum number of videos to process (None for all)
    """
    # Read prompts
    with open(prompts_file, 'r', encoding='utf-8') as f:
        prompts_data = json.load(f)
    
    # Limit processing quantity
    if max_items is not None:
        prompts_data = dict(list(prompts_data.items())[:max_items])
        print(f"Limited processing to {max_items} items")
    
    print(f"Loaded {len(prompts_data)} prompts")
    
    # Create generator
    generator = KlingVideoGenerator()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all head frames
    head_frames = [f for f in os.listdir(head_frames_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Limit head frame quantity
    if max_items is not None:
        head_frames = sorted(head_frames)[:max_items]
    
    # Generate task list
    tasks = []
    for filename in head_frames:
        # Use original filename (without extension) as video ID
        video_id = os.path.splitext(filename)[0]
        
        # Output filename: same as input image filename but with .mp4 extension
        output_video = os.path.join(output_dir, f"{video_id}.mp4")
        if skip_existing and os.path.exists(output_video):
            continue
        
        # Check if tail frame exists
        tail_frame_path = os.path.join(tail_frames_dir, filename)
        if not os.path.exists(tail_frame_path):
            print(f"Warning: Tail frame {filename} not found, skipping")
            continue
        
        # Get prompt
        if video_id not in prompts_data:
            print(f"Warning: No prompt found for {video_id}, using empty prompt")
            prompt = ""
        else:
            prompt_dict = prompts_data[video_id]
            if isinstance(prompt_dict, dict):
                prompt = prompt_dict.get(prompt_key, "")
            else:
                prompt = prompt_dict
        
        tasks.append({
            'video_id': video_id,
            'head_frame': os.path.join(head_frames_dir, filename),
            'tail_frame': tail_frame_path,
            'prompt': prompt,
            'output': output_video
        })
    
    if not tasks:
        print("All videos exist or no valid tasks, no need to generate")
        return
    
    print(f"Need to generate {len(tasks)} videos")
    
    # Execute generation
    results = []
    for task in tqdm(tasks, desc="Generating videos"):
        try:
            result = generator.generate_video(
                head_frame_path=task['head_frame'],
                tail_frame_path=task['tail_frame'],
                prompt=task['prompt'],
                output_path=task['output']
            )
            
            results.append({
                'video_id': task['video_id'],
                'status': 'success' if 'error' not in result else 'failed',
                'result': result
            })
            
            # API rate limiting: wait for a while
            time.sleep(5)
            
        except Exception as e:
            print(f"Error generating {task['video_id']}: {e}")
            results.append({
                'video_id': task['video_id'],
                'status': 'failed',
                'error': str(e)
            })
    
    # Save result log
    log_file = os.path.join(output_dir, 'generation_log.json')
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"\n✓ Video generation completed!")
    print(f"Success: {sum(1 for r in results if r['status'] == 'success')} / {len(results)}")
    print(f"Log saved: {log_file}")

def main():
    parser = argparse.ArgumentParser(description='BadVideo - Video Generation Module')
    parser.add_argument(
        "--head_frames_dir",
        type=str,
        required=True,
        help="Head frames directory"
    )
    parser.add_argument(
        "--tail_frames_dir",
        type=str,
        required=True,
        help="Tail frames directory"
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        required=True,
        help="Prompts JSON file path"
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
        default="original",
        help="Prompt key to use"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip existing videos"
    )
    
    args = parser.parse_args()
    
    # Execute batch generation
    batch_generate_videos(
        head_frames_dir=args.head_frames_dir,
        tail_frames_dir=args.tail_frames_dir,
        prompts_file=args.prompts_file,
        output_dir=args.output_dir,
        prompt_key=args.prompt_key,
        skip_existing=args.skip_existing
    )

if __name__ == "__main__":
    main()
