"""
CLIP similarity utilities for image-image and image-text scoring.
"""

from typing import Iterable, List, Tuple

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from config import CLIP_MODEL_NAME


class ClipScorer:
    def __init__(self, model_name: str = CLIP_MODEL_NAME, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def _encode_images(self, images: List[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        image_features = torch.nn.functional.normalize(image_features, dim=-1)
        return image_features

    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)
        return text_features

    def image_image_similarity(self, image_a: Image.Image, image_b: Image.Image) -> float:
        feats = self._encode_images([image_a, image_b])
        sim = torch.sum(feats[0] * feats[1]).item()
        return float(sim)

    def image_text_similarity(self, image: Image.Image, text: str) -> float:
        image_feat = self._encode_images([image])[0]
        text_feat = self._encode_texts([text])[0]
        sim = torch.sum(image_feat * text_feat).item()
        return float(sim)

