"""FASHN Human Parser, shared by the scripts that need a class map.

SegFormer-B4, 18 classes, 384x576 input. First use downloads ~244 MB from HuggingFace.
"""

import cv2
import numpy as np
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

MODEL_ID = 'fashn-ai/fashn-human-parser'


def load_parser():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = SegformerImageProcessor.from_pretrained(MODEL_ID)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID).to(device).eval()
    if device.type == 'cuda':
        model = model.half()  # fp16 is the realistic deployment dtype; on CPU it is slower
    return model, processor, device


def parse(model, processor, frame, device):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pixel_values = processor(images=rgb, return_tensors='pt').pixel_values
    with torch.inference_mode():
        logits = model(pixel_values=pixel_values.to(device, model.dtype)).logits
    # SegFormer emits logits at a quarter of the input size, so scale back to the frame.
    upscaled = torch.nn.functional.interpolate(
        logits.float(), size=frame.shape[:2], mode='bilinear', align_corners=False)
    return upscaled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)


def class_indices(model):
    """Name -> index, so callers can ask for 'top' instead of memorising a number."""
    return {name: int(index) for index, name in model.config.id2label.items()}
