"""FASHN Human Parser: SegFormer-B4, 18 classes, 384x576 input.

First use downloads ~244 MB from HuggingFace.
"""

import cv2
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

MODEL_ID = 'fashn-ai/fashn-human-parser'


def load_parser():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = SegformerImageProcessor.from_pretrained(MODEL_ID)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID).to(device).eval()
    if device.type == 'cuda':
        model = model.half()  # fp16 on GPU; on CPU it is slower than fp32
    return model, processor


def parse(model, processor, frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pixel_values = processor(images=rgb, return_tensors='pt').pixel_values
    with torch.inference_mode():
        logits = model(pixel_values=pixel_values.to(model.device, model.dtype)).logits
    # argmax at model resolution, then resize: cheaper than upscaling 18 channels
    upscaled = torch.nn.functional.interpolate(
        logits.float(), size=pixel_values.shape[-2:], mode='bilinear', align_corners=False)
    classes = upscaled.argmax(dim=1)[0].to(torch.uint8).cpu().numpy()
    return cv2.resize(classes, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)


def class_indices(model):
    return {name: int(index) for index, name in model.config.id2label.items()}
