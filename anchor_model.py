"""
The model meant to replace the silhouette heuristics: a cutout in, 7 anchors out.

A pretrained image backbone with a small regression head on top. Coordinates
rather than heatmaps, because the mirror needs anchors to about a twentieth of a
shoulder width - which regression reaches on a few hundred garments - and a
heatmap decoder is one more moving part to get wrong.

The backbone brings what a garment looks like; only the head has to be taught
where its shoulders are, which is the whole reason a small dataset can work.
"""
import torch
from torch import nn
from transformers import AutoModel

from garment_overlay import POINT_NAMES

CHECKPOINT = "anchor_model.pt"
BACKBONE = "microsoft/resnet-18"
POOL = 4        # the feature map is pooled to POOL x POOL, not to a single vector: a global
                # average throws away where things are, which is all a keypoint is
DROPOUT = 0.2

# What the backbone was pretrained on. Applied inside forward() so that training
# and inference cannot drift apart over a normalisation nobody remembers.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class AnchorNet(nn.Module):
    """Predicts the 7 anchors as fractions of the crop, in POINT_NAMES order."""

    def __init__(self, backbone=BACKBONE):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone)
        channels = self.backbone.config.hidden_sizes[-1]
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(POOL),
            nn.Flatten(),
            nn.Dropout(DROPOUT),
            nn.Linear(channels * POOL * POOL, 2 * len(POINT_NAMES)),
        )
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, images):
        normalised = (images - self.mean) / self.std
        return self.head(self.backbone(normalised).last_hidden_state)


def span_errors(predicted, target):
    """Per-anchor error in reference spans - the same currency the hand-clicked
    garments are scored in, so these numbers compare directly to the heuristics'."""
    predicted = predicted.reshape(-1, len(POINT_NAMES), 2)
    target = target.reshape(-1, len(POINT_NAMES), 2)
    left, right = POINT_NAMES.index("left_shoulder"), POINT_NAMES.index("right_shoulder")
    # Off the labels, never the prediction: a collapsed guess would otherwise
    # divide by nothing and score beautifully.
    span = torch.linalg.vector_norm(target[:, left] - target[:, right], dim=-1).clamp(min=1e-6)
    return torch.linalg.vector_norm(predicted - target, dim=-1) / span[:, None]


def span_error(predicted, target):
    """The same, as one number for a batch."""
    return span_errors(predicted, target).mean()
