"""A U-Net Bridge, frozen CLIP vision tower, and concept head."""

import hashlib
import math
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from transformers import CLIPVisionModelWithProjection


class Bridge(nn.Module):
    """Map up_blocks.2.attentions.2 features into frozen CLIP ViT-L/14 space.
    """

    layer = "up_blocks.2.attentions.2"
    filename = "ub.2.att.2_bridge.pt"

    def __init__(self, clip):
        super().__init__()
        self.project = nn.Conv2d(640, 1024, 1)
        self.residual = nn.Sequential(
            nn.SiLU(), nn.Conv2d(1024, 128, 1), nn.SiLU(), nn.Conv2d(128, 1024, 1)
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.vision = clip.vision_model.float().requires_grad_(False)
        self.projection = clip.visual_projection.float().requires_grad_(False)

    def forward(self, feature, return_tokens=False):
        """Return a unit CLS vector, optionally with the projected patch tokens."""
        if feature.ndim != 4 or feature.shape[1:] != (640, 16, 16) or not torch.isfinite(feature).all():
            raise ValueError("Expected finite [B,640,16,16] U-Net features")
        with torch.autocast(feature.device.type, dtype=torch.bfloat16):
            patch = self.project(feature.float())
            patch = (patch + self.residual(patch)).flatten(2).transpose(1, 2)
            cls = self.vision.embeddings.class_embedding.expand(len(patch), 1, -1)
            hidden = torch.cat((cls, patch), dim=1) + self.vision.embeddings.position_embedding.weight[None]
            hidden = self.vision.pre_layrnorm(hidden)
            tokens = self.vision.encoder(inputs_embeds=hidden).last_hidden_state
            if return_tokens:
                projected = self.projection(self.vision.post_layernorm(tokens))
                projected = F.normalize(projected.float(), dim=-1)
                return projected[:, 0], projected[:, 1:]
            embedding = self.projection(self.vision.post_layernorm(tokens[:, 0]))
            return F.normalize(embedding.float(), dim=-1)


class ConceptHead(nn.Module):
    """A 768 -> 96 -> 1 MLP head on standardized Bridge CLS embeddings."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(768, 96), nn.GELU(), nn.Dropout(0.05), nn.Linear(96, 1))

    def forward(self, embedding):
        return self.net(embedding).squeeze(-1)


class ConceptDetector:
    def __init__(self, bridge, head, mean, scale, thresholds, concept):
        self.bridge = bridge.eval().requires_grad_(False)
        self.head = head.eval().requires_grad_(False)
        self.mean, self.scale, self.concept = mean, scale, concept
        self.thresholds = {int(t): float(v) for t, v in thresholds.items()}

    @contextmanager
    def capture(self, unet):
        features = []

        def save(_module, _args, output):
            feature = output[0] if isinstance(output, (tuple, list)) else getattr(output, "sample", output)
            features.append(F.interpolate(feature[1:2].detach(), size=(16, 16),
                                           mode="bilinear", align_corners=False))

        handle = unet.get_submodule(Bridge.layer).register_forward_hook(save)
        try:
            yield features
        finally:
            handle.remove()

    @torch.no_grad()
    def score(self, features):
        """Return a raw presence logit, not a probability or a spatial map."""
        embedding = self.bridge(features[0])
        logits = self.head((embedding - self.mean) / self.scale)
        return float(logits.item())


def load_detector(model_dir, concept="nudity", *, clip_path=None, device="cuda"):
    """Load a matched Bridge/head pair from one tack-clipxxx directory.
    """
    model_dir = Path(model_dir)
    bridge_path = model_dir / Bridge.filename
    checkpoint = torch.load(bridge_path, map_location="cpu", weights_only=True)
    head_state = torch.load(model_dir / f"{concept}_head.pt", map_location="cpu", weights_only=True)
    clip, info = CLIPVisionModelWithProjection.from_pretrained(
        clip_path or model_dir / "clip", torch_dtype=torch.float32,
        local_files_only=True, use_safetensors=True, output_loading_info=True,
    )
    bridge = Bridge(clip)
    state = bridge.state_dict()
    expected = {name for name, parameter in bridge.named_parameters() if parameter.requires_grad}
    for name, value in checkpoint["state_dict"].items():
        state[name] = value
    bridge.load_state_dict(state)
    head = ConceptHead()
    head.load_state_dict(head_state["state_dict"])
    mean, scale = head_state["normalization_mean"], head_state["normalization_scale"]
    return ConceptDetector(bridge.to(device), head.to(device), mean.to(device), scale.to(device), head_state["thresholds"], concept)
