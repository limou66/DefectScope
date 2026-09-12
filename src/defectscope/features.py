"""Independent feature extractors; no third-party anomaly implementation."""
from pathlib import Path
import numpy as np
from PIL import Image

def load_image(path, size=96):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255

class TextureFeatures:
    name = "texture"
    def __init__(self, size=96, patch=6):
        if size < 24 or patch < 2 or size % patch:
            raise ValueError("size must be >=24 and divisible by patch >=2")
        self.size, self.patch = size, patch

    def __call__(self, image):
        a = np.asarray(image, dtype=np.float32)
        if a.shape != (self.size, self.size, 3) or not np.isfinite(a).all():
            raise ValueError("expected a finite RGB array at configured resolution")
        p, g = self.patch, self.size // self.patch
        blocks = a.reshape(g, p, g, p, 3)
        mean = blocks.mean((1, 3))
        std = blocks.std((1, 3))
        gray = a.mean(2)
        gx = np.abs(np.diff(gray, axis=1, append=gray[:, -1:]))
        gy = np.abs(np.diff(gray, axis=0, append=gray[-1:, :]))
        grad = np.stack([gx, gy], -1).reshape(g, p, g, p, 2).mean((1, 3))
        padded = np.pad(mean, ((1, 1), (1, 1), (0, 0)), mode="edge")
        context = sum(padded[y:y+g, x:x+g] for y in range(3) for x in range(3)) / 9
        return np.concatenate([mean, std, grad, context, mean-context], -1).astype(np.float32)

class ResNetFeatures:
    """Optional frozen ImageNet backbone. Its weights are not our original work."""
    name = "resnet18"
    def __init__(self, size=96, patch=6):
        import os
        os.environ.setdefault("TORCH_HOME", str(Path.cwd()/".cache"/"torch"))
        import torch
        from torchvision.models import resnet18, ResNet18_Weights
        torch.set_num_threads(min(4, torch.get_num_threads()))
        self.torch, self.size, self.patch = torch, size, patch
        if size < 32 or size % patch:
            raise ValueError("invalid deep feature resolution")
        self.net = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
        for param in self.net.parameters():
            param.requires_grad_(False)

    def __call__(self, image):
        torch, net = self.torch, self.net
        x = torch.from_numpy(np.array(image, dtype=np.float32)).permute(2, 0, 1)[None]
        x = (x - torch.tensor([.485, .456, .406])[None, :, None, None]) / torch.tensor([.229, .224, .225])[None, :, None, None]
        with torch.inference_mode():
            x = net.maxpool(net.relu(net.bn1(net.conv1(x))))
            a = net.layer1(x)
            b = net.layer2(a)
            g = self.size // self.patch
            f = torch.cat([torch.nn.functional.adaptive_avg_pool2d(z, (g, g)) for z in (a, b)], 1)
        return f[0].permute(1, 2, 0).numpy()

def make_extractor(name="texture", size=96, patch=6):
    if name == "texture":
        return TextureFeatures(size, patch)
    if name == "resnet18":
        return ResNetFeatures(size, patch)
    raise ValueError("unknown extractor: " + name)
