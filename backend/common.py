"""Shared model, preprocessing, DICOM reading and Grad-CAM code."""
import io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models
from monai.transforms import Compose, Resize, ScaleIntensity

CLASSES = ["NORMAL", "PNEUMONIA"]
IMG_SIZE = 224
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# MONAI transforms: resize + intensity scaling to [0, 1]
_base_tf = Compose([Resize((IMG_SIZE, IMG_SIZE)), ScaleIntensity(minv=0.0, maxv=1.0)])


def to_tensor(x) -> torch.Tensor:
    """MONAI returns MetaTensor; convert to a plain float tensor."""
    if hasattr(x, "as_tensor"):
        x = x.as_tensor()
    return torch.as_tensor(x).float()


def load_unnormalised(img: Image.Image) -> torch.Tensor:
    """PIL image -> 3x224x224 tensor in [0, 1]."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32).transpose(2, 0, 1)
    return to_tensor(_base_tf(arr))


def normalise(x: torch.Tensor) -> torch.Tensor:
    return (x - MEAN) / STD


def pil_to_input(img: Image.Image) -> torch.Tensor:
    return normalise(load_unnormalised(img))


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------- model
class DenseNetCAM(nn.Module):
    """DenseNet121 that can also return the last feature maps (for Grad-CAM)."""

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        net = None
        if pretrained:
            try:
                net = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            except Exception as e:  # offline etc.
                print(f"[warn] could not download ImageNet weights ({e}); training from scratch")
        if net is None:
            net = models.densenet121(weights=None)
        self.features = net.features
        self.classifier = nn.Linear(net.classifier.in_features, num_classes)

    def forward(self, x, return_maps: bool = False):
        act = F.relu(self.features(x))  # not in-place, so gradients can be taken
        logits = self.classifier(F.adaptive_avg_pool2d(act, 1).flatten(1))
        return (logits, act) if return_maps else logits


# ------------------------------------------------------------- Grad-CAM
def grad_cam(model: DenseNetCAM, x: torch.Tensor, class_idx=None):
    """Returns (cam[224x224] in 0..1, class probabilities, class index)."""
    model.eval()
    with torch.enable_grad():
        logits, act = model(x, return_maps=True)
        probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
        c = int(logits.argmax(1).item()) if class_idx is None else class_idx
        grads = torch.autograd.grad(logits[0, c], act)[0]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * act).sum(1, keepdim=True)).detach()
    cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam.cpu().numpy(), probs, c


def _jet(v: np.ndarray) -> np.ndarray:
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def heatmap_rgba(cam: np.ndarray, size: int = 448) -> Image.Image:
    """Transparent PNG heatmap: low attention is see-through."""
    c = Image.fromarray((cam * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)
    c = np.asarray(c, dtype=np.float32) / 255.0
    rgb = (_jet(c) * 255).astype(np.uint8)
    alpha = (np.clip((c - 0.25) / 0.75, 0, 1) * 210).astype(np.uint8)
    return Image.fromarray(np.dstack([rgb, alpha]), "RGBA")


# ------------------------------------------------------------ file input
def read_image(data: bytes, filename: str):
    """Reads PNG/JPG or DICOM bytes. Returns (PIL RGB image, is_dicom)."""
    is_dicom = filename.lower().endswith((".dcm", ".dicom")) or data[128:132] == b"DICM"
    if not is_dicom:
        return Image.open(io.BytesIO(data)).convert("RGB"), False

    import pydicom

    ds = pydicom.dcmread(io.BytesIO(data))
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 3:  # RGB image or multi-frame (CT/MRI volume)
        arr = arr[..., 0] if arr.shape[-1] in (3, 4) else arr[arr.shape[0] // 2]
    arr = arr * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))
    lo, hi = np.percentile(arr, (1, 99))
    arr = np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = 1 - arr
    return Image.fromarray((arr * 255).astype(np.uint8)).convert("RGB"), True
