import os
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.io import read_image

from vtp_detect.Model.vtp_hf import VTPConfig, VTPModel


def _imagenet_normalize() -> T.Normalize:
    return T.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )


def _ceil_to_multiple(n: int, m: int) -> int:
    return (n + m - 1) // m * m


@torch.no_grad()
def _pad_bchw_to_patch_multiple(
    x: torch.Tensor,
    patch_size: int = 16,
    pad_value: float = 0.0,
) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError(f"Expected BCHW tensor, got shape {tuple(x.shape)}")
    _, _, h, w = x.shape
    h2 = _ceil_to_multiple(h, patch_size)
    w2 = _ceil_to_multiple(w, patch_size)
    pad_h, pad_w = h2 - h, w2 - w
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h), value=pad_value)


@torch.no_grad()
def _preprocess(
    image_tensor: torch.Tensor,
    norm: T.Normalize,
    device: Union[str, torch.device],
    dtype: torch.dtype,
    patch_size: int,
) -> Tuple[torch.Tensor, torch.dtype, torch.device]:
    """[B,H,W,3] in [0,1] -> BCHW on device, padded to patch multiple, normalized."""
    input_dtype = image_tensor.dtype
    input_device = image_tensor.device
    x = image_tensor.permute(0, 3, 1, 2).to(device, dtype=torch.float32)
    x = _pad_bchw_to_patch_multiple(x, patch_size=patch_size, pad_value=0.0)
    x = x.to(dtype=dtype)
    x = norm(x)
    return x, input_dtype, input_device


@torch.no_grad()
def _postprocess_feature_dict(
    out: Dict[str, torch.Tensor],
    input_device: torch.device,
    input_dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    return {k: v.to(input_device, dtype=input_dtype) for k, v in out.items()}


class Detector(object):
    """Vision feature extractor wrapping ``VTPModel`` (HF layout)."""

    def __init__(
        self,
        pretrained_path: Union[str, None] = None,
        dtype="auto",
        device: str = "cpu",
        patch_size: Optional[int] = None,
    ) -> None:
        self.device = device
        if dtype == "auto":
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
            elif torch.cuda.is_available():
                self.dtype = torch.float16
            else:
                self.dtype = torch.float32
        elif isinstance(dtype, str):
            td = getattr(torch, dtype, None)
            if td is None or not isinstance(td, torch.dtype):
                raise ValueError(
                    f"Unknown dtype string '{dtype}'. "
                    "Use a torch name like 'float32', 'float16', 'bfloat16'."
                )
            self.dtype = td
        else:
            self.dtype = dtype

        if pretrained_path is not None:
            self.model = VTPModel.from_pretrained(pretrained_path)
            print("[INFO][Detector::__init__]")
            print("\t model loaded from:", pretrained_path)
        else:
            cfg = VTPConfig(train_clip=False, train_reconstruction=False)
            self.model = VTPModel(cfg)

        self.model = self.model.to(self.device, dtype=self.dtype)
        self.model.eval()
        self.model.requires_grad_(False)

        effective_patch = (
            patch_size
            if patch_size is not None
            else int(self.model.config.vision_patch_size)
        )
        self.patch_size = effective_patch

        self._norm = _imagenet_normalize()
        self.is_valid = pretrained_path is not None

    def loadModel(self, pretrained_path: str) -> bool:
        try:
            loaded = VTPModel.from_pretrained(pretrained_path)
        except Exception as e:
            print("[ERROR][Detector::loadModel]")
            print("\t failed to load pretrained:", pretrained_path)
            print("\t", repr(e))
            self.is_valid = False
            return False

        loaded = loaded.to(self.device, dtype=self.dtype)
        loaded.eval()
        loaded.requires_grad_(False)
        self.model = loaded
        if self.patch_size != int(self.model.config.vision_patch_size):
            self.patch_size = int(self.model.config.vision_patch_size)

        print("[INFO][Detector::loadModel]")
        print("\t model loaded from:", pretrained_path)
        self.is_valid = True
        return True

    @torch.no_grad()
    def detect(
        self,
        image_tensor: torch.Tensor,
        use_bottleneck: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Extract last-layer vision features.

        Args:
            image_tensor: ``[B, H, W, 3]``, float, values in ``[0, 1]``. Padded bottom/right to
                a multiple of ``patch_size``, ImageNet-normalized (no resize). RGB order.
            use_bottleneck: Passed to ``VTPModel.get_last_layer_feature`` (reconstruction bottleneck).

        Returns:
            Dict with ``cls_token`` ``[B, D]`` and ``patch_tokens`` ``[B, N, D]``.
        """
        x, input_dtype, input_device = _preprocess(
            image_tensor,
            self._norm,
            self.device,
            self.dtype,
            self.patch_size,
        )

        device_type = self.device if isinstance(self.device, str) else self.device.type
        device_type = device_type.split(":")[0]
        use_amp = device_type == "cuda" and self.dtype in (
            torch.float16,
            torch.bfloat16,
        )
        if use_amp:
            with torch.autocast(device_type, dtype=self.dtype):
                feats = self.model.get_last_layer_feature(x, use_bottleneck=use_bottleneck)
        else:
            feats = self.model.get_last_layer_feature(x, use_bottleneck=use_bottleneck)

        assert isinstance(feats, dict)
        return _postprocess_feature_dict(feats, input_device, input_dtype)

    @torch.no_grad()
    def detectFile(
        self,
        image_file_path: str,
        use_bottleneck: bool = False,
    ) -> Union[Dict[str, torch.Tensor], None]:
        if not os.path.exists(image_file_path):
            print("[ERROR][Detector::detectFile]")
            print("\t image file not exist!")
            print("\t image_file_path:", image_file_path)
            return None

        try:
            img = read_image(image_file_path)
        except Exception:
            print("[ERROR][Detector::detectFile]")
            print("\t failed to read image!")
            print("\t image_file_path:", image_file_path)
            return None

        if img.shape[0] != 3:
            print("[ERROR][Detector::detectFile]")
            print("\t expected 3-channel RGB image")
            return None

        t = (img.float() / 255.0).permute(1, 2, 0).unsqueeze(0)
        return self.detect(t, use_bottleneck=use_bottleneck)
