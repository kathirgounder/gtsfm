"""Base abstraction for geometry transformer models (VGGT, AnySplat, etc.)."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch


@dataclass
class GeometryTransformerConfig:
    """Shared configuration for geometry transformer models."""

    seed: int = 42
    confidence_threshold: float = 5.0
    max_num_points: int = 100000
    dtype: Optional[Union[str, torch.dtype]] = None


@dataclass
class GeometryTransformerOutput:
    """Unified output from a geometry transformer forward pass.

    Core fields are populated by all implementations (VGGT, AnySplat, etc.).
    Optional fields accommodate model-specific outputs like Gaussian splats.
    """

    device: torch.device
    dtype: torch.dtype
    images: torch.Tensor  # (N, 3, H, W)
    extrinsic: torch.Tensor  # (N, 3, 4) or (N, 4, 4)
    intrinsic: torch.Tensor  # (N, 3, 3)
    depth_map: torch.Tensor  # (N, H, W)
    depth_confidence: torch.Tensor  # (N, H, W)
    dense_points: torch.Tensor  # (N, H, W, 3)

    # Optional Gaussian splatting outputs (populated by AnySplat).
    splats: Optional[Any] = None
    decoder: Optional[Any] = None
    gaussian_metadata: Optional[dict[str, Any]] = None


class GeometryTransformer(abc.ABC):
    """Abstract base class for models that predict geometry from multi-view images.

    Subclasses (e.g. VggtGeometryTransformer) implement :meth:`predict` to run
    model inference and return a :class:`GeometryTransformerOutput`.
    """

    @abc.abstractmethod
    def predict(
        self,
        images: torch.Tensor,
        **kwargs: Any,
    ) -> GeometryTransformerOutput:
        """Run model inference on a batch of images.

        Args:
            images: Tensor shaped ``(N, 3, H, W)``.
            **kwargs: Implementation-specific options (e.g. ``model``, ``weights_path``).

        Returns:
            A :class:`GeometryTransformerOutput` containing poses, depths, points, etc.
        """
