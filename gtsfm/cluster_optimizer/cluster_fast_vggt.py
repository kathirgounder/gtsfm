"""FastVGGT-enabled cluster optimizer built on top of :class:`ClusterVGGT`."""

from __future__ import annotations

from typing import Any, Optional, Union

import torch

from gtsfm.cluster_optimizer.cluster_vggt import ClusterVGGT
from gtsfm.frontend.vggt_geometry_transformer import VggtGeometryConfig, VggtGeometryTransformer


class ClusterFastVGGT(ClusterVGGT):
    """Thin wrapper around ``ClusterVGGT`` that wires FastVGGT parameters."""

    def __init__(
        self,
        *args,
        merging: Optional[int] = None,
        vis_attn_map: bool = False,
        enable_protection: bool = False,
        fast_dtype: Optional[Union[str, torch.dtype]] = "bfloat16",
        extra_model_kwargs: Optional[dict[str, Any]] = None,
        geometry_transformer: Optional[VggtGeometryTransformer] = None,
        **kwargs,
    ) -> None:
        """Configure an accelerated VGGT cluster optimizer.

        Args:
            merging: Optional number of tokens to merge, passed to the VGGT constructor.
            vis_attn_map: Whether to enable VGGT's attention-map visualization logic.
            enable_protection: Whether to enable FastVGGT's important-token protection switch.
            fast_dtype: Override for the inference dtype (defaults to BF16 to match FastVGGT).
            extra_model_kwargs: Additional VGGT constructor kwargs to merge after the FastVGGT defaults.
            geometry_transformer: Optional pre-built geometry transformer. If provided, FastVGGT
                model kwargs and dtype are not applied; the transformer is used as-is.
            *args/**kwargs: Forwarded to :class:`ClusterVGGT`.
        """

        if geometry_transformer is None:
            model_kwargs: dict[str, Any] = {}

            if extra_model_kwargs is not None:
                model_kwargs.update(extra_model_kwargs)

            def _setdefault(key: str, value: Any) -> None:
                if value is None:
                    return
                model_kwargs.setdefault(key, value)

            _setdefault("merging", merging)
            _setdefault("enable_point", False)
            _setdefault("enable_track", False)
            if vis_attn_map:
                model_kwargs.setdefault("vis_attn_map", True)
            if enable_protection:
                model_kwargs.setdefault("enable_protection", True)

            geometry_config = VggtGeometryConfig(
                dtype=fast_dtype,
                model_ctor_kwargs=model_kwargs,
            )
            geometry_transformer = VggtGeometryTransformer(config=geometry_config)

        super().__init__(
            *args,
            geometry_transformer=geometry_transformer,
            **kwargs,
        )
