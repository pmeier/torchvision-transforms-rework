from typing import Union, Tuple, Dict, Any, Optional

import torch
from torchvision.utils._internal import StrEnum

from ._feature import Feature, DEFAULT


class BoundingBoxFormat(StrEnum):
    # this is just for test purposes
    _SENTINEL = -1
    XYXY = StrEnum.auto()
    XYWH = StrEnum.auto()
    CXCYWH = StrEnum.auto()


def _xywh_to_xyxy(xywh: torch.Tensor) -> torch.Tensor:
    xyxy = xywh.clone()
    xyxy[..., 2:] += xyxy[..., :2]
    return xyxy


def _xyxy_to_xywh(xyxy: torch.Tensor) -> torch.Tensor:
    xywh = xyxy.clone()
    xywh[..., 2:] -= xywh[..., :2]
    return xywh


def _cxcywh_to_xyxy(cxcywh: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = torch.unbind(cxcywh, dim=-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack((x1, y1, x2, y2), dim=-1)


def _xyxy_to_cxcywh(xyxy: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = torch.unbind(xyxy, dim=-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack((cx, cy, w, h), dim=-1)


def convert_bounding_box_format(
    bounding_box: torch.Tensor,
    *,
    old_format: BoundingBoxFormat,
    new_format: BoundingBoxFormat,
) -> torch.Tensor:
    if new_format == old_format:
        return bounding_box

    if old_format == BoundingBoxFormat.XYWH:
        bounding_box = _xywh_to_xyxy(bounding_box)
    elif old_format == BoundingBoxFormat.CXCYWH:
        bounding_box = _cxcywh_to_xyxy(bounding_box)

    if new_format == BoundingBoxFormat.XYWH:
        bounding_box = _xyxy_to_xywh(bounding_box)
    elif new_format == BoundingBoxFormat.CXCYWH:
        bounding_box = _xyxy_to_cxcywh(bounding_box)

    return bounding_box


class BoundingBox(Feature):
    formats = BoundingBoxFormat
    format: BoundingBoxFormat
    image_size: Tuple[int, int]

    @classmethod
    def _parse_meta_data(
        cls,
        format: Union[str, BoundingBoxFormat] = DEFAULT,  # type: ignore[assignment]
        image_size: Optional[Tuple[int, int]] = DEFAULT,  # type: ignore[assignment]
    ) -> Dict[str, Tuple[Any, Any]]:
        if isinstance(format, str):
            format = BoundingBoxFormat[format]
        format_fallback = BoundingBoxFormat.XYXY
        return dict(
            format=(format, format_fallback),
            # FIXME
            image_size=(image_size, (0, 0)),
        )

    def to_format(self, format: Union[str, BoundingBoxFormat]) -> "BoundingBox":
        if isinstance(format, str):
            format = BoundingBoxFormat[format]

        return BoundingBox(
            convert_bounding_box_format(self, old_format=self.format, new_format=format),
            like=self,
            format=format,
        )
