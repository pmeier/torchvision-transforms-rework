# THIS FILE IS auto-generated!!

from typing import Any, Tuple, TypeVar

import torchvision.transforms.functional as F
from torchvision import features

FEATURE_SPECIFIC_DEFAULT = object()

T = TypeVar("T", bound=features.Feature)


__all__ = [
    "horizontal_flip",
    "resize",
]


@F.utils.dispatches
def horizontal_flip(input: T) -> T:
    """ADDME"""
    pass


@F.utils.implements(horizontal_flip, features.Image)
def _horizontal_flip_image(input: features.Image) -> features.Image:
    return features.Image(F.horizontal_flip_image(input), like=input)


@F.utils.implements(horizontal_flip, features.BoundingBox)
def _horizontal_flip_bounding_box(input: features.BoundingBox) -> features.BoundingBox:
    return features.BoundingBox(
        F.horizontal_flip_bounding_box(input.to_format("xyxy"), image_size=input.image_size), like=input, format="xyxy"
    ).to_format(input.format)


@F.utils.dispatches
def resize(
    input: T,
    *,
    size: Tuple[int, int],
    interpolation_mode=FEATURE_SPECIFIC_DEFAULT,  # type: ignore[assignment]
) -> T:
    """ADDME"""
    pass


@F.utils.implements(resize, features.Image)
def _resize_image(
    input: features.Image, *, size: Tuple[int, int], interpolation_mode: str = "bilinear"
) -> features.Image:
    return features.Image(F.resize_image(input, size=size, interpolation_mode=interpolation_mode), like=input)


@F.utils.implements(resize, features.BoundingBox)
def _resize_bounding_box(input: features.BoundingBox, *, size: Tuple[int, int], **_: Any) -> features.BoundingBox:
    return features.BoundingBox(
        F.resize_bounding_box(input.to_format("xyxy"), old_image_size=input.image_size, new_image_size=size),
        like=input,
        image_size=size,
        format="xyxy",
    ).to_format(input.format)


@F.utils.implements(resize, features.SegmentationMask)
def _resize_segmentation_mask(
    input: features.SegmentationMask, *, size: Tuple[int, int], interpolation_mode: str = "nearest"
) -> features.SegmentationMask:
    return features.SegmentationMask(
        F.resize_segmentation_mask(input, size=size, interpolation_mode=interpolation_mode), like=input
    )
