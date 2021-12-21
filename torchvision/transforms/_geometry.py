from typing import Tuple, Dict, Any

import torchvision.transforms.functional as F
from torchvision.transforms import Transform


class HorizontalFlip(Transform):
    DISPATCHER = F.horizontal_flip


class Resize(Transform):
    DISPATCHER = F.resize

    def __init__(self, size: Tuple[int, int]) -> None:
        super().__init__()
        self.size = size

    def get_params(self, sample: Any) -> Dict[str, Any]:
        return dict(size=self.size)
