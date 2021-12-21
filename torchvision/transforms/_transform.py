import collections.abc
from typing import Any, Dict, Optional, Callable

from torch import nn
from torchvision import features


class Transform(nn.Module):
    DISPATCHER: Callable

    def get_params(self, sample: Any) -> Dict[str, Any]:
        return dict()

    def _apply_recursively(self, sample: Any, *, params: Dict[str, Any]) -> Any:
        """Recurses through a sample and invokes :meth:`Transform.transform` on non-container elements.
        If an element is not supported by the transform, it is returned untransformed.
        Args:
            sample: Sample.
            params: Parameter dictionary ``params`` that will be passed to ``feature_transform(input, **params)``.
        """
        # We explicitly exclude str's here since they are self-referential and would cause an infinite recursion loop:
        # "a" == "a"[0][0]...
        if isinstance(sample, collections.abc.Sequence) and not isinstance(sample, str):
            return [self._apply_recursively(item, params=params) for item in sample]
        elif isinstance(sample, collections.abc.Mapping):
            return {name: self._apply_recursively(item, params=params) for name, item in sample.items()}
        else:
            feature_type = type(sample)
            if not (issubclass(feature_type, features.Feature) and feature_type is not features.Feature):
                return sample

            # comment
            return type(self).DISPATCHER(sample, **params)

    def forward(
        self,
        *inputs: Any,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        if params is None:
            params = self.get_params(sample)
        return self._apply_recursively(sample, params=params)
