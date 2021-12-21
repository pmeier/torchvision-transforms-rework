import functools
import inspect
import pathlib
from copy import copy
from textwrap import dedent
from textwrap import indent as _indent
from typing import Any
import importlib

import warnings

try:
    import yaml
except ModuleNotFoundError:
    raise ModuleNotFoundError() from None

HERE = pathlib.Path(__file__).parent


try:
    import ufmt

    with open(HERE.parent / ".pre-commit-config.yaml") as file:
        repo = next(
            repo for repo in yaml.load(file, yaml.Loader)["repos"] for hook in repo["hooks"] if hook["id"] == "ufmt"
        )

    expected_versions = {ufmt: repo["rev"].replace("v", "")}
    for dependency in repo["hooks"][0]["additional_dependencies"]:
        name, version = [item.strip() for item in dependency.split("==")]
        expected_versions[importlib.import_module(name)] = version

    for module, expected_version in expected_versions.items():
        if module.__version__ != expected_version:
            warnings.warn("foo")
except ModuleNotFoundError:
    ufmt = None

import torchvision.transforms.functional as F

from torchvision import features
from torchvision.utils._internal import camel_to_snake_case


class ManualAnnotation:
    def __init__(self, repr):
        self.repr = repr

    def __repr__(self):
        return self.repr


FEATURE_SPECIFIC_DEFAULT = ManualAnnotation("FEATURE_SPECIFIC_DEFAULT")
GENERIC_FEATURE_TYPE = ManualAnnotation("T")


FUNCTIONAL_ROOT = HERE.parent / "torchvision" / "transforms" / "functional"


def main(config_path=FUNCTIONAL_ROOT / "dispatch.yaml", dispatch_path=FUNCTIONAL_ROOT / "_dispatch.py"):
    with open(config_path) as file:
        dispatch_config = yaml.load(file, yaml.Loader)

    functions = []

    for dispatcher_name, feature_type_configs in dispatch_config.items():
        feature_type_configs = validate_feature_type_configs(feature_type_configs)
        kernel_params, implementer_params = make_kernel_and_implementer_params(feature_type_configs)
        dispatcher_params = make_dispatcher_params(implementer_params)

        functions.append(DispatcherFunction(name=dispatcher_name, params=dispatcher_params))
        functions.extend(
            [
                IMPLEMENTER_FUNCTION_TYPE_MAP.get(feature_type, ImplementerFunction)(
                    dispatcher_name=dispatcher_name,
                    feature_type=feature_type,
                    params=implementer_params[feature_type],
                    kernel=config["kernel"],
                    kernel_params=kernel_params[feature_type],
                    kernel_param_name_map=config["kwargs_overwrite"],
                    meta_overwrite=config["meta_overwrite"],
                )
                for feature_type, config in feature_type_configs.items()
            ]
        )

    with open(dispatch_path, "w") as file:
        file.write(ufmt_format(make_file_content(functions)))


def validate_feature_type_configs(feature_type_configs):
    try:
        feature_type_configs = {
            getattr(features, feature_type_name): config for feature_type_name, config in feature_type_configs.items()
        }
    except AttributeError as error:
        # unknown feature type
        raise

    for feature_type, config in feature_type_configs.items():
        optional_keys = {"kwargs_overwrite", "meta_overwrite"}
        unknown_keys = config.keys() - {"kernel", *optional_keys}
        if unknown_keys:
            raise KeyError

        try:
            config["kernel"] = getattr(F, config["kernel"])
        except KeyError:
            # no kernel provided
            raise
        except AttributeError:
            # kernel not accessible
            raise

        # check kernel signature with current transforms logic
        # better: check unary
        signature = inspect.signature(config["kernel"])

        for key in optional_keys:
            config.setdefault(key, dict())

    # TODO: bunchify the individual configs
    return feature_type_configs


def make_kernel_and_implementer_params(feature_type_configs):
    kernel_params = {}
    implementer_params = {}
    for feature_type, config in feature_type_configs.items():
        kernel_params[feature_type] = list(inspect.signature(config["kernel"]).parameters.values())[1:]
        implementer_params[feature_type] = [
            Parameter(
                name=config["kwargs_overwrite"].get(kernel_param.name, kernel_param.name),
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=kernel_param.default,
                annotation=kernel_param.annotation,
            )
            for kernel_param in kernel_params[feature_type]
            if not config["kwargs_overwrite"].get(kernel_param.name, "").startswith(".")
        ]
    return kernel_params, implementer_params


def make_dispatcher_params(implementer_params):
    # not using a set here to keep the order
    dispatcher_param_names = []
    for params in implementer_params.values():
        dispatcher_param_names.extend([param.name for param in params])
    dispatcher_param_names = unique(dispatcher_param_names)

    dispatcher_params = []
    need_kwargs_ignore = set()
    for name in dispatcher_param_names:
        dispatcher_param_candidates = set()
        for feature_type, params in implementer_params.items():
            params = {param.name: param for param in params}
            if name not in params:
                need_kwargs_ignore.add(feature_type)
                continue

            dispatcher_param_candidates.add(params[name])

        if len(dispatcher_param_candidates) == 1:
            dispatcher_params.append(copy(dispatcher_param_candidates.pop()))
            continue

        if len({param.annotation for param in dispatcher_param_candidates}) > 1:
            raise TypeError

        for param in dispatcher_param_candidates:
            param.feature_specific_default = True

        dispatcher_params.append(Parameter(name=name, kind=Parameter.KEYWORD_ONLY, default=FEATURE_SPECIFIC_DEFAULT))

    for feature_type in need_kwargs_ignore:
        implementer_params[feature_type].append(Parameter(name="_", kind=Parameter.VAR_KEYWORD, annotation=Any))

    return dispatcher_params


def make_file_content(functions):
    header = dedent(
        f"""
        # THIS FILE IS auto-generated!!

        from typing import Any, Tuple, TypeVar
        from torchvision import features
        import torchvision.transforms.functional as F

        {FEATURE_SPECIFIC_DEFAULT} = object()

        {GENERIC_FEATURE_TYPE} = TypeVar("{GENERIC_FEATURE_TYPE}", bound=features.Feature)
        """
    ).strip()

    __all__ = "\n".join(
        (
            "__all__ = [",
            *[
                indent(f"{format_value(function.name)},")
                for function in functions
                if isinstance(function, DispatcherFunction)
            ],
            "]",
        )
    )
    return (
        "\n\n\n".join(
            (
                header,
                __all__,
                *[str(function) for function in functions],
            )
        )
        + "\n"
    )


class Parameter(inspect.Parameter):
    def __init__(
        self,
        *args,
        feature_specific_default=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.feature_specific_default = feature_specific_default


class Signature(inspect.Signature):
    def __str__(self):
        parts = super().__str__().split(repr(FEATURE_SPECIFIC_DEFAULT))
        return f"{FEATURE_SPECIFIC_DEFAULT},  # type: ignore[assignment]\n".join(
            [
                parts[0],
                *[part.lstrip(",") for part in parts[1:]],
            ]
        )


class Function:
    def __init__(self, *, decorator=None, name, signature, docstring=None, body=("pass",)):
        self.decorator = decorator
        self.name = name
        self.signature = signature
        self.docstring = docstring
        self.body = body

    def __str__(self):
        lines = []
        if self.decorator:
            lines.append(f"@{self.decorator}")
        lines.append(f"def {self.name}{self.signature}:")
        if self.docstring:
            lines.append(indent('"""' + self.docstring + '"""'))
        lines.extend([indent(line) for line in self.body])
        return "\n".join(lines)


class DispatcherFunction(Function):
    def __init__(self, *, name, params, input_name="input"):
        for param in params:
            param._kind = Parameter.KEYWORD_ONLY
        signature = Signature(
            parameters=[
                Parameter(
                    name=input_name,
                    kind=Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=GENERIC_FEATURE_TYPE,
                ),
                *params,
            ],
            return_annotation=GENERIC_FEATURE_TYPE,
        )
        super().__init__(
            decorator="F.utils.dispatches",
            name=name,
            signature=signature,
            docstring="ADDME",
        )


class ImplementerFunction(Function):
    def __init__(
        self,
        *,
        dispatcher_name,
        feature_type,
        input_name="input",
        params,
        kernel,
        kernel_params,
        kernel_param_name_map,
        meta_overwrite,
    ):
        feature_type_usage = ManualAnnotation(f"features.{feature_type.__name__}")

        body = []
        params_without_default = [
            Parameter(name=input_name, kind=inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=feature_type_usage)
        ]
        for param in params:
            if param.feature_specific_default:
                body.extend(
                    (
                        f"if {param.name} is {FEATURE_SPECIFIC_DEFAULT}:",
                        indent(f"{param.name} = {format_value(param.default)}"),
                    )
                )

            params_without_default.append(Parameter(name=param.name, kind=param.kind, annotation=param.annotation))

        kernel_call = self._make_kernel_call(
            input_name=input_name,
            kernel=kernel,
            kernel_params=kernel_params,
            kernel_param_name_map=kernel_param_name_map,
        )
        feature_type_wrapper = self._make_feature_type_wrapper(
            kernel_call,
            input_name=input_name,
            meta_overwrite=meta_overwrite,
            feature_type_usage=feature_type_usage,
        )
        body.append(f"return {feature_type_wrapper}")

        super().__init__(
            decorator=f"F.utils.implements({dispatcher_name}, {feature_type_usage})",
            name=f"_{dispatcher_name}_{camel_to_snake_case(feature_type.__name__)}",
            signature=Signature(parameters=params_without_default, return_annotation=feature_type_usage),
            body=body,
        )

    def _make_kernel_call(
        self,
        *,
        input_name,
        kernel,
        kernel_params,
        kernel_param_name_map,
    ):
        call_args = [input_name]
        for param in kernel_params:
            dispatcher_param_name = kernel_param_name_map.get(param.name, param.name)
            if dispatcher_param_name.startswith("."):
                dispatcher_param_name = input_name + dispatcher_param_name
            call_args.append(f"{param.name}={dispatcher_param_name}")
        return f"F.{kernel.__name__}({', '.join(call_args)})"

    def _make_feature_type_wrapper(self, content, *, input_name, meta_overwrite, feature_type_usage):
        wrapper = f"{feature_type_usage}({content}, like={input_name}"
        meta_overwrite_call_args = ", ".join(
            f"{meta_name}={dispatcher_param_name}" for meta_name, dispatcher_param_name in meta_overwrite.items()
        )
        if meta_overwrite_call_args:
            wrapper += f", {meta_overwrite_call_args}"
        return f"{wrapper})"


class BoundingBoxImplementerFunction(ImplementerFunction):
    def __init__(self, *, meta_overwrite, **params):
        if "format" in meta_overwrite:
            raise KeyError("format will be set automatically")
        meta_overwrite["format"] = format_value("xyxy")
        super().__init__(meta_overwrite=meta_overwrite, **params)

    def _make_kernel_call(self, *, input_name, kernel, **kwargs):
        return (
            super()
            ._make_kernel_call(input_name=input_name, kernel=kernel, **kwargs)
            .replace(
                f"{kernel.__name__}({input_name}",
                f'{kernel.__name__}({input_name}.to_format("xyxy")',
            )
        )

    def _make_feature_type_wrapper(self, content, *, input_name, **kwargs):
        return (
            f"{super()._make_feature_type_wrapper(content, input_name=input_name, **kwargs)}"
            f".to_format({input_name}.format)"
        )


IMPLEMENTER_FUNCTION_TYPE_MAP = {features.BoundingBox: BoundingBoxImplementerFunction}


def ufmt_format(content):
    if ufmt is None:
        return content

    from ufmt.core import make_black_config
    from usort.config import Config as UsortConfig

    black_config = make_black_config(HERE)
    usort_config = UsortConfig.find(HERE)

    return ufmt.ufmt_string(path=HERE, content=content, usort_config=usort_config, black_config=black_config)


def indent(text, level=1):
    return _indent(text, prefix=" " * (level * 4))


def format_value(default):
    if isinstance(default, str):
        return f'"{default}"'

    return repr(default)


def unique(seq):
    return functools.reduce(
        lambda unique_list, item: unique_list.append(item) or unique_list if item not in unique_list else unique_list,
        seq,
        [],
    )


if __name__ == "__main__":
    main()
