from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from starlite._signature.parsing.utils import parse_fn_signature
from starlite.exceptions import ImproperlyConfiguredException
from starlite.types import AnyCallable, Empty
from starlite.utils.helpers import unwrap_partial

if TYPE_CHECKING:
    from starlite._signature.models.base import SignatureModel
    from starlite.plugins import SerializationPluginProtocol

try:
    from starlite._signature.models.pydantic_signature_model import PydanticSignatureModel
except ImportError:
    PydanticSignatureModel = Empty  # type: ignore


try:
    from starlite._signature.models.attrs_signature_model import AttrsSignatureModel
except ImportError:
    AttrsSignatureModel = Empty  # type: ignore

__all__ = ("create_signature_model", "get_signature_model")


def get_signature_model(value: Any) -> type[SignatureModel]:
    """Retrieve and validate the signature model from a provider or handler."""
    try:
        return cast("type[SignatureModel]", value.signature_model)
    except AttributeError as e:  # pragma: no cover
        raise ImproperlyConfiguredException(f"The 'signature_model' attribute for {value} is not set") from e


def create_signature_model(
    fn: AnyCallable,
    plugins: list[SerializationPluginProtocol],
    dependency_name_set: set[str],
    preferred_backend: Literal["pydantic", "attrs"] = "attrs",
) -> type[SignatureModel]:
    """Create a model for a callable's signature. The model can than be used to parse and validate before passing it to
    the callable.

    Args:
        fn: A callable.
        plugins: A list of plugins.
        dependency_name_set: A set of dependency names
        preferred_backend: Validation/Parsing backend to prefer, if installed

    Returns:
        A _signature model.
    """
    is_pydantic_installed = PydanticSignatureModel is not Empty  # type: ignore[comparison-overlap]
    is_attrs_installed = AttrsSignatureModel is not Empty  # type: ignore[comparison-overlap]

    unwrapped_fn = cast("AnyCallable", unwrap_partial(fn))
    fn_name = getattr(fn, "__name__", "anonymous")
    fn_module = getattr(fn, "__module__", None)

    if fn_name == "<lambda>":
        fn_name = "anonymous"

    parsed_params, return_annotation, field_plugin_mappings, dependency_names = parse_fn_signature(
        fn=unwrapped_fn,
        plugins=plugins,
        dependency_name_set=dependency_name_set,
    )

    should_prefer_pydantic = (
        (is_pydantic_installed and preferred_backend == "pydantic")
        or (is_pydantic_installed and not is_attrs_installed)
        or (
            is_pydantic_installed
            and any(p.annotation and hasattr(p.annotation, "__get_validators__") for p in parsed_params)
        )
    )

    model_class = cast("SignatureModel", PydanticSignatureModel if should_prefer_pydantic else AttrsSignatureModel)

    return model_class.create(
        fn_name=fn_name,
        fn_module=fn_module,
        parsed_params=parsed_params,
        return_annotation=return_annotation,
        field_plugin_mappings=field_plugin_mappings,
        dependency_names={*dependency_name_set, *dependency_names},
    )
