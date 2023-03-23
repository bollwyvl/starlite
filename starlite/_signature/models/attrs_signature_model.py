from __future__ import annotations

import traceback
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, cast

from starlite._signature.field import SignatureField
from starlite._signature.models.base import ErrorMessage, SignatureModel
from starlite.exceptions import MissingDependencyException
from starlite.params import BodyKwarg, DependencyKwarg, ParameterKwarg
from starlite.types import Empty

try:
    import attr
    import attrs
    import cattrs
except ImportError as e:
    raise MissingDependencyException("attrs is not installed") from e

if TYPE_CHECKING:
    from starlite._signature.parsing import ParsedSignatureParameter
    from starlite.connection import ASGIConnection
    from starlite.plugins import PluginMapping

__all__ = ("AttrsSignatureModel",)


def _extract_exceptions(e: Any) -> list[ErrorMessage]:
    """Extracts and normalizes cattrs exceptions.

    Args:
        e: An ExceptionGroup - which is a py3.11 feature. We use hasattr instead of instance checks to avoid installing this.

    Returns:
        A list of normalized exception messages.
    """
    messages: list[ErrorMessage] = []
    if hasattr(e, "exceptions"):
        for exc in cast(list[Exception], e.exceptions):
            if hasattr(exc, "exceptions"):
                messages.extend(_extract_exceptions(exc))
            elif err_format := [line for line in traceback.format_exception(exc) if "@" in line]:
                messages.append(
                    {"key": err_format[0].split("@")[1].replace("attribute", "").strip(), "message": str(exc)}
                )
    return messages


@attr.define
class AttrsSignatureModel(SignatureModel):
    """Model that represents a function signature that uses a pydantic specific type or types."""

    @classmethod
    def parse_values_from_connection_kwargs(cls, connection: ASGIConnection, **kwargs: Any) -> dict[str, Any]:
        try:
            signature = cattrs.structure(kwargs, cls)
        except cattrs.ClassValidationError as e:
            raise cls._create_exception(messages=_extract_exceptions(e), connection=connection) from e

        return cast("dict[str, Any]", cattrs.unstructure(signature))

    def to_dict(self) -> dict[str, Any]:
        return attrs.asdict(self)

    @classmethod
    def populate_signature_fields(cls) -> None:
        cls.fields = {
            k: SignatureField.create(
                field_type=attribute.type,
                name=k,
                default_value=attribute.default if attribute.default is not attr.NOTHING else Empty,
                kwarg_model=attribute.metadata.get("kwargs_model", None) if attribute.metadata else None,
                extra=attribute.metadata or None,
            )
            for k, attribute in attrs.fields_dict(cls).items()
        }

    @classmethod
    def create(
        cls,
        fn_name: str,
        fn_module: str | None,
        parsed_params: list[ParsedSignatureParameter],
        return_annotation: Any,
        field_plugin_mappings: dict[str, PluginMapping],
        dependency_names: set[str],
    ) -> type[SignatureModel]:
        attributes: dict[str, Any] = {}

        for parameter in parsed_params:
            if isinstance(parameter.default, (ParameterKwarg, BodyKwarg)):
                attribute = attr.attrib(
                    type=parameter.annotation,
                    metadata={
                        **asdict(parameter.default),
                        "kwargs_model": parameter.default,
                        "parsed_parameter": parameter,
                    },
                    default=parameter.default.default if parameter.default.default is not Empty else attr.NOTHING,
                )
            elif isinstance(parameter.default, DependencyKwarg):
                attribute = attr.attrib(
                    type=Any if parameter.should_skip_validation else parameter.annotation,
                    default=parameter.default.default if parameter.default.default is not Empty else None,
                )
            elif parameter.should_skip_validation:
                attribute = attr.attrib(type=Any)
            elif parameter.default_defined:
                attribute = attr.attrib(type=parameter.annotation, default=parameter.default)
            else:
                attribute = attr.attrib(type=parameter.annotation, default=None if parameter.optional else attr.NOTHING)

            attributes[parameter.name] = attribute

        model: type[AttrsSignatureModel] = attrs.make_class(
            f"{fn_name}_signature_model",
            attrs=attributes,
            bases=(AttrsSignatureModel,),
            slots=True,
            kw_only=True,
        )
        model.return_annotation = return_annotation  # pyright: ignore
        model.field_plugin_mappings = field_plugin_mappings  # pyright: ignore
        model.dependency_name_set = dependency_names  # pyright: ignore
        model.populate_signature_fields()  # pyright: ignore
        return model
