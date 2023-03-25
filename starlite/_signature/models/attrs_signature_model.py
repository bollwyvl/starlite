from __future__ import annotations

import re
import traceback
from dataclasses import asdict
from datetime import date, datetime, time, timedelta, timezone
from functools import partial
from pathlib import PurePath
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Mapping,
    cast,
)
from uuid import UUID

from _decimal import Decimal
from dateutil.parser import ParserError, parse
from pytimeparse.timeparse import timeparse

from starlite._signature.field import SignatureField
from starlite._signature.models.base import ErrorMessage, SignatureModel
from starlite.connection import ASGIConnection, Request, WebSocket
from starlite.datastructures import ImmutableState, MultiDict, State, UploadFile
from starlite.exceptions import MissingDependencyException
from starlite.params import BodyKwarg, DependencyKwarg, ParameterKwarg
from starlite.types import Empty
from starlite.utils.predicates import get_origin_or_inner_type

try:
    import attr
    import attrs
    import cattrs
except ImportError as e:
    raise MissingDependencyException("attrs is not installed") from e

if TYPE_CHECKING:
    from starlite._signature.parsing import ParsedSignatureParameter
    from starlite.plugins import PluginMapping


key_re = re.compile("@ attribute (.*)|'(.*)'")

__all__ = ("AttrsSignatureModel",)


try:
    import pydantic

    def _structure_base_model(value: Any, cls: type[pydantic.BaseModel]) -> pydantic.BaseModel:
        if isinstance(value, Mapping):
            return cls(**value)
        return cls()

    def _unstructure_base_model(value: pydantic.BaseModel) -> dict[str, Any]:
        return value.dict()

    pydantic_hooks: list[tuple[type[Any], Callable[[Any, type[Any]], Any], Callable[[Any], Any]]] = [
        (pydantic.BaseModel, _structure_base_model, _unstructure_base_model),
    ]
except ImportError:
    pydantic_hooks = []


def _structure_datetime(value: Any, cls: type[datetime]) -> datetime:
    try:
        return cls.utcfromtimestamp(float(value)).replace(tzinfo=timezone.utc)
    except TypeError:
        pass

    try:
        return parse(value)
    except ParserError as e:
        raise ValueError from e


def _unstructure_datetime(value: datetime) -> str:
    return value.isoformat()


def _structure_date(value: Any, cls: type[date]) -> date:
    if isinstance(value, (float, int, Decimal)):
        return cls.fromtimestamp(float(value))

    dt = _structure_datetime(value=value, cls=datetime)
    return cls(year=dt.year, month=dt.month, day=dt.day)


def _unstructure_date(value: date) -> str:
    return value.isoformat()


def _structure_time(value: Any, cls: type[time]) -> time:
    if isinstance(value, str):
        return cls.fromisoformat(value)
    dt = _structure_datetime(value=value, cls=datetime)
    return cls(hour=dt.hour, minute=dt.minute, second=dt.second, microsecond=dt.microsecond, tzinfo=dt.tzinfo)


def _unstructure_time(value: time) -> str:
    return value.isoformat()


def _structure_timedelta(value: Any, cls: type[timedelta]) -> timedelta:
    if isinstance(value, (float, int, Decimal)):
        return cls(seconds=int(value))
    if isinstance(value, str):
        return cls(seconds=timeparse(value))
    raise ValueError


def _unstructure_timedelta(value: timedelta) -> float:
    return value.total_seconds()


def _structure_decimal(value: Any, cls: type[Decimal]) -> Decimal:
    return cls(str(value))


def _unstructure_decimal(value: Decimal) -> str:
    return str(value)


def _structure_path(value: Any, cls: type[PurePath]) -> PurePath:
    return cls(str(value))


def _unstructure_path(value: PurePath) -> str:
    return str(value)


def _structure_uuid(value: Any, cls: type[UUID]) -> UUID:
    return cls(str(value))


def _unstructure_uuid(value: PurePath) -> str:
    return str(value)


def _structure_multidict(value: Any, cls: type[MultiDict]) -> MultiDict:
    return cls(value)


def _unstructure_multidict(value: MultiDict) -> dict[str, Any]:
    return value.dict()


def _structure_starlite_class(value: Any, _: type[Any]) -> Any:
    return value


def _unstructure_starlite_class(value: Any) -> Any:
    return value


hooks: list[tuple[type[Any], Callable[[Any, type[Any]], Any], Callable[[Any], Any]]] = [
    (ASGIConnection, _structure_starlite_class, _unstructure_starlite_class),
    (Decimal, _structure_decimal, _unstructure_decimal),
    (ImmutableState, _structure_starlite_class, _unstructure_starlite_class),
    (PurePath, _structure_path, _unstructure_path),
    (Request, _structure_starlite_class, _unstructure_starlite_class),
    (State, _structure_starlite_class, _unstructure_starlite_class),
    (UUID, _structure_uuid, _unstructure_uuid),
    (UploadFile, _structure_starlite_class, _unstructure_starlite_class),
    (WebSocket, _structure_starlite_class, _unstructure_starlite_class),
    (date, _structure_date, _unstructure_date),
    (datetime, _structure_datetime, _unstructure_datetime),
    (time, _structure_time, _unstructure_time),
    (timedelta, _structure_timedelta, _unstructure_timedelta),
    *pydantic_hooks,
]


class Converter(cattrs.Converter):
    def __init__(self) -> None:
        super().__init__()

        for cls, structure_hook, unstructure_hook in hooks:
            self.register_structure_hook(cls, structure_hook)
            self.register_unstructure_hook(cls, unstructure_hook)


_converter: Converter = Converter()


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
            elif err_format := [line for line in traceback.format_exception(exc) if key_re.search(line)]:
                messages.append({"key": key_re.findall(err_format[0])[0][1].strip(), "message": str(exc)})
    return messages


def _create_validators(
    annotation: Any, kwargs_model: BodyKwarg | ParameterKwarg
) -> list[Callable[[Any, attrs.Attribute[Any], Any], Any]]:
    validators: list[Callable[[Any, attrs.Attribute[Any], Any], Any]] = [
        attrs.validators.instance_of(get_origin_or_inner_type(annotation) or annotation)
    ]

    for value, validator in [
        (kwargs_model.gt, attrs.validators.gt),
        (kwargs_model.ge, attrs.validators.ge),
        (kwargs_model.lt, attrs.validators.lt),
        (kwargs_model.le, attrs.validators.le),
        (kwargs_model.min_length, attrs.validators.min_len),
        (kwargs_model.max_length, attrs.validators.max_len),
        (kwargs_model.min_items, attrs.validators.min_len),
        (kwargs_model.max_items, attrs.validators.max_len),
        (kwargs_model.regex, partial(attrs.validators.matches_re, flags=0)),
    ]:
        if value is not None:
            validators.append(validator(value))  # type: ignore

    return validators


@attr.define
class AttrsSignatureModel(SignatureModel):
    """Model that represents a function signature that uses a pydantic specific type or types."""

    @classmethod
    def parse_values_from_connection_kwargs(cls, connection: ASGIConnection, **kwargs: Any) -> dict[str, Any]:
        try:
            signature = _converter.structure(obj=kwargs, cl=cls)
        except (cattrs.ClassValidationError, ValueError, TypeError, AttributeError) as e:
            raise cls._create_exception(messages=_extract_exceptions(e), connection=connection) from e

        return cast("dict[str, Any]", _converter.unstructure(obj=signature))

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
                    validator=_create_validators(annotation=parameter.annotation, kwargs_model=parameter.default),
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
