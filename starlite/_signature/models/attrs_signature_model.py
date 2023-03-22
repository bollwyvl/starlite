from __future__ import annotations

from typing import TYPE_CHECKING, Any

from starlite._signature.models.base import SignatureModel

if TYPE_CHECKING:
    from starlite._signature.parsing import ParsedSignatureParameter
    from starlite.connection import ASGIConnection
    from starlite.plugins import PluginMapping

__all__ = ("AttrsSignatureModel",)


class AttrsSignatureModel(SignatureModel):
    """Model that represents a function signature that uses a pydantic specific type or types."""

    @classmethod
    def parse_values_from_connection_kwargs(cls, connection: ASGIConnection, **kwargs: Any) -> dict[str, Any]:
        pass

    def to_dict(self) -> dict[str, Any]:
        pass

    @classmethod
    def populate_signature_fields(cls) -> None:
        pass

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
        pass
