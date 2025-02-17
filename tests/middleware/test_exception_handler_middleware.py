import json
from typing import TYPE_CHECKING, Any, Optional

import pytest
from _pytest.capture import CaptureFixture
from starlette.exceptions import HTTPException as StarletteHTTPException
from structlog.testing import capture_logs

from starlite import Request, Response, Starlite, get
from starlite.exceptions import (
    HTTPException,
    InternalServerException,
    ValidationException,
)
from starlite.logging.config import LoggingConfig, StructLoggingConfig
from starlite.middleware.exceptions import ExceptionHandlerMiddleware
from starlite.middleware.exceptions.middleware import get_exception_handler
from starlite.status_codes import HTTP_400_BAD_REQUEST, HTTP_500_INTERNAL_SERVER_ERROR
from starlite.testing import TestClient, create_test_client
from starlite.types import ExceptionHandlersMap

if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture

    from starlite.datastructures import State
    from starlite.types import Scope
    from starlite.types.callable_types import GetLogger


async def dummy_app(scope: Any, receive: Any, send: Any) -> None:
    return None


middleware = ExceptionHandlerMiddleware(dummy_app, False, {})


def test_default_handle_http_exception_handling_extra_object() -> None:
    response = middleware.default_http_exception_handler(
        Request(scope={"type": "http", "method": "GET"}),  # type: ignore
        HTTPException(detail="starlite_exception", extra={"key": "value"}),
    )
    assert response.status_code == HTTP_500_INTERNAL_SERVER_ERROR
    assert json.loads(response.body) == {
        "detail": "starlite_exception",
        "extra": {"key": "value"},
        "status_code": 500,
    }


def test_default_handle_http_exception_handling_extra_none() -> None:
    response = middleware.default_http_exception_handler(
        Request(scope={"type": "http", "method": "GET"}),  # type: ignore
        HTTPException(detail="starlite_exception"),
    )
    assert response.status_code == HTTP_500_INTERNAL_SERVER_ERROR
    assert json.loads(response.body) == {"detail": "starlite_exception", "status_code": 500}


def test_default_handle_starlite_http_exception_handling() -> None:
    response = middleware.default_http_exception_handler(
        Request(scope={"type": "http", "method": "GET"}),  # type: ignore
        HTTPException(detail="starlite_exception"),
    )
    assert response.status_code == HTTP_500_INTERNAL_SERVER_ERROR
    assert json.loads(response.body) == {"detail": "starlite_exception", "status_code": 500}


def test_default_handle_starlite_http_exception_extra_list() -> None:
    response = middleware.default_http_exception_handler(
        Request(scope={"type": "http", "method": "GET"}),  # type: ignore
        HTTPException(detail="starlite_exception", extra=["extra-1", "extra-2"]),
    )
    assert response.status_code == HTTP_500_INTERNAL_SERVER_ERROR
    assert json.loads(response.body) == {
        "detail": "starlite_exception",
        "extra": ["extra-1", "extra-2"],
        "status_code": 500,
    }


def test_default_handle_starlette_http_exception_handling() -> None:
    response = middleware.default_http_exception_handler(
        Request(scope={"type": "http", "method": "GET"}),  # type: ignore
        StarletteHTTPException(detail="starlite_exception", status_code=HTTP_500_INTERNAL_SERVER_ERROR),
    )
    assert response.status_code == HTTP_500_INTERNAL_SERVER_ERROR
    assert json.loads(response.body) == {
        "detail": "starlite_exception",
        "status_code": 500,
    }


def test_default_handle_python_http_exception_handling() -> None:
    response = middleware.default_http_exception_handler(
        Request(scope={"type": "http", "method": "GET"}), AttributeError("oops")  # type: ignore
    )
    assert response.status_code == HTTP_500_INTERNAL_SERVER_ERROR
    assert json.loads(response.body) == {
        "detail": repr(AttributeError("oops")),
        "status_code": HTTP_500_INTERNAL_SERVER_ERROR,
    }


def test_exception_handler_middleware_exception_handlers_mapping() -> None:
    @get("/")
    def handler() -> None:
        ...

    def exception_handler(request: Request, exc: Exception) -> Response:
        return Response(content={"an": "error"}, status_code=HTTP_500_INTERNAL_SERVER_ERROR)

    app = Starlite(route_handlers=[handler], exception_handlers={Exception: exception_handler}, openapi_config=None)
    assert app.asgi_router.root_route_map_node.children["/"].asgi_handlers["GET"][0].exception_handlers == {  # type: ignore
        Exception: exception_handler
    }


def test_exception_handler_middleware_calls_app_level_after_exception_hook() -> None:
    @get("/test")
    def handler() -> None:
        raise RuntimeError()

    async def after_exception_hook_handler(exc: Exception, scope: "Scope", state: "State") -> None:
        assert isinstance(exc, RuntimeError)
        assert scope["app"]
        assert not state.called
        state.called = True

    with create_test_client(handler, after_exception=[after_exception_hook_handler]) as client:
        setattr(client.app.state, "called", False)
        assert not client.app.state.called
        response = client.get("/test")
        assert response.status_code == HTTP_500_INTERNAL_SERVER_ERROR
        assert client.app.state.called


@pytest.mark.parametrize(
    "is_debug, logging_config, should_log",
    [
        (True, LoggingConfig(log_exceptions="debug"), True),
        (False, LoggingConfig(log_exceptions="debug"), False),
        (True, LoggingConfig(log_exceptions="always"), True),
        (False, LoggingConfig(log_exceptions="always"), True),
        (True, LoggingConfig(log_exceptions="never"), False),
        (False, LoggingConfig(log_exceptions="never"), False),
        (True, None, False),
        (False, None, False),
    ],
)
def test_exception_handler_default_logging(
    get_logger: "GetLogger",
    caplog: "LogCaptureFixture",
    is_debug: bool,
    logging_config: Optional[LoggingConfig],
    should_log: bool,
) -> None:
    @get("/test")
    def handler() -> None:
        raise ValueError("Test debug exception")

    app = Starlite([handler], logging_config=logging_config, debug=is_debug)

    with caplog.at_level("ERROR", "starlite"), TestClient(app=app) as client:
        client.app.logger = get_logger("starlite")
        response = client.get("/test")
        assert response.status_code == HTTP_500_INTERNAL_SERVER_ERROR
        assert "Test debug exception" in response.text

        if should_log:
            assert len(caplog.records) == 1
            assert caplog.records[0].levelname == "ERROR"
            assert caplog.records[0].message.startswith(
                "exception raised on http connection to route /test\n\nTraceback (most recent call last):\n"
            )
        else:
            assert not caplog.records
            assert "exception raised on http connection request to route /test" not in response.text


@pytest.mark.parametrize(
    "is_debug, logging_config, should_log",
    [
        (True, StructLoggingConfig(log_exceptions="debug"), True),
        (False, StructLoggingConfig(log_exceptions="debug"), False),
        (True, StructLoggingConfig(log_exceptions="always"), True),
        (False, StructLoggingConfig(log_exceptions="always"), True),
        (True, StructLoggingConfig(log_exceptions="never"), False),
        (False, StructLoggingConfig(log_exceptions="never"), False),
        (True, None, False),
        (False, None, False),
    ],
)
def test_exception_handler_struct_logging(
    get_logger: "GetLogger",
    capsys: CaptureFixture,
    is_debug: bool,
    logging_config: Optional[LoggingConfig],
    should_log: bool,
) -> None:
    @get("/test")
    def handler() -> None:
        raise ValueError("Test debug exception")

    app = Starlite([handler], logging_config=logging_config, debug=is_debug)

    with TestClient(app=app) as client, capture_logs() as cap_logs:
        response = client.get("/test")
        assert response.status_code == HTTP_500_INTERNAL_SERVER_ERROR
        assert "Test debug exception" in response.text

        if should_log:
            assert len(cap_logs) == 1
            assert cap_logs[0].get("connection_type") == "http"
            assert cap_logs[0].get("path") == "/test"
            assert cap_logs[0].get("traceback")
            assert cap_logs[0].get("event") == "uncaught exception"
            assert cap_logs[0].get("log_level") == "error"
        else:
            assert not cap_logs


def test_traceback_truncate_default_logging(
    get_logger: "GetLogger",
    caplog: "LogCaptureFixture",
) -> None:
    @get("/test")
    def handler() -> None:
        raise ValueError("Test debug exception")

    app = Starlite([handler], logging_config=LoggingConfig(log_exceptions="always", traceback_line_limit=1))

    with caplog.at_level("ERROR", "starlite"), TestClient(app=app) as client:
        client.app.logger = get_logger("starlite")
        response = client.get("/test")
        assert response.status_code == HTTP_500_INTERNAL_SERVER_ERROR
        assert "Test debug exception" in response.text

        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == "ERROR"
        assert caplog.records[0].message == (
            "exception raised on http connection to route /test\n\nTraceback (most recent call last):\nValueError: Test debug exception\n"
        )


def test_traceback_truncate_struct_logging() -> None:
    @get("/test")
    def handler() -> None:
        raise ValueError("Test debug exception")

    app = Starlite([handler], logging_config=StructLoggingConfig(log_exceptions="always", traceback_line_limit=1))

    with TestClient(app=app) as client, capture_logs() as cap_logs:
        response = client.get("/test")
        assert response.status_code == HTTP_500_INTERNAL_SERVER_ERROR
        assert len(cap_logs) == 1
        assert cap_logs[0].get("traceback") == "ValueError: Test debug exception\n"


def handler(_: Any, __: Any) -> Any:
    return None


def handler_2(_: Any, __: Any) -> Any:
    return None


@pytest.mark.parametrize(
    ["mapping", "exc", "expected"],
    [
        ({}, Exception, None),
        ({HTTP_400_BAD_REQUEST: handler}, ValidationException(), handler),
        ({InternalServerException: handler}, InternalServerException(), handler),
        ({HTTP_500_INTERNAL_SERVER_ERROR: handler}, Exception(), handler),
        ({TypeError: handler}, TypeError(), handler),
        ({Exception: handler}, ValidationException(), handler),
        ({ValueError: handler}, ValidationException(), handler),
        ({ValidationException: handler}, Exception(), None),
        ({HTTP_500_INTERNAL_SERVER_ERROR: handler}, ValidationException(), None),
        ({HTTP_500_INTERNAL_SERVER_ERROR: handler, HTTPException: handler_2}, ValidationException(), handler_2),
        ({HTTPException: handler, ValidationException: handler_2}, ValidationException(), handler_2),
        ({HTTPException: handler, ValidationException: handler_2}, InternalServerException(), handler),
        ({HTTP_500_INTERNAL_SERVER_ERROR: handler, HTTPException: handler_2}, InternalServerException(), handler),
    ],
)
def test_get_exception_handler(mapping: ExceptionHandlersMap, exc: Exception, expected: Any) -> None:
    assert get_exception_handler(mapping, exc) == expected
