from __future__ import annotations

import math
import shutil
import string
from datetime import timedelta
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, Mock, patch

import anyio
import pytest
from _pytest.fixtures import FixtureRequest

from starlite.exceptions import ImproperlyConfiguredException
from starlite.stores.file import FileStore
from starlite.stores.memory import MemoryStore
from starlite.stores.redis import RedisStore
from starlite.stores.registry import StoreRegistry

if TYPE_CHECKING:
    from redis.asyncio import Redis

    from starlite.stores.base import NamespacedStore, Store


@pytest.fixture()
def mock_redis() -> None:
    patch("starlite.Store.redis_backend.Redis")


async def test_get(store: Store) -> None:
    key = "key"
    value = b"value"
    assert await store.get("foo") is None

    await store.set(key, value, 60)

    stored_value = await store.get(key)
    assert stored_value == value


async def test_set(store: Store) -> None:
    values: dict[str, bytes | str] = {"key_1": b"value_1", "key_2": "value_2"}

    for key, value in values.items():
        await store.set(key, value)

    for key, value in values.items():
        stored_value = await store.get(key)
        assert stored_value == value if isinstance(value, bytes) else value.encode("utf-8")


@pytest.mark.parametrize("key", [*list(string.punctuation), "foo\xc3\xbc", ".."])
async def test_set_special_chars_key(store: Store, key: str) -> None:
    # ensure that stores handle special chars correctly
    value = b"value"

    await store.set(key, value)
    assert await store.get(key) == value


async def test_expires(store: Store) -> None:
    expiry = 0.01 if not isinstance(store, RedisStore) else 1  # redis doesn't allow fractional values
    await store.set("foo", b"bar", expires_in=expiry)  # type: ignore[arg-type]

    await anyio.sleep(expiry + 0.01)

    stored_value = await store.get("foo")

    assert stored_value is None


@pytest.mark.parametrize("renew_for", [10, timedelta(seconds=10)])
async def test_get_and_renew(store: Store, renew_for: int | timedelta) -> None:
    expiry = 0.01 if not isinstance(store, RedisStore) else 1  # redis doesn't allow fractional values

    await store.set("foo", b"bar", expires_in=expiry)  # type: ignore[arg-type]
    await store.get("foo", renew_for=renew_for)
    await anyio.sleep(expiry + 0.01)

    stored_value = await store.get("foo")

    assert stored_value is not None


async def test_delete(store: Store) -> None:
    key = "key"
    await store.set(key, b"value", 60)

    await store.delete(key)

    fake_redis_value = await store.get(key)
    assert fake_redis_value is None


async def test_delete_empty(store: Store) -> None:
    # assert that this does not raise an exception
    await store.delete("foo")


async def test_exists(store: Store) -> None:
    assert await store.exists("foo") is False

    await store.set("foo", b"bar")

    assert await store.exists("foo") is True


async def test_expires_in_not_set(store: Store) -> None:
    assert await store.expires_in("foo") is None

    await store.set("foo", b"bar")
    assert await store.expires_in("foo") == -1


async def test_delete_all(store: Store) -> None:
    keys = []
    for i in range(10):
        key = f"key-{i}"
        keys.append(key)
        await store.set(key, b"value", expires_in=10 if i % 2 else None)

    await store.delete_all()

    assert not any([await store.get(key) for key in keys])


async def test_expires_in(store: Store) -> None:
    assert await store.expires_in("foo") is None

    await store.set("foo", "bar")
    assert await store.expires_in("foo") == -1

    await store.set("foo", "bar", expires_in=10)
    assert math.ceil(await store.expires_in("foo") / 10) * 10 == 10  # type: ignore[operator]


@patch("starlite.stores.redis.Redis")
@patch("starlite.stores.redis.ConnectionPool.from_url")
def test_redis_with_client_default(connection_pool_from_url_mock: Mock, mock_redis: Mock) -> None:
    backend = RedisStore.with_client()
    connection_pool_from_url_mock.assert_called_once_with(
        url="redis://localhost:6379", db=None, port=None, username=None, password=None, decode_responses=False
    )
    mock_redis.assert_called_once_with(connection_pool=connection_pool_from_url_mock.return_value)
    assert backend._redis is mock_redis.return_value


@patch("starlite.stores.redis.Redis")
@patch("starlite.stores.redis.ConnectionPool.from_url")
def test_redis_with_non_default(connection_pool_from_url_mock: Mock, mock_redis: Mock) -> None:
    url = "redis://localhost"
    db = 2
    port = 1234
    username = "user"
    password = "password"
    backend = RedisStore.with_client(url=url, db=db, port=port, username=username, password=password)
    connection_pool_from_url_mock.assert_called_once_with(
        url=url, db=db, port=port, username=username, password=password, decode_responses=False
    )
    mock_redis.assert_called_once_with(connection_pool=connection_pool_from_url_mock.return_value)
    assert backend._redis is mock_redis.return_value


async def test_redis_delete_all(redis_store: RedisStore) -> None:
    await redis_store._redis.set("test_key", b"test_value")

    keys = []
    for i in range(10):
        key = f"key-{i}"
        keys.append(key)
        await redis_store.set(key, b"value", expires_in=10 if i % 2 else None)

    await redis_store.delete_all()

    assert not any([await redis_store.get(key) for key in keys])
    assert await redis_store._redis.get("test_key") == b"test_value"  # check it doesn't delete other values


async def test_redis_delete_all_no_namespace_raises(fake_redis: Redis) -> None:
    redis_store = RedisStore(redis=fake_redis, namespace=None)

    with pytest.raises(ImproperlyConfiguredException):
        await redis_store.delete_all()


def test_redis_namespaced_key(redis_store: RedisStore) -> None:
    assert redis_store.namespace == "STARLITE"
    assert redis_store._make_key("foo") == "STARLITE:foo"


def test_redis_with_namespace(redis_store: RedisStore) -> None:
    namespaced_test = redis_store.with_namespace("TEST")
    namespaced_test_foo = namespaced_test.with_namespace("FOO")
    assert namespaced_test.namespace == "STARLITE_TEST"
    assert namespaced_test_foo.namespace == "STARLITE_TEST_FOO"
    assert namespaced_test._redis is redis_store._redis


def test_redis_namespace_explicit_none(fake_redis: Redis) -> None:
    assert RedisStore.with_client(url="redis://127.0.0.1", namespace=None).namespace is None
    assert RedisStore(redis=fake_redis, namespace=None).namespace is None


async def test_file_init_directory(file_store: FileStore) -> None:
    shutil.rmtree(file_store.path)
    await file_store.set("foo", b"bar")


async def test_file_path(file_store: FileStore) -> None:
    await file_store.set("foo", b"bar")

    assert await (file_store.path / "foo").exists()


def test_file_with_namespace(file_store: FileStore) -> None:
    namespaced = file_store.with_namespace("foo")
    assert namespaced.path == file_store.path / "foo"


@pytest.mark.parametrize("invalid_char", string.punctuation)
def test_file_with_namespace_invalid_namespace_char(file_store: FileStore, invalid_char: str) -> None:
    with pytest.raises(ValueError):
        file_store.with_namespace("foo" + invalid_char)


@pytest.fixture(params=["redis_store", "file_store"])
def namespaced_store(request: FixtureRequest) -> NamespacedStore:
    return cast("NamespacedStore", request.getfixturevalue(request.param))


async def test_namespaced_store_get_set(namespaced_store: NamespacedStore) -> None:
    foo_namespaced = namespaced_store.with_namespace("foo")
    await namespaced_store.set("bar", b"starlite namespace")
    await foo_namespaced.set("bar", b"foo namespace")

    assert await namespaced_store.get("bar") == b"starlite namespace"
    assert await foo_namespaced.get("bar") == b"foo namespace"


async def test_namespaced_store_does_not_propagate_up(namespaced_store: NamespacedStore) -> None:
    foo_namespace = namespaced_store.with_namespace("FOO")
    await foo_namespace.set("foo", b"foo-value")
    await namespaced_store.set("bar", b"bar-value")

    await foo_namespace.delete_all()

    assert await foo_namespace.get("foo") is None
    assert await namespaced_store.get("bar") == b"bar-value"


async def test_namespaced_store_delete_all_propagates_down(namespaced_store: NamespacedStore) -> None:
    foo_namespace = namespaced_store.with_namespace("FOO")
    await foo_namespace.set("foo", b"foo-value")
    await namespaced_store.set("bar", b"bar-value")

    await namespaced_store.delete_all()

    assert await foo_namespace.get("foo") is None
    assert await namespaced_store.get("bar") is None


@pytest.mark.parametrize("store_fixture", ["memory_store", "file_store"])
async def test_memory_delete_expired(store_fixture: str, request: FixtureRequest) -> None:
    store = request.getfixturevalue(store_fixture)

    expect_expired: list[str] = []
    expect_not_expired: list[str] = []
    for i in range(10):
        key = f"key-{i}"
        expires_in = 0.001 if i % 2 == 0 else None
        await store.set(key, b"value", expires_in=expires_in)
        (expect_expired if expires_in else expect_not_expired).append(key)

    await anyio.sleep(0.002)
    await store.delete_expired()

    assert not any([await store.exists(key) for key in expect_expired])
    assert all([await store.exists(key) for key in expect_not_expired])


def test_registry_get(memory_store: MemoryStore) -> None:
    default_factory = MagicMock()
    default_factory.return_value = memory_store
    registry = StoreRegistry(default_factory=default_factory)
    default_factory.reset_mock()

    assert registry.get("foo") is memory_store
    assert registry.get("foo") is memory_store
    assert "foo" in registry._stores
    default_factory.assert_called_once_with("foo")


def test_registry_register(memory_store: MemoryStore) -> None:
    registry = StoreRegistry()

    registry.register("foo", memory_store)

    assert registry.get("foo") is memory_store


def test_registry_register_exist_raises(memory_store: MemoryStore) -> None:
    registry = StoreRegistry({"foo": memory_store})

    with pytest.raises(ValueError):
        registry.register("foo", memory_store)


def test_registry_register_exist_override(memory_store: MemoryStore) -> None:
    registry = StoreRegistry({"foo": memory_store})

    registry.register("foo", memory_store, allow_override=True)
    assert registry.get("foo") is memory_store
