"""Op registry: registration, name lookup, backend normalization, fallback chain.

The registry owns the backend logic exactly once (``naive`` / ``auto`` /
compiled labels), so these tests drive a local ``OpRegistry`` with sentinel
kernels instead of depending on which compiled kernels happen to exist.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest

from memory_machines.ops import dispatch as _dispatch  # noqa: F401  (self-registers ops)
from memory_machines.ops.registry import (
    OpRegistry,
    OpSpec,
    cpu_loader,
    dispatch_op,
    get_op,
    list_ops,
    register_op,
    taichi_loader,
)


def _naive(*args: Any, **kwargs: Any) -> tuple[str, tuple[Any, ...], dict[str, Any]]:
    return ("naive", args, kwargs)


def _make_registry(**spec_kwargs: Any) -> OpRegistry:
    registry = OpRegistry()
    registry.register(OpSpec(name="probe", naive=_naive, **spec_kwargs))
    return registry


@pytest.mark.parametrize(
    "name",
    ["causal_conv1d", "causal_conv1d_update", "universal_pe_attention"],
    ids=["causal_conv1d", "causal_conv1d_update", "universal_pe_attention"],
)
def test_registered_ops_are_lookupable_by_name(name: str) -> None:
    spec = get_op(name)
    assert spec.name == name
    assert callable(spec.naive)
    assert name in list_ops()


def test_get_op_unknown_name_raises_key_error() -> None:
    with pytest.raises(KeyError, match="unknown op 'not_an_op'"):
        get_op("not_an_op")


def test_naive_backend_returns_the_reference() -> None:
    registry = _make_registry()
    assert registry.dispatch("probe", "naive", 1, 2, flag=True) == ("naive", (1, 2), {"flag": True})


@pytest.mark.parametrize("backend", ["naive", "  NAIVE  ", "Naive"], ids=["lower", "padded", "mixed-case"])
def test_backend_normalization_is_case_and_space_insensitive(backend: str) -> None:
    registry = _make_registry()
    assert registry.dispatch("probe", backend)[0] == "naive"


def test_dispatch_unknown_backend_raises_actionable_value_error() -> None:
    registry = _make_registry()
    with pytest.raises(ValueError, match="Unknown backend: 'tpu'"):
        registry.dispatch("probe", "tpu")
    with pytest.raises(ValueError, match="Valid:"):
        registry.dispatch("probe", "tpu")


def test_dispatch_non_str_backend_raises_type_error() -> None:
    registry = _make_registry()
    with pytest.raises(TypeError, match="backend must be a str"):
        registry.dispatch("probe", 3)  # type: ignore[arg-type]


def test_unknown_op_raises_before_backend_resolution() -> None:
    registry = _make_registry()
    with pytest.raises(KeyError, match="unknown op 'missing'"):
        registry.dispatch("missing", "naive")


@pytest.mark.parametrize("backend", ["taichi", "gpu"], ids=["taichi", "gpu"])
def test_requested_taichi_backend_without_kernel_raises_not_implemented(backend: str) -> None:
    registry = _make_registry()
    with pytest.raises(NotImplementedError, match="no Taichi kernel registered"):
        registry.dispatch("probe", backend)


@pytest.mark.parametrize("backend", ["cpu", "compiled"], ids=["cpu", "compiled"])
def test_requested_cpu_backend_without_kernel_raises_not_implemented(backend: str) -> None:
    registry = _make_registry()
    with pytest.raises(NotImplementedError, match="no CPU cpp kernel registered"):
        registry.dispatch("probe", backend)


def test_loader_returning_none_falls_back_to_naive_with_warning() -> None:
    registry = _make_registry(compiled_cpu=lambda: None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = registry.dispatch("probe", "auto", 7)
    assert result == ("naive", (7,), {})
    assert len(caught) == 1
    warning = caught[0]
    assert issubclass(warning.category, UserWarning)
    assert "probe" in str(warning.message)
    assert "falling back to the pure-PyTorch reference" in str(warning.message)


def test_auto_prefers_compiled_kernel_over_naive() -> None:
    registry = _make_registry(compiled_cpu=lambda: lambda *a, **k: ("cpu", a))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = registry.dispatch("probe", "auto", 1)
    assert result == ("cpu", (1,))
    assert caught == []


def test_auto_tries_taichi_before_cpu() -> None:
    order: list[str] = []

    def taichi_kernel(*args: Any) -> str:
        order.append("taichi")
        return "taichi"

    def cpu_kernel(*args: Any) -> str:
        order.append("cpu")
        return "cpu"

    registry = _make_registry(compiled_taichi=lambda: taichi_kernel, compiled_cpu=lambda: cpu_kernel)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a downgrade warning here would fail the test
        assert registry.dispatch("probe", "auto") == "taichi"
    assert order == ["taichi"]


def test_auto_falls_through_taichi_failure_to_cpu() -> None:
    def broken_taichi() -> Any:
        raise RuntimeError("no taichi installed")

    registry = _make_registry(compiled_taichi=broken_taichi, compiled_cpu=lambda: lambda: "cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert registry.dispatch("probe", "auto") == "cpu"


def test_auto_warning_names_every_failed_loader() -> None:
    def broken() -> Any:
        raise RuntimeError("boom")

    registry = _make_registry(compiled_taichi=broken, compiled_cpu=lambda: None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert registry.dispatch("probe", "auto")[0] == "naive"
    message = str(caught[0].message)
    assert "Taichi: boom" in message
    assert "CPU cpp:" in message


def test_explicit_cpu_backend_surfaces_loader_failure() -> None:
    def broken() -> Any:
        raise RuntimeError("kernels not built")

    registry = _make_registry(compiled_cpu=broken)
    with pytest.raises(NotImplementedError, match="kernels not built"):
        registry.dispatch("probe", "cpu")


def test_explicit_taichi_backend_surfaces_loader_failure() -> None:
    def broken() -> Any:
        raise RuntimeError("taichi missing")

    registry = _make_registry(compiled_taichi=broken)
    with pytest.raises(NotImplementedError, match="taichi missing"):
        registry.dispatch("probe", "taichi")


def test_register_rejects_duplicate_name() -> None:
    registry = _make_registry()
    with pytest.raises(ValueError, match="op already registered: 'probe'"):
        registry.register(OpSpec(name="probe", naive=_naive))


def test_register_op_returns_the_spec_and_populates_the_global_registry() -> None:
    name = "test_registry_probe_op"
    spec = register_op(OpSpec(name=name, naive=_naive, description="test-only probe"))
    assert spec.name == name
    assert get_op(name) is spec
    assert name in list_ops()
    assert dispatch_op(name, "naive")[0] == "naive"


def test_opspec_requires_name_and_naive() -> None:
    with pytest.raises(ValueError, match="name must be non-empty"):
        OpSpec(name="", naive=_naive)
    with pytest.raises(ValueError, match="naive implementation required"):
        OpSpec(name="probe", naive=None)  # type: ignore[arg-type]


def test_bundled_loaders_resolve_to_none_until_kernels_land() -> None:
    # Documented contract: a loader returns None until the compiled kernel is
    # written, which is what makes backend="auto" fall back instead of raising.
    assert cpu_loader("causal_conv1d")() is None
    assert taichi_loader("causal_conv1d")() is None
