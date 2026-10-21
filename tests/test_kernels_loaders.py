"""Fallback contracts for the lazy compiled-kernel loaders.

This checkout ships no CPU cpp sources (``kernels/native/cpu`` is empty) and no
Taichi kernel submodules, so the loaders must resolve to ``None`` instead of
raising and the op dispatcher must degrade to the pure-PyTorch reference.

Anything that would need a real compiled artifact is exercised through fakes at
the module boundary (``load``/``import_module``/``ti.init``); no test pretends a
kernel was built.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
import warnings
from pathlib import Path
from typing import Any

import pytest
import torch

from memory_machines.kernels import cpu_loader as cpu_loader_module
from memory_machines.kernels import taichi as taichi_loader_module
from memory_machines.ops.registry import OpRegistry, OpSpec
from memory_machines.ops.registry import cpu_loader as make_cpu_loader


def _point_at_native_dir(monkeypatch: pytest.MonkeyPatch, native_dir: Path) -> None:
    """Repoint the loader at ``native_dir`` and clear its cached extension."""
    monkeypatch.setattr(cpu_loader_module, "_NATIVE_DIR", native_dir)
    monkeypatch.setattr(cpu_loader_module, "_ext", None)


def _install_fake_load(monkeypatch: pytest.MonkeyPatch, result: Any, calls: list[dict[str, Any]]) -> None:
    """Replace ``torch.utils.cpp_extension.load`` with a recorder.

    ``result`` is returned on success, or raised when it is an exception.
    """

    def fake_load(*, name: str, sources: list[str], verbose: bool) -> Any:
        calls.append({"name": name, "sources": list(sources), "verbose": verbose})
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(cpu_loader_module, "load", fake_load)


class _FakeTi:
    """Stand-in for the ``taichi`` module: records ``init`` arch selections."""

    def __init__(self) -> None:
        self.cpu = "ARCH_CPU"
        self.cuda = "ARCH_CUDA"
        self.vulkan = "ARCH_VULKAN"
        self.init_calls: list[Any] = []

    def init(self, arch: Any) -> None:
        self.init_calls.append(arch)


def _load_fresh_taichi_loader() -> types.ModuleType:
    """Execute the loader module body again without touching the imported one."""
    source = taichi_loader_module.__file__
    assert source is not None
    spec = importlib.util.spec_from_file_location("memory_machines.kernels.taichi._fresh", source)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- cpu_loader: extension resolution -------------------------------------


def test_get_cpu_extension_returns_none_when_native_dir_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _point_at_native_dir(monkeypatch, tmp_path / "absent")
    calls: list[dict[str, Any]] = []
    _install_fake_load(monkeypatch, object(), calls)

    assert cpu_loader_module.get_cpu_extension() is None
    assert calls == []


def test_get_cpu_extension_returns_none_when_native_dir_has_no_cpp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    native_dir = tmp_path / "cpu"
    native_dir.mkdir()
    (native_dir / "notes.txt").write_text("not a source", encoding="utf-8")
    _point_at_native_dir(monkeypatch, native_dir)
    calls: list[dict[str, Any]] = []
    _install_fake_load(monkeypatch, object(), calls)

    assert cpu_loader_module.get_cpu_extension() is None
    assert calls == []


def test_get_cpu_extension_compiles_every_cpp_source_in_sorted_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    native_dir = tmp_path / "cpu"
    native_dir.mkdir()
    for name in ("zeta.cpp", "alpha.cpp"):
        (native_dir / name).write_text("// kernel", encoding="utf-8")
    (native_dir / "README.md").write_text("ignored", encoding="utf-8")
    _point_at_native_dir(monkeypatch, native_dir)
    calls: list[dict[str, Any]] = []
    sentinel = object()
    _install_fake_load(monkeypatch, sentinel, calls)

    assert cpu_loader_module.get_cpu_extension() is sentinel
    assert len(calls) == 1
    assert calls[0]["name"] == "memory_machines_cpu_kernels"
    assert calls[0]["verbose"] is False
    assert calls[0]["sources"] == sorted([str(native_dir / "alpha.cpp"), str(native_dir / "zeta.cpp")])


def test_get_cpu_extension_is_cached_after_first_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    native_dir = tmp_path / "cpu"
    native_dir.mkdir()
    source = native_dir / "alpha.cpp"
    source.write_text("// kernel", encoding="utf-8")
    _point_at_native_dir(monkeypatch, native_dir)
    calls: list[dict[str, Any]] = []
    sentinel = object()
    _install_fake_load(monkeypatch, sentinel, calls)

    assert cpu_loader_module.get_cpu_extension() is sentinel
    source.unlink()
    assert cpu_loader_module.get_cpu_extension() is sentinel
    assert len(calls) == 1


def test_get_cpu_extension_propagates_build_failure_without_caching_it(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    native_dir = tmp_path / "cpu"
    native_dir.mkdir()
    (native_dir / "alpha.cpp").write_text("// kernel", encoding="utf-8")
    _point_at_native_dir(monkeypatch, native_dir)
    calls: list[dict[str, Any]] = []
    _install_fake_load(monkeypatch, RuntimeError("ninja: build failed"), calls)

    with pytest.raises(RuntimeError, match="build failed"):
        cpu_loader_module.get_cpu_extension()

    sentinel = object()
    _install_fake_load(monkeypatch, sentinel, calls)
    assert cpu_loader_module.get_cpu_extension() is sentinel


# --- cpu_loader: kernel symbol lookup -------------------------------------


def test_get_cpu_kernel_returns_none_without_extension(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    native_dir = tmp_path / "cpu"
    native_dir.mkdir()
    _point_at_native_dir(monkeypatch, native_dir)

    assert cpu_loader_module.get_cpu_kernel("softmax") is None
    assert cpu_loader_module.get_cpu_kernel("softmax") is None


def test_get_cpu_kernel_prefers_forward_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    forward = lambda *a: "forward"
    bare = lambda *a: "bare"
    monkeypatch.setattr(cpu_loader_module, "_ext", types.SimpleNamespace(softmax_forward=forward, softmax=bare))

    assert cpu_loader_module.get_cpu_kernel("softmax") is forward


def test_get_cpu_kernel_accepts_bare_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    bare = lambda *a: "bare"
    monkeypatch.setattr(cpu_loader_module, "_ext", types.SimpleNamespace(softmax=bare))

    assert cpu_loader_module.get_cpu_kernel("softmax") is bare


def test_get_cpu_kernel_skips_none_forward_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    bare = lambda *a: "bare"
    monkeypatch.setattr(cpu_loader_module, "_ext", types.SimpleNamespace(softmax_forward=None, softmax=bare))

    assert cpu_loader_module.get_cpu_kernel("softmax") is bare


def test_get_cpu_kernel_returns_none_for_unknown_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cpu_loader_module, "_ext", types.SimpleNamespace(other_forward=lambda: None))

    assert cpu_loader_module.get_cpu_kernel("softmax") is None


def test_get_cpu_kernel_is_stable_across_calls_and_builds_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    native_dir = tmp_path / "cpu"
    native_dir.mkdir()
    (native_dir / "alpha.cpp").write_text("// kernel", encoding="utf-8")
    _point_at_native_dir(monkeypatch, native_dir)
    calls: list[dict[str, Any]] = []
    compiled = types.SimpleNamespace(softmax_forward=lambda *a: "compiled")
    _install_fake_load(monkeypatch, compiled, calls)

    first = cpu_loader_module.get_cpu_kernel("softmax")
    second = cpu_loader_module.get_cpu_kernel("softmax")

    assert first is second is compiled.softmax_forward
    assert len(calls) == 1


# --- taichi loader: availability and lazy initialization ------------------


def test_taichi_loader_module_imports_without_taichi_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "taichi", None)

    fresh = _load_fresh_taichi_loader()

    assert fresh.get_taichi_kernel("no_such_kernel_module") is None


def test_import_taichi_raises_clear_error_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(taichi_loader_module, "_ti", None)
    monkeypatch.setitem(sys.modules, "taichi", None)

    with pytest.raises(ImportError, match="taichi is not installed"):
        taichi_loader_module._import_taichi()


def test_import_taichi_caches_the_imported_module(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_ti = types.ModuleType("taichi")
    monkeypatch.setattr(taichi_loader_module, "_ti", None)
    monkeypatch.setitem(sys.modules, "taichi", fake_ti)

    assert taichi_loader_module._import_taichi() is fake_ti

    monkeypatch.delitem(sys.modules, "taichi")
    assert taichi_loader_module._import_taichi() is fake_ti


def test_get_taichi_kernel_returns_none_for_missing_module(monkeypatch: pytest.MonkeyPatch) -> None:
    # The loader must not need taichi at all to answer "no such kernel".
    monkeypatch.setattr(taichi_loader_module, "_import_taichi", lambda: pytest.fail("get_taichi_kernel must not import taichi"))

    assert taichi_loader_module.get_taichi_kernel("definitely_not_a_kernel_module") is None


def test_get_taichi_kernel_returns_documented_kernel_attribute(monkeypatch: pytest.MonkeyPatch) -> None:
    kernel = lambda x: ("taichi", x)
    fake_module = types.ModuleType("memory_machines.kernels.taichi.fake_op")
    fake_module.kernel = kernel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "memory_machines.kernels.taichi.fake_op", fake_module)

    assert taichi_loader_module.get_taichi_kernel("fake_op") is kernel


def test_get_taichi_kernel_returns_none_when_module_lacks_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = types.ModuleType("memory_machines.kernels.taichi.fake_op")
    fake_module.not_a_kernel = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "memory_machines.kernels.taichi.fake_op", fake_module)

    assert taichi_loader_module.get_taichi_kernel("fake_op") is None


def test_get_taichi_kernel_returns_none_when_module_import_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_import(name: str, package: str | None = None) -> types.ModuleType:
        raise ImportError(f"broken kernel module: {name}")

    monkeypatch.setattr(taichi_loader_module.importlib, "import_module", broken_import)

    assert taichi_loader_module.get_taichi_kernel("fake_op") is None


def test_ensure_taichi_raises_when_taichi_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(taichi_loader_module, "_initialized", False)
    monkeypatch.setattr(taichi_loader_module, "_ti", None)
    monkeypatch.setitem(sys.modules, "taichi", None)

    with pytest.raises(ImportError, match="taichi is not installed"):
        taichi_loader_module.ensure_taichi()


def test_ensure_taichi_recovers_after_a_missing_taichi_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(taichi_loader_module, "_initialized", False)
    monkeypatch.setattr(taichi_loader_module, "_ti", None)
    monkeypatch.setitem(sys.modules, "taichi", None)
    with pytest.raises(ImportError):
        taichi_loader_module.ensure_taichi()

    fake_ti = _FakeTi()
    monkeypatch.setattr(taichi_loader_module, "_import_taichi", lambda: fake_ti)
    taichi_loader_module.ensure_taichi()

    assert fake_ti.init_calls == ["ARCH_CPU"]


@pytest.mark.parametrize(
    ("tarch", "cuda_available", "expected_arch"),
    [
        (None, False, "ARCH_CPU"),
        (None, True, "ARCH_CUDA"),
        ("cpu", True, "ARCH_CPU"),
        (" cuda ", False, "ARCH_CUDA"),
        ("vulkan", False, "ARCH_VULKAN"),
    ],
    ids=["default-cpu", "default-cuda", "explicit-cpu-wins", "whitespace-stripped", "vulkan-backend"],
)
def test_ensure_taichi_selects_arch(monkeypatch: pytest.MonkeyPatch, tarch: str | None, cuda_available: bool, expected_arch: str) -> None:
    fake_ti = _FakeTi()
    monkeypatch.setattr(taichi_loader_module, "_initialized", False)
    monkeypatch.setattr(taichi_loader_module, "_import_taichi", lambda: fake_ti)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    if tarch is None:
        monkeypatch.delenv("TARCH", raising=False)
    else:
        monkeypatch.setenv("TARCH", tarch)

    taichi_loader_module.ensure_taichi()

    assert fake_ti.init_calls == [expected_arch]


def test_ensure_taichi_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_ti = _FakeTi()
    monkeypatch.setattr(taichi_loader_module, "_initialized", False)
    monkeypatch.setattr(taichi_loader_module, "_import_taichi", lambda: fake_ti)
    monkeypatch.delenv("TARCH", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    taichi_loader_module.ensure_taichi()
    taichi_loader_module.ensure_taichi()

    assert fake_ti.init_calls == ["ARCH_CPU"]


def test_ensure_taichi_rejects_unknown_tarch(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_ti = _FakeTi()
    monkeypatch.setattr(taichi_loader_module, "_initialized", False)
    monkeypatch.setattr(taichi_loader_module, "_import_taichi", lambda: fake_ti)
    monkeypatch.setenv("TARCH", "bogus_backend")

    with pytest.raises(AttributeError):
        taichi_loader_module.ensure_taichi()
    assert fake_ti.init_calls == []


# --- registry: the fallback the loaders feed ------------------------------


def test_dispatch_auto_uses_compiled_kernel_without_warning() -> None:
    registry = OpRegistry()
    registry.register(OpSpec(name="demo_op", naive=lambda x: ("naive", x), compiled_cpu=lambda: lambda x: ("cpu", x)))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert registry.dispatch("demo_op", "auto", 7) == ("cpu", 7)


def test_dispatch_auto_falls_back_when_cpu_loader_returns_none() -> None:
    registry = OpRegistry()
    registry.register(OpSpec(name="demo_op", naive=lambda x: ("naive", x), compiled_cpu=lambda: None))

    with pytest.warns(UserWarning, match=r"demo_op: compiled kernel unavailable .*CPU cpp: loader returned None"):
        assert registry.dispatch("demo_op", "auto", 7) == ("naive", 7)


def test_dispatch_cpu_backend_raises_when_no_kernel_is_registered() -> None:
    registry = OpRegistry()
    registry.register(OpSpec(name="demo_op", naive=lambda x: ("naive", x)))

    with pytest.raises(NotImplementedError, match="no CPU cpp kernel registered"):
        registry.dispatch("demo_op", "cpu", 7)


def test_dispatch_auto_with_real_loaders_warns_and_uses_reference(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    native_dir = tmp_path / "cpu"
    native_dir.mkdir()
    _point_at_native_dir(monkeypatch, native_dir)
    monkeypatch.setattr(taichi_loader_module, "_ti", None)
    monkeypatch.setattr(taichi_loader_module, "_initialized", False)
    monkeypatch.setitem(sys.modules, "taichi", None)

    def load_taichi_kernel() -> Any:
        # Real kernel modules call ensure_taichi() during their own import, so
        # this is the error shape the dispatcher actually sees on a host
        # without taichi.
        taichi_loader_module.ensure_taichi()
        return None

    registry = OpRegistry()
    registry.register(
        OpSpec(
            name="demo_op",
            naive=lambda x: ("naive", x),
            compiled_cpu=make_cpu_loader("demo_op"),
            compiled_taichi=load_taichi_kernel,
        )
    )

    with pytest.warns(UserWarning) as record:
        result = registry.dispatch("demo_op", "auto", 7)

    message = str(record[0].message)
    assert result == ("naive", 7)
    assert "Taichi: taichi is not installed" in message
    assert "CPU cpp: loader returned None" in message
