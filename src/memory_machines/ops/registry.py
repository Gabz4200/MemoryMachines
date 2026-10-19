"""Central op registry with a generic compiled-first dispatcher.

Design
------
Every op in the project is registered here once with an ``OpSpec`` that
captures:

* ``naive``   - the pure-PyTorch reference implementation (always works;
  this is the ``backend="naive"`` path and the fallback target for ``auto``)
* ``torch_op`` - optional ``torch.library.custom_op`` facade that is kept in
  sync with the compiled kernels
* ``compiled_cpu`` / ``compiled_taichi`` - optional lazy loaders that return
  a compiled kernel callable. Each loader returns ``None`` until the actual
  kernel lands, so dispatch falls back to the naive path without raising.

The backend logic lives here exactly once; the per-op wrappers in
``memory_machines.ops.dispatch`` are thin, typed adapters around
``registry.dispatch``. Adding a new op therefore means registering one
``OpSpec`` here plus its reference implementation — no duplicated backend
branching per op.

Backend resolution (``dispatch(name, backend, *args, **kwargs)``):
  * ``naive``            -> call ``spec.naive`` directly, no warning
  * ``auto``             -> try compiled loaders (taichi then cpu) in order;
                            on failure emit a single UserWarning naming the op
                            and fall back to ``naive``
  * ``taichi`` / ``cpu`` -> try only that compiled loader; raise a clear
                            ``NotImplementedError`` / ``RuntimeError`` if it
                            is not yet implemented
  * anything else        -> ``ValueError``

Compiled kernels (Taichi / CPU cpp) are intentionally *not* implemented yet
for every op. Until they land, ``auto`` always falls back to the pure-PyTorch
reference, so callers always get a working result.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Optional


class OpSpec:
    """Declarative description of one op's implementation surface."""

    def __init__(
        self,
        *,
        name: str,
        naive: Callable[..., Any],
        torch_op: Optional[Callable[..., Any]] = None,
        compiled_cpu: Optional[Callable[[], Optional[Callable]]] = None,
        compiled_taichi: Optional[Callable[[], Optional[Callable]]] = None,
        description: str = "",
    ):
        if not name:
            raise ValueError("OpSpec.name must be non-empty")
        if naive is None:
            raise ValueError(f"op {name!r}: naive implementation required")
        self.name = name
        self.naive = naive
        self.torch_op = torch_op
        self.compiled_cpu = compiled_cpu
        self.compiled_taichi = compiled_taichi
        self.description = description


_VALID_BACKENDS = ("naive", "auto", "cpu", "gpu", "taichi", "compiled")


def _normalize_backend(backend: Any) -> str:
    if not isinstance(backend, str):
        raise TypeError(f"backend must be a str, got {type(backend).__name__}")
    norm = backend.strip().lower()
    if norm not in _VALID_BACKENDS:
        raise ValueError(f"Unknown backend: {backend!r}. Valid: {sorted(_VALID_BACKENDS)}")
    return norm


def _emit_downgrade(op: str, reason: str) -> None:
    warnings.warn(
        f"{op}: compiled kernel unavailable ({reason}); falling back to the pure-PyTorch reference. To silence this warning, pass backend='naive' explicitly.",
        UserWarning,
        stacklevel=2,
    )


def _try_load_and_call(loader: Optional[Callable[[], Optional[Callable]]], *args: Any, **kwargs: Any) -> tuple[Optional[Any], Optional[BaseException]]:
    """Resolve ``loader`` to a kernel callable and call it.

    Returns ``(result, None)`` on success, ``(None, exc)`` if the loader is
    missing or raises (kernel not implemented / not installed / load failure).
    """
    if loader is None:
        return None, NotImplementedError(f"no compiled kernel registered")
    try:
        kernel = loader()
        if kernel is None:
            return None, NotImplementedError("loader returned None")
        return kernel(*args, **kwargs), None
    except Exception as exc:  # kernel not implemented / not installed
        return None, exc


class OpRegistry:
    """Registry of op specs + the generic backend dispatcher."""

    def __init__(self) -> None:
        self._specs: dict[str, OpSpec] = {}

    def register(self, spec: OpSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"op already registered: {spec.name!r}")
        self._specs[spec.name] = spec

    def get(self, name: str) -> OpSpec:
        try:
            return self._specs[name]
        except KeyError:
            raise KeyError(f"unknown op {name!r}; registered: {sorted(self._specs)}") from None

    def names(self) -> list[str]:
        return sorted(self._specs)

    def dispatch(self, name: str, backend: str = "auto", *args: Any, **kwargs: Any) -> Any:
        """Dispatch op ``name`` through ``backend``.

        See module docstring for the resolution rules.
        """
        spec = self.get(name)
        backend = _normalize_backend(backend)
        if backend == "naive":
            return spec.naive(*args, **kwargs)
        if backend == "auto":
            return self._dispatch_auto(spec, *args, **kwargs)
        if backend in ("gpu", "taichi"):
            return self._dispatch_taichi(spec, *args, **kwargs)
        if backend in ("cpu", "compiled"):
            return self._dispatch_cpu(spec, *args, **kwargs)
        raise AssertionError(f"unreachable backend {backend!r}")  # pragma: no cover

    def _dispatch_auto(self, spec: OpSpec, *args: Any, **kwargs: Any) -> Any:
        reasons: list[str] = []
        if spec.compiled_taichi is not None:
            result, exc = _try_load_and_call(spec.compiled_taichi, *args, **kwargs)
            if exc is None:
                return result
            reasons.append(f"Taichi: {exc}")
        if spec.compiled_cpu is not None:
            result, exc = _try_load_and_call(spec.compiled_cpu, *args, **kwargs)
            if exc is None:
                return result
            reasons.append(f"CPU cpp: {exc}")
        reason = "; ".join(reasons) if reasons else "no compiled kernel registered"
        _emit_downgrade(spec.name, reason)
        return spec.naive(*args, **kwargs)

    def _dispatch_taichi(self, spec: OpSpec, *args: Any, **kwargs: Any) -> Any:
        if spec.compiled_taichi is None:
            raise NotImplementedError(f"{spec.name}: compiled backend 'taichi' unavailable: no Taichi kernel registered; use backend='auto' (warns + falls back to reference) or backend='naive'.")
        result, exc = _try_load_and_call(spec.compiled_taichi, *args, **kwargs)
        if exc is None:
            return result
        raise NotImplementedError(f"{spec.name}: compiled backend 'taichi' unavailable: {exc}; use backend='auto' (warns + falls back to reference) or backend='naive'.")

    def _dispatch_cpu(self, spec: OpSpec, *args: Any, **kwargs: Any) -> Any:
        if spec.compiled_cpu is None:
            raise NotImplementedError(f"{spec.name}: compiled backend 'cpu' unavailable: no CPU cpp kernel registered; use backend='auto' (warns + falls back to reference) or backend='naive'.")
        result, exc = _try_load_and_call(spec.compiled_cpu, *args, **kwargs)
        if exc is None:
            return result
        raise NotImplementedError(f"{spec.name}: compiled backend 'cpu' unavailable: {exc}; use backend='auto' (warns + falls back to reference) or backend='naive'.")


_REGISTRY = OpRegistry()


def register_op(spec: OpSpec) -> OpSpec:
    """Register an ``OpSpec`` on the global registry; returns ``spec``."""
    _REGISTRY.register(spec)
    return spec


def get_op(name: str) -> OpSpec:
    return _REGISTRY.get(name)


def list_ops() -> list[str]:
    return _REGISTRY.names()


def dispatch_op(name: str, backend: str = "auto", *args: Any, **kwargs: Any) -> Any:
    """Generic entry point: dispatch op ``name`` through ``backend``."""
    return _REGISTRY.dispatch(name, backend, *args, **kwargs)


__all__ = [
    "OpSpec",
    "OpRegistry",
    "register_op",
    "get_op",
    "list_ops",
    "dispatch_op",
]
