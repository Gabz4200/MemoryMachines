r"""Rank budget: name a projection's role and decide its bottleneck width.

:class:`RankPolicy` maps a role (Key, Value, FFN gate, GDN-2 decay, ...) to a
fraction of the smaller dimension, and :func:`apply_low_rank` walks a model once
at the composition root replacing the projections that benefit. The role table
is here because it is the one place a new projection must be declared; a
projection whose path is not in the table raises instead of silently keeping a
dense budget.
"""

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from torch import nn

from .linear_init import factorize_linear
from .linears import FactoredLinear

__all__ = ["RANK_ROLES", "RankConfig", "RankPolicy", "apply_low_rank", "linear_role"]


LinearLike = nn.Linear | FactoredLinear

#: Roles that :func:`linear_role` can return. Every role is a knob in
#: :attr:`~memory_machines.utils.config_schema.MemoryMachineConfig.rank_ratios`.
RANK_ROLES = frozenset(
    {
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_o",
        "ffn_gate",
        "ffn_up",
        "ffn_down",
        "csa_kv",
        "csa_q",
        "csa_v",
        "csa_q_latent",
        "csa_index_q",
        "csa_index_k",
        "csa_group",
        "csa_out",
        "gdn2_q",
        "gdn2_k",
        "gdn2_v",
        "gdn2_o",
        "gdn2_gate",
        "gdn2_decay",
        "titan_q",
        "titan_k",
        "titan_v",
        "titan_o",
        "titan_gate",
        "engram",
        "blt_patch",
        "blt_latent",
        "ced_mix",
        "dspark",
    }
)

_ATTENTION_TAGS = (".attn.", ".attention.", ".swa.", ".cross.")
_TITAN_TAGS = (".memory.", ".titan", ".hope")

#: (role, ratio) pairs already reported as unprofitable, so a small test config
#: does not repeat one warning per model build.
_WARNED_UNPROFITABLE: set[tuple[str, float]] = set()
#: Role of a projection whose leaf name alone decides it. ``None`` means the
#: role is a dense invariant (readout, width adapter, identity-init gate,
#: single-scalar gate), so no budget applies.
_LEAF_ROLES: dict[str, str | None] = {
    "head": None,
    "output": None,
    "decoder_in_proj": None,
    # Already rank-bounded by construction: a scalar/small-N output (CVPE
    # gamma/beta/offset, F-RVM spatial OV) or a single-vector input (its MLP
    # input layer has fan-in 1 or N).
    "proj": None,
    "gamma_proj": None,
    "beta_proj": None,
    "mlp_in": None,
    "mlp_in_nd": None,
    "offset_out": None,
    "offset_out_nd": None,
    # Neutral-start gates and selector heads: out width is 1 or num_heads.
    "score_proj": None,
    "weight_proj": None,
    # Identity-initialized short-conv adapters (LivConv1d) and the LIV block's
    # projections; their init is the identity, which has no low-rank peer.
    "input_projection": None,
    "output_projection": None,
    "trm_mix": "ced_mix",
    "z_proj": "ced_mix",
    "kv_proj": "csa_kv",
    "q_down": "csa_q_latent",
    "q_up_main": "csa_q",
    "q_up_index": "csa_index_q",
    "k_index_proj": "csa_index_k",
    "patch_init_proj": "blt_patch",
    "patch_out_proj": "blt_patch",
    "patch_proj": "blt_patch",
    "hidden_proj": "blt_latent",
    "stats_proj": "blt_latent",
    "draft_proj": "dspark",
    "value_proj": "engram",
    "fuse": "titan_gate",
    "w_proj": "gdn2_decay",
    "b_proj": "gdn2_gate",
    "out_proj": "attn_o",
}

#: Path fragments that outrank the leaf name, checked in order.
_PATH_ROLES: tuple[tuple[str, str | None], ...] = (
    ("up.proj", None),
    (".grouped_out.group_projs.", "csa_group"),
    (".grouped_out.out_proj", "csa_out"),
    (".key_projs.", "engram"),
    (".f_proj.", "gdn2_gate"),
    (".g_proj.", "gdn2_gate"),
)

#: Leaves shared by every token mixer; the path decides which family they serve.
_MIXER_STEMS = frozenset({"q_proj", "k_proj", "v_proj", "o_proj"})

#: FFN up/down leaves; only the ``.ffn`` subtree uses them.
_FFN_OUT_STEMS = frozenset({"up_proj", "down_proj"})


def linear_role(name: str) -> str | None:
    r"""linear_role(name) -> str | None

    Classify a projection by its qualified module path.

    The role names the capacity budget the projection gets, so the policy can
    treat, for example, a Key projection (rank-deficient in trained LLMs) apart
    from a Value projection (dense spectrum). Projections that must stay dense
    because factoring them is meaningless or wrong -- the vocabulary readout,
    width adapters, identity-initialized gates, and single-scalar gates --
    return ``None``.

    Args:
        name (str): dotted module path from ``named_modules()``, for example
            ``"encoder_layers.0.attention.inner.q_proj"``.

    Returns:
        str | None: a member of :data:`RANK_ROLES`, or ``None`` to keep the
        projection dense.

    Examples::

        >>> linear_role("encoder_layers.0.attention.inner.k_proj")
        'attn_k'
        >>> linear_role("decoder.layers.0.q_up_main")
        'csa_q'
        >>> linear_role("head") is None
        True
    """
    leaf = name.rsplit(".", 1)[-1]
    for fragment, role in _PATH_ROLES:
        if fragment in name:
            return role
    if leaf in _LEAF_ROLES:
        return _LEAF_ROLES[leaf]
    if leaf == "gate_proj":
        return "ffn_gate" if ".ffn" in name else "titan_gate"
    if leaf in _FFN_OUT_STEMS:
        if ".ffn" not in name:
            return None
        return "ffn_up" if leaf == "up_proj" else "ffn_down"
    if leaf not in _MIXER_STEMS:
        raise ValueError(
            f"no rank role for projection {name!r}: add its leaf to _LEAF_ROLES (``None`` for a dense invariant) "
            "or its path fragment to _PATH_ROLES, otherwise a new projection would silently keep a dense budget"
        )
    if any(tag in name for tag in _TITAN_TAGS):
        return f"titan_{leaf[0]}"
    if ".gdn2." in name:
        return f"gdn2_{leaf[0]}"
    if leaf == "v_proj" and not any(tag in name for tag in _ATTENTION_TAGS):
        return "csa_v"
    return f"attn_{leaf[0]}"


class RankConfig(Protocol):
    r"""RankConfig

    The two configuration fields :meth:`RankPolicy.from_config` reads. Any
    object providing them works, which keeps the block layer free of an import
    of the config module.
    """

    low_rank_mode: str
    rank_ratios: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class RankPolicy:
    r"""RankPolicy(mode='factored', ratios={})

    Per-role rank budget: a role with ratio ``r`` gets
    ``int(min(in_features, out_features) * r)`` bottleneck width, truncated to
    an integer. The ratio is relative to the smaller width, so a
    ``4096 -> 128`` projection can never ask for a rank above ``128``.

    A ratio of ``1.0`` keeps the projection dense, and so does any ratio whose
    rank would not actually save parameters: a factored ``in x out`` projection
    costs ``rank * (in + out)`` against ``in * out``, so the budget only applies
    when ``rank < in * out / (in + out)``. For a square ``d x d`` matrix that
    means ``r < 0.5``; asking for 85% of the dimension would cost 1.7x the dense
    parameter count, so the budget is rejected instead of silently paid for.
    Value, output, and feed-forward up/down projections have a dense (isotropic)
    spectrum, so keeping them dense is the intended setting.

    Args:
        mode (str, optional): ``"factored"`` applies the budget, ``"dense"``
            disables it everywhere. Default: ``"factored"``.
        ratios (Mapping[str, float], optional): role -> fraction of
            ``min(in_features, out_features)``, in ``(0, 1]``. Default: empty.

    Raises:
        ValueError: on an unknown mode, an unknown role, or a ratio outside
            ``(0, 1]``.
    """

    mode: str = "factored"
    ratios: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in ("dense", "factored"):
            raise ValueError(f"mode ({self.mode!r}) must be 'dense' or 'factored'")
        unknown = set(self.ratios) - RANK_ROLES
        if unknown:
            raise ValueError(f"unknown rank role(s) {sorted(unknown)}; known roles: {sorted(RANK_ROLES)}")
        bad = {role: r for role, r in self.ratios.items() if not 0.0 < r <= 1.0}
        if bad:
            raise ValueError(f"rank ratios must be in (0, 1], got {bad}")

    @classmethod
    def from_config(cls, config: "RankConfig") -> "RankPolicy":
        r"""from_config(config) -> RankPolicy

        Build a policy from ``config.low_rank_mode`` and
        ``config.rank_ratios`` (a sequence of ``(role, ratio)`` pairs).

        Args:
            config (RankConfig): any object exposing those two attributes;
                typically
                :class:`~memory_machines.utils.config_schema.MemoryMachineConfig`.

        Returns:
            RankPolicy: the resolved policy.
        """
        return cls(mode=str(config.low_rank_mode), ratios=dict(config.rank_ratios))

    def rank_for(self, in_features: int, out_features: int, role: str | None) -> int | None:
        r"""rank_for(in_features, out_features, role) -> int | None

        Bottleneck width for one projection.

        Args:
            in_features (int): projection input width.
            out_features (int): projection output width.
            role (str | None): role from :func:`linear_role`; ``None`` means
                the projection is a dense invariant.

        Returns:
            int | None: the rank to factor at, or ``None`` to keep it dense.
        """
        if self.mode != "factored" or role is None:
            return None
        ratio = self.ratios.get(role, 1.0)
        if ratio >= 1.0:
            return None
        cap = min(in_features, out_features)
        rank = max(1, int(cap * ratio))
        if rank >= cap or rank * (in_features + out_features) >= in_features * out_features:
            return None
        return rank


def apply_low_rank(module: nn.Module, policy: RankPolicy) -> int:
    r"""apply_low_rank(module, policy) -> int

    Replace every eligible projection under ``module`` with a rank-budgeted
    :class:`FactoredLinear`, in place.

    Walks the full module tree, so it is the single composition-root hook: call
    it once on a model after its layers are built and initialized, and every
    nested attention, FFN, memory, and local-model projection is covered.
    Already-factored projections are skipped, so the call is idempotent.

    Args:
        module (nn.Module): model or block to rewrite.
        policy (RankPolicy): role -> rank budget.

    Returns:
        int: number of projections replaced.

    Examples::

        >>> import torch
        >>> from torch import nn
        >>> net = nn.ModuleDict({"q_proj": nn.Linear(8, 8)})
        >>> apply_low_rank(net, RankPolicy(ratios={"attn_q": 0.25}))
        1
        >>> isinstance(net["q_proj"], FactoredLinear)
        True
    """
    if policy.mode != "factored":
        return 0
    plan: list[tuple[str, nn.Linear, int]] = []
    unprofitable: dict[str, float] = {}
    # A factored projection stores its factors as nn.Linear, so skip them:
    # factorizing a factor would nest the budget and break idempotency.
    factor_children = {f"{name}.{leaf}" for name, child in module.named_modules() if isinstance(child, FactoredLinear) for leaf in ("in_linear", "out_linear")}
    for name, child in module.named_modules():
        if not isinstance(child, nn.Linear) or name in factor_children:
            continue
        role = linear_role(name)
        rank = policy.rank_for(child.in_features, child.out_features, role)
        if rank is not None:
            plan.append((name, child, rank))
            continue
        if role is not None and 0.0 < policy.ratios.get(role, 1.0) < 1.0:
            unprofitable[role] = policy.ratios[role]
    if unprofitable:
        # Staying silent would look like the budget was applied. Each role is
        # reported once per process: a small test config legitimately cannot
        # afford a ratio that the real width can.
        fresh = {role: ratio for role, ratio in unprofitable.items() if (role, ratio) not in _WARNED_UNPROFITABLE}
        _WARNED_UNPROFITABLE.update(fresh.items())
        if fresh:
            asked = ", ".join(f"{role!r} ({ratio:g})" for role, ratio in sorted(fresh.items()))
            warnings.warn(
                f"rank budget not applied for {asked}: a factored in x out projection costs rank * (in + out) parameters, "
                "so the requested ratio is not cheaper than dense here; lower it below min(in, out) / (in + out) "
                "or set it to 1.0 to say 'dense' deliberately",
                stacklevel=2,
            )
    for name, child, rank in plan:
        parent_path, _, leaf = name.rpartition(".")
        parent = module.get_submodule(parent_path) if parent_path else module
        setattr(parent, leaf, factorize_linear(child, rank, preserve_columns=leaf == "out_proj"))
    return len(plan)
