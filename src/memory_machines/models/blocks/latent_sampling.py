r"""Reparameterized Gaussian sampling for the BLT decoder boundary.

The sampler borrows the VAE reparameterization trick without a variational
objective: a small head predicts a per-patch mean and standard deviation, a
standard-normal draw ``eps ~ N(0, 1)`` is scaled by the temperature and the
predicted std, and byte cross-entropy backpropagates through both predictions.
``sigma`` acts as a per-patch confidence knob rather than a variational
posterior, so there is no KL to a zero-mean prior.
"""

from dataclasses import dataclass

import torch
from torch import nn

from memory_machines.utils.config_schema import MemoryMachineConfig


@dataclass
class LatentSample:
    """One reparameterized draw plus the statistics used to produce it.

    Attributes:
        latent: Sampled latent states, same shape as the hidden states.
        mean: Predicted mean.
        std: Predicted standard deviation (positive).
        regularization: Variance-only KL to ``N(mean, target_std**2)``; a
            scalar that anchors the std and prevents collapse.
    """

    latent: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    regularization: torch.Tensor


class GaussianLatentSampler(nn.Module):
    r"""GaussianLatentSampler(config)

    Predict per-patch Gaussian latents and draw one reparameterized sample.

    A two-layer MLP maps each hidden state to a residual mean and a log
    standard deviation. The mean head starts at zero (``mean == hidden``) and
    the log-std head starts at zero (``std == blt_latent_target_std``), so the
    sampler is initially a small isotropic perturbation of the decoder state.

    The draw is ``z = mu + (eps * temperature) * sigma`` with
    ``eps ~ N(0, 1)``: temperature scales the noise range without changing the
    relative confidence predicted by the std head. Training uses ``temperature
    = 1`` (``z = mu + eps * sigma``); ``temperature = 0`` returns the mean.

    Cross-entropy alone drives ``std`` to zero: adding independent noise to
    logits can only increase the expected loss (cross-entropy is convex in the
    logits), and ``std = 0`` is a local minimum. The returned
    :attr:`LatentSample.regularization` is therefore a variance-only KL term
    against a same-mean Gaussian with the target std,

    .. math::
        D_{KL}(\mathcal{N}(\mu, \sigma^2) \,\|\, \mathcal{N}(\mu, t^2))
        = \log \frac{t}{\sigma} + \frac{\sigma^2}{2 t^2} - \frac{1}{2},

    which leaves the mean untouched and has its minimum at ``sigma == t``.

    Args:
        config (:class:`~memory_machines.utils.config_schema.MemoryMachineConfig`):
            supplies the hidden width and the ``blt_latent_*`` hyperparameters.

    Examples::

        >>> cfg = MemoryMachineConfig(hidden_size=16, vocab_size=32,
        ...     num_attention_heads=2, num_key_value_heads=1,
        ...     ced_num_encoder_layers=1, ced_num_decoder_layers=1,
        ...     csa_top_k=2, csa_num_output_groups=2,
        ...     blt_local_hidden_size=16, blt_local_layers=1,
        ...     blt_local_window_size=8, blt_latent_sampling=True)
        >>> sampler = GaussianLatentSampler(cfg)
        >>> h = torch.zeros(1, 2, 16)
        >>> sampler(h, noise=torch.zeros_like(h)).latent.shape
        torch.Size([1, 2, 16])
    """

    def __init__(self, config: MemoryMachineConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.target_std = config.blt_latent_target_std
        self.min_std = config.blt_latent_min_std
        self.default_temperature = config.blt_latent_temperature
        self.initializer_range = config.initializer_range
        self.hidden_proj = nn.Linear(self.hidden_size, 2 * self.hidden_size)
        self.activation = nn.SiLU()
        self.stats_proj = nn.Linear(2 * self.hidden_size, 2 * self.hidden_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Restore the identity-mean and target-std initialization."""
        std = self.initializer_range
        nn.init.trunc_normal_(self.hidden_proj.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
        nn.init.zeros_(self.hidden_proj.bias)
        nn.init.zeros_(self.stats_proj.weight)
        nn.init.zeros_(self.stats_proj.bias)

    def forward(self, hidden_states: torch.Tensor, *, temperature: float | None = None, noise: torch.Tensor | None = None) -> LatentSample:
        r"""Draw one Gaussian latent per patch.

        Args:
            hidden_states: Decoder patch states, shape ``(..., hidden_size)``.
            temperature: Scale factor on the standard-normal draw; ``0``
                returns the predicted mean. Default: ``None`` (the configured
                ``blt_latent_temperature``).
            noise: Optional standard-normal draw with the same shape as
                ``hidden_states``. Default: ``None`` (sample with
                :func:`torch.randn_like`).

        Returns:
            A :class:`LatentSample` with ``latent``, ``mean``, ``std``, and the
            scalar variance regularizer.

        Raises:
            ValueError: If ``temperature`` is negative or ``noise`` does not
                match the hidden-state shape.
        """
        if temperature is None:
            temperature = self.default_temperature
        if temperature < 0:
            raise ValueError(f"temperature ({temperature}) must be non-negative")
        stats = self.stats_proj(self.activation(self.hidden_proj(hidden_states)))
        delta, raw_log_std = stats[..., : self.hidden_size], stats[..., self.hidden_size :]
        mean = hidden_states + delta
        # Clamp the log-std so exp cannot overflow on long runs; the floor keeps
        # the variance KL finite.
        std = (self.target_std * torch.exp(raw_log_std.clamp(-8.0, 8.0))).clamp_min(self.min_std)
        if noise is None:
            noise = torch.randn_like(mean)
        elif noise.shape != mean.shape:
            raise ValueError(f"noise shape {tuple(noise.shape)} must match hidden state shape {tuple(mean.shape)}")
        latent = mean + (noise * temperature) * std
        return LatentSample(latent=latent, mean=mean, std=std, regularization=self._variance_kl(std))

    def _variance_kl(self, std: torch.Tensor) -> torch.Tensor:
        """Return the mean KL between ``N(mu, std**2)`` and ``N(mu, target**2)``."""
        ratio = std.float() / self.target_std
        return (-torch.log(ratio) + 0.5 * ratio.square() - 0.5).mean()
