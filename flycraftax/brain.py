"""Shiu et al. 2024 LIF on a sparse connectome, in JAX."""
import math
from typing import NamedTuple


class BrainParams(NamedTuple):
    dt_ms: float = 0.1
    v_rest: float = -52.0
    v_reset: float = -52.0
    v_th: float = -45.0
    t_mbr: float = 20.0
    tau: float = 5.0
    t_rfc: float = 2.2
    t_dly: float = 1.8
    w_syn: float = 0.275
    kick: float = 0.275 * 250

    @property
    def n_dly(self) -> int:
        return round(self.t_dly / self.dt_ms)

    @property
    def n_rfc(self) -> int:
        return round(self.t_rfc / self.dt_ms)

    def decay(self) -> tuple[float, float, float]:
        a = math.exp(-self.dt_ms / self.tau)
        b = math.exp(-self.dt_ms / self.t_mbr)
        c = (b - a) * self.tau / (self.t_mbr - self.tau)
        return a, b, c
