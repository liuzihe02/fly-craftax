"""MaleCNS v1.0 loader: three feather files in, one signed sparse graph out."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather

_CACHE_VERSION = 1  # bump when the loader's output changes

SIGN = {
    "acetylcholine": 1.0,
    "gaba": -1.0,
    "glutamate": -1.0,
    "histamine": -1.0,
    "dopamine": 1.0,
    "serotonin": 1.0,
    "octopamine": 1.0,
}


@dataclass
class Connectome:
    body_id: np.ndarray
    type: np.ndarray
    cls: np.ndarray
    superclass: np.ndarray
    side: np.ndarray
    sign: np.ndarray
    pre: np.ndarray
    post: np.ndarray
    count: np.ndarray

    @property
    def n(self) -> int:
        return len(self.body_id)

    def index(self, type_prefix=None, types=None, cls=None, side=None) -> np.ndarray:
        keep = np.ones(self.n, dtype=bool)
        if type_prefix is not None:
            keep &= np.char.startswith(self.type.astype(str), type_prefix)
        if types is not None:
            keep &= np.isin(self.type, types)
        if cls is not None:
            keep &= self.cls == cls
        if side is not None:
            keep &= self.side == side
        return np.flatnonzero(keep).astype(np.int32)

    def signed_count(self) -> np.ndarray:
        return (self.sign[self.pre] * self.count).astype(np.float32)


def _neurons(data_dir: Path) -> pd.DataFrame:
    ann = pd.read_feather(data_dir / "body-annotations.feather")
    ann = ann[ann.superclass.notna() & ~ann.superclass.str.startswith("vnc")]
    nt = pd.read_feather(
        data_dir / "body-neurotransmitters.feather",
        columns=["body", "consensus_nt", "celltype_predicted_nt"],
    )
    df = ann.merge(nt, left_on="bodyId", right_on="body", how="left")
    sign = df.consensus_nt.map(SIGN)
    fallback = df.celltype_predicted_nt.map(SIGN)
    df["sign"] = sign.fillna(fallback).fillna(0.0).astype(np.float32)
    df["side"] = df.somaSide.fillna(df.rootSide).fillna("unknown")
    return df.sort_values("bodyId").reset_index(drop=True)


def _edges(data_dir: Path, body_id: np.ndarray, threshold: int) -> tuple[np.ndarray, ...]:
    tbl = feather.read_table(
        data_dir / "edges-traced.feather", columns=["body_pre", "body_post", "weight"]
    )
    e = tbl.to_pandas()
    e = e[e.weight >= threshold]
    pre = np.searchsorted(body_id, e.body_pre.values)
    post = np.searchsorted(body_id, e.body_post.values)
    # searchsorted can land at len(body_id); the clamped equality checks reject those too.
    ok = (body_id[np.minimum(pre, len(body_id) - 1)] == e.body_pre.values) & (
        body_id[np.minimum(post, len(body_id) - 1)] == e.body_post.values
    )
    return (
        pre[ok].astype(np.int32),
        post[ok].astype(np.int32),
        e.weight.values[ok].astype(np.int32),
    )


def load_connectome(threshold: int = 5, data_dir: Path = Path("data")) -> Connectome:
    cache = data_dir / f"connectome_v{_CACHE_VERSION}_t{threshold}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return Connectome(**{k: z[k] for k in z.files})

    df = _neurons(data_dir)
    body_id = df.bodyId.values.astype(np.int64)
    pre, post, count = _edges(data_dir, body_id, threshold)
    sign = df["sign"].values
    keep = sign[pre] != 0
    conn = Connectome(
        body_id=body_id,
        type=df.type.fillna("").values.astype(object),
        cls=df["class"].fillna("").values.astype(object),
        superclass=df.superclass.values.astype(object),
        side=df.side.values.astype(object),
        sign=sign,
        pre=pre[keep],
        post=post[keep],
        count=count[keep],
    )
    np.savez(cache, **conn.__dict__)
    return conn


def subgraph(conn: Connectome, seed_idx, hops: int = 2, max_n: int = 2000):
    """Seeds plus `hops` downstream hops, ranked by incoming signed magnitude, capped at `max_n`.

    Returns (node_idx, pre, post, signed_count) with pre/post re-indexed into
    0..len(node_idx)-1; edges are every connectome edge among the selected nodes.
    """
    sc = conn.signed_count()
    mag = np.abs(sc)
    selected = [int(i) for i in np.asarray(seed_idx)]
    chosen = set(selected)
    frontier = np.asarray(selected, np.int32)
    for _ in range(hops):
        if len(selected) >= max_n:
            break
        m = np.isin(conn.pre, frontier)
        score = np.bincount(conn.post[m], weights=mag[m], minlength=conn.n)
        nz = np.flatnonzero(score > 0)                      # keep the argsort off all 146k
        # stable: the max_n cut falls on a score tie, so the node set must not depend
        # on the sort implementation.
        cand = [i for i in nz[np.argsort(-score[nz], kind="stable")] if i not in chosen]
        cand = cand[: max_n - len(selected)]
        selected += cand
        chosen.update(cand)
        frontier = np.asarray(cand, np.int32)
    nodes = np.asarray(selected, np.int32)
    remap = np.full(conn.n, -1, np.int32)
    remap[nodes] = np.arange(len(nodes), dtype=np.int32)
    keep = (remap[conn.pre] >= 0) & (remap[conn.post] >= 0)
    return nodes, remap[conn.pre[keep]], remap[conn.post[keep]], sc[keep]
