# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Multi-GPU correctness test for Attn2DFlashInferAttention.

Spawns ``R*C`` ranks via MPI, runs the 2D-mesh CP backend on a cyclic
shard of a single request, all-gathers the result, and compares to a
single-GPU full-causal SDPA computed on the unsharded Q/K/V. The
single-GPU SDPA serves as the (formerly in-tree) vanilla-torch oracle
that was removed in favor of ``attn2d_flashinfer.py``.
"""

import os
import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
import torch.distributed as dist
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

import tensorrt_llm

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

# MPIPoolExecutor leaks a worker thread on first use; keep CI green.
pytestmark = pytest.mark.threadleak(enabled=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init_torch_dist(rank: int, world_size: int) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29577")
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank,
            device_id=torch.device(f"cuda:{rank}"),
        )


class _FakeMapping:
    """Minimal stand-in for ``Mapping`` exposing only attn2d-relevant attrs.

    The full ``Mapping`` class binds PGs through ``MpiTopology`` (no
    ``cp_group_pg``) or ``DeviceMeshTopologyImpl`` (Ray path); neither is a
    natural fit for this test, so we wire the subgroups by hand.

    Exposes both the rank-list properties (``cp_group`` /
    ``attn2d_row_group`` / ``attn2d_col_group``) consumed by the
    ``ops.py`` collective wrappers and the corresponding ``_pg``
    properties used in the PG-mode dispatch branch.  The test forces
    that branch by setting ``TLLM_DISABLE_MPI=1`` in the worker.
    """

    def __init__(self, *, cp_rank: int, R: int, C: int, cp_pg, row_pg, col_pg):
        self.cp_rank = cp_rank
        self.cp_size = R * C
        self._R = R
        self._C = C
        self._cp_pg = cp_pg
        self._row_pg = row_pg
        self._col_pg = col_pg

    def has_cp_attn2d(self) -> bool:
        return True

    @property
    def attn2d_row_size(self) -> int:
        return self._R

    @property
    def attn2d_col_size(self) -> int:
        return self._C

    @property
    def attn2d_row_rank(self) -> int:
        return self.cp_rank % self._R

    @property
    def attn2d_col_rank(self) -> int:
        return self.cp_rank // self._R

    @property
    def cp_group(self):
        return list(range(self.cp_size))

    @property
    def attn2d_row_group(self):
        R, C = self._R, self._C
        row_rank = self.attn2d_row_rank
        cp_group = self.cp_group
        return [cp_group[row_rank + k * R] for k in range(C)]

    @property
    def attn2d_col_group(self):
        R = self._R
        col_rank = self.attn2d_col_rank
        cp_group = self.cp_group
        return [cp_group[col_rank * R + k] for k in range(R)]

    @property
    def attn2d_row_group_pg(self):
        return self._row_pg

    @property
    def attn2d_col_group_pg(self):
        return self._col_pg

    @property
    def cp_group_pg(self):
        return self._cp_pg


def _reference_full_causal_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Full-causal SDPA on unsharded Q/K/V. Returns ``[L, num_heads, D]``."""
    L = q.shape[0]
    q = q.view(L, num_heads, head_dim)
    k = k.view(L, num_kv_heads, head_dim)
    v = v.view(L, num_kv_heads, head_dim)
    n_rep = num_heads // num_kv_heads
    if n_rep > 1:
        k = (
            k[:, :, None, :]
            .expand(L, num_kv_heads, n_rep, head_dim)
            .reshape(L, num_heads, head_dim)
        )
        v = (
            v[:, :, None, :]
            .expand(L, num_kv_heads, n_rep, head_dim)
            .reshape(L, num_heads, head_dim)
        )
    q_h = q.transpose(0, 1).contiguous()  # [H, L, D]
    k_h = k.transpose(0, 1).contiguous()
    v_h = v.transpose(0, 1).contiguous()
    out = torch.nn.functional.scaled_dot_product_attention(q_h, k_h, v_h, is_causal=True)
    return out.transpose(0, 1).contiguous()  # [L, H, D]


# ---------------------------------------------------------------------------
# Per-rank work
# ---------------------------------------------------------------------------


@torch.inference_mode()
def _run_attn2d_check(
    world_size: int,
    rank: int,
    R: int,
    C: int,
    L: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_name: str,
    seed: int,
) -> None:
    # The ops.py collective wrappers branch on ``mpi_disabled()``.  This
    # test uses hand-built torch PGs (no TRT-LLM-internal NCCL comm pool
    # registration), so force the PG-mode dispatch by setting the env
    # var before importing the backend.
    os.environ["TLLM_DISABLE_MPI"] = "1"

    # Lazy import so the test module is collectable on hosts without
    # flashinfer; the parent test gating ensures we only reach here when
    # it is available.
    from tensorrt_llm._torch.attention_backend.attn2d_flashinfer import (
        Attn2DFlashInferAttention,
        Attn2DFlashInferAttentionMetadata,
    )
    from tensorrt_llm._torch.attention_backend.interface import (
        AttentionForwardArgs,
        PredefinedAttentionMask,
    )

    dtype = getattr(torch, dtype_name)
    P = R * C
    assert world_size == P
    L_per = L // P
    assert L_per * P == L, "v0 backend requires L divisible by cp_size"
    hidden = num_heads * head_dim

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    _init_torch_dist(rank, world_size)

    try:
        # Build subgroups.  ``dist.new_group`` requires every rank in WORLD
        # to call it with the same ``ranks`` list in the same order, even
        # for groups it is not a member of.
        cp_pg = dist.new_group(ranks=list(range(P)))
        row_pgs = [
            dist.new_group(ranks=[row_idx + k * R for k in range(C)]) for row_idx in range(R)
        ]
        col_pgs = [
            dist.new_group(ranks=[col_idx * R + k for k in range(R)]) for col_idx in range(C)
        ]

        cp_rank = rank
        row_idx = cp_rank % R
        col_idx = cp_rank // R
        mapping = _FakeMapping(
            cp_rank=cp_rank, R=R, C=C, cp_pg=cp_pg, row_pg=row_pgs[row_idx], col_pg=col_pgs[col_idx]
        )

        # Deterministic Q/K/V: every rank seeds the same CPU generator and
        # moves the result to its own device. Ensures all ranks agree on
        # the global tensor without an explicit broadcast.
        gen = torch.Generator().manual_seed(seed)
        q_full = torch.randn(L, hidden, dtype=dtype, generator=gen).to(device)
        k_full = torch.randn(L, num_kv_heads * head_dim, dtype=dtype, generator=gen).to(device)
        v_full = torch.randn(L, num_kv_heads * head_dim, dtype=dtype, generator=gen).to(device)

        # Cyclic-shard locally: rank r owns positions {p: p%P == r}.
        q_local = q_full[cp_rank::P].contiguous()
        k_local = k_full[cp_rank::P].contiguous()
        v_local = v_full[cp_rank::P].contiguous()

        backend = Attn2DFlashInferAttention(
            layer_idx=0, num_heads=num_heads, head_dim=head_dim, num_kv_heads=num_kv_heads
        )

        metadata = Attn2DFlashInferAttentionMetadata(
            max_num_requests=1,
            max_num_tokens=L_per,
            seq_lens=torch.tensor([L_per], dtype=torch.int32),
            num_contexts=1,
            mapping=mapping,
        )
        metadata.prepare()
        forward_args = AttentionForwardArgs(attention_mask=PredefinedAttentionMask.CAUSAL)

        out_local = backend.forward(q_local, k_local, v_local, metadata, forward_args=forward_args)
        # out_local: [L_per, num_heads * head_dim]

        # All-gather every rank's output to enable a single-GPU comparison.
        out_gathered = torch.empty(P, L_per, hidden, dtype=dtype, device=device)
        dist.all_gather_into_tensor(out_gathered.view(-1), out_local.contiguous().view(-1))

        if rank == 0:
            # gathered[p, i] is the token at absolute position p + i*P.
            # Interleave back to position order: out_sorted[k=i*P+p] = gathered[p, i].
            out_full = out_gathered.permute(1, 0, 2).reshape(L, hidden)
            out_full = out_full.view(L, num_heads, head_dim)

            out_ref = _reference_full_causal_sdpa(
                q_full, k_full, v_full, num_heads, num_kv_heads, head_dim
            )

            # FlashInfer FA2 + LSE-merge in bf16 vs torch SDPA accumulates
            # a few bits of noise on the unsharded path. Tolerance chosen
            # empirically — tighten if the kernel changes.
            atol = 2e-2 if dtype == torch.bfloat16 else 1e-3
            rtol = 1e-2 if dtype == torch.bfloat16 else 1e-3
            torch.testing.assert_close(out_full, out_ref, atol=atol, rtol=rtol)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _entrypoint(world_size, R, C, L, num_heads, num_kv_heads, head_dim, dtype_name, seed):
    """MPIPoolExecutor entry; each invocation runs in its own process."""
    rank = tensorrt_llm.mpi_rank()
    try:
        _run_attn2d_check(
            world_size, rank, R, C, L, num_heads, num_kv_heads, head_dim, dtype_name, seed
        )
    except Exception:
        traceback.print_exc()
        raise
    return True


# ---------------------------------------------------------------------------
# Test entry point
# ---------------------------------------------------------------------------


# (R, C) coverage:
#   (2, 2): square, Q-split with s=1 (both causal + strict-causal tiles)
#   (2, 4), (4, 2): Q-split s=2 and K-split s=2
#   (2, 8), (8, 2): s=4 splits (skipped automatically if < 16 GPUs)
#   (3, 2): coprime -> custom_mask fallback branch
@pytest.mark.parametrize("R,C", [(2, 2), (2, 4), (4, 2), (2, 8), (8, 2), (3, 2)])
@pytest.mark.parametrize("num_heads,num_kv_heads", [(4, 4), (8, 2)])
@pytest.mark.parametrize("dtype_name", ["bfloat16"])
def test_attn2d_flashinfer_matches_full_causal_sdpa(R, C, num_heads, num_kv_heads, dtype_name):
    """Backend output across R*C GPUs must match a single-GPU full-causal SDPA."""
    pytest.importorskip("flashinfer")

    P = R * C
    if torch.cuda.device_count() < P:
        pytest.skip(f"needs {P} CUDA devices, have {torch.cuda.device_count()}")

    # Keep L small but >= a couple of P-blocks so the cyclic interleaving
    # is exercised non-trivially.  s=4 splits need enough rows per
    # sub-tensor to keep the kernel happy.
    L = max(64, 8 * P)
    head_dim = 64
    seed = 0x5EED

    args = (P, R, C, L, num_heads, num_kv_heads, head_dim, dtype_name, seed)
    with MPIPoolExecutor(max_workers=P) as ex:
        results = ex.map(_entrypoint, *zip(*[args] * P))
        for r in results:
            assert r is True
