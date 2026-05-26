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
    properties used in the PG-mode dispatch branch.  Same instance
    works for both dispatch modes:

      * PG mode (``TLLM_DISABLE_MPI=1``): row/col allgathers + row
        all_to_all read the ``_pg`` properties.
      * MPI mode (``TLLM_DISABLE_MPI`` unset): row/col allgathers + row
        all_to_all read the rank-list properties; ``cp_group_pg`` is
        still consumed by ``_redistribute_kv_to_row_major`` for the
        K/V mesh-transpose, which uses ``dist.batch_isend_irecv``.
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
    # ``mpi_disabled()`` is read per-call from the env var, so the
    # caller's entrypoint (``_entrypoint`` for PG mode,
    # ``_entrypoint_mpi`` for MPI mode) selects the dispatch path by
    # setting / unsetting ``TLLM_DISABLE_MPI`` before invoking this
    # function.  The hand-built ``cp_pg`` below is needed in BOTH modes:
    # ``_redistribute_kv_to_row_major`` uses ``dist.batch_isend_irecv``
    # for the K/V mesh-transpose, and ``MpiTopology`` does not expose
    # ``cp_group_pg`` (raises ``NotImplementedError``), so even MPI-mode
    # runs need a torch PG for that one call.

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
    # Per-rank cyclic shard length.  For L not divisible by P, ranks
    # r < L%P get one extra token; the backend pads internally so all
    # collectives stay uniform.
    L_local = (L - rank + P - 1) // P  # ceil((L - rank) / P)
    L_max = (L + P - 1) // P
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
        assert q_local.shape[0] == L_local

        backend = Attn2DFlashInferAttention(
            layer_idx=0, num_heads=num_heads, head_dim=head_dim, num_kv_heads=num_kv_heads
        )

        metadata = Attn2DFlashInferAttentionMetadata(
            max_num_requests=1,
            max_num_tokens=L_local,
            seq_lens=torch.tensor([L_local], dtype=torch.int32),
            num_contexts=1,
            mapping=mapping,
            total_input_lens=torch.tensor([L], dtype=torch.int32),
        )
        metadata.prepare()
        forward_args = AttentionForwardArgs(attention_mask=PredefinedAttentionMask.CAUSAL)

        out_local = backend.forward(q_local, k_local, v_local, metadata, forward_args=forward_args)
        # out_local: [L_local, num_heads * head_dim]
        assert out_local.shape[0] == L_local

        # Pad to L_max for a uniform all_gather, then trim to real
        # positions on rank 0.  Padding rows carry virtual positions
        # >= L and are discarded by the position slice below.
        if L_local < L_max:
            pad = out_local.new_zeros(L_max - L_local, hidden)
            out_local_padded = torch.cat([out_local, pad], dim=0)
        else:
            out_local_padded = out_local
        out_gathered = torch.empty(P, L_max, hidden, dtype=dtype, device=device)
        dist.all_gather_into_tensor(out_gathered.view(-1), out_local_padded.contiguous().view(-1))

        if rank == 0:
            # gathered[r, i] is the token at absolute position r + i*P.
            # Permute (P, L_max, hidden) -> (L_max, P, hidden) -> flat
            # position order, then slice to L real positions.
            out_full = out_gathered.permute(1, 0, 2).reshape(L_max * P, hidden)[:L]
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
    """MPIPoolExecutor entry (PG-mode dispatch); each invocation runs in
    its own process.

    Forces ``TLLM_DISABLE_MPI=1`` so the ``attn2d_*`` collective wrappers
    in ``tensorrt_llm._torch.distributed.ops`` go through the
    ``torch.distributed`` ProcessGroup path (rank-list keyed comm pool
    not required).
    """
    os.environ["TLLM_DISABLE_MPI"] = "1"
    rank = tensorrt_llm.mpi_rank()
    try:
        _run_attn2d_check(
            world_size, rank, R, C, L, num_heads, num_kv_heads, head_dim, dtype_name, seed
        )
    except Exception:
        traceback.print_exc()
        raise
    return True


def _entrypoint_mpi(world_size, R, C, L, num_heads, num_kv_heads, head_dim, dtype_name, seed):
    """MPIPoolExecutor entry (MPI-mode dispatch).

    Mirrors ``_entrypoint`` but explicitly *unsets* ``TLLM_DISABLE_MPI``
    so the ``attn2d_*`` collective wrappers take the MPI-backed branch:
    row/col allgathers go through ``torch.ops.trtllm.allgather`` keyed
    on rank lists (the TRT-LLM NCCL comm pool bootstraps comms over
    MPI), and the row all_to_all uses ``torch.ops.trtllm.alltoall_helix``.

    The K/V mesh-transpose still needs a torch PG; ``_run_attn2d_check``
    hand-builds one regardless of dispatch mode (see its docstring).
    """
    os.environ.pop("TLLM_DISABLE_MPI", None)
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
#   (1, 4): R==1 fast path (skips K/V mesh-transpose and col all-gather);
#           hits Q-split with k_sorted = k, v_sorted = v
#   (4, 1): C==1 fast path (skips Q row all-gather and the row-group
#           all_to_all + LSE merge); output returns straight from the
#           K-split branch
#   (2, 6): Q-split with s=3 -- ranks with col_idx >= 2 see strict-
#           causal on every t tile, exercising the all-strict pattern
# L_extra coverage:
#   0: L divisible by P (uniform shard sizes)
#   1: L = L_base + 1 -> rank 0 has one extra token (uneven sharding,
#      pad/trim path exercised; for the coprime (3,2) mesh this also
#      pairs an uneven and an even rank in the K/V mesh-transpose)
@pytest.mark.parametrize(
    "R,C",
    [(2, 2), (2, 4), (4, 2), (2, 8), (8, 2), (3, 2), (1, 4), (4, 1), (2, 6)],
)
@pytest.mark.parametrize("num_heads,num_kv_heads", [(4, 4), (8, 2)])
@pytest.mark.parametrize("dtype_name", ["bfloat16"])
@pytest.mark.parametrize("L_extra", [0, 1])
def test_attn2d_flashinfer_matches_full_causal_sdpa(
    R, C, num_heads, num_kv_heads, dtype_name, L_extra
):
    """Backend output across R*C GPUs must match a single-GPU full-causal SDPA."""
    pytest.importorskip("flashinfer")

    P = R * C
    if torch.cuda.device_count() < P:
        pytest.skip(f"needs {P} CUDA devices, have {torch.cuda.device_count()}")

    # Pick an L_base that is >= 64 and exactly divisible by P, then add
    # L_extra so we can sweep both the divisible and uneven cases.
    L_base = max(64, 8 * P)
    L_base = ((L_base + P - 1) // P) * P
    L = L_base + L_extra
    head_dim = 64
    seed = 0x5EED

    args = (P, R, C, L, num_heads, num_kv_heads, head_dim, dtype_name, seed)
    with MPIPoolExecutor(max_workers=P) as ex:
        results = ex.map(_entrypoint, *zip(*[args] * P))
        for r in results:
            assert r is True


# ---------------------------------------------------------------------------
# Single-mesh focused tests: head_dim=128 and float16
# ---------------------------------------------------------------------------
#
# Kept off the main cross-product to avoid blowing up the test count -- one
# focused mesh per axis is enough to verify the backend doesn't regress on
# the alternate kernel dispatch (head_dim=128) or accumulator dtype
# (float16).  (2, 2) chosen because it's the cheapest mesh (4 GPUs) that
# still exercises every gather + the row-group all_to_all + LSE-merge tail.


def test_attn2d_flashinfer_head_dim_128():
    """Verify backend correctness at head_dim=128.

    FlashInfer routes head_dim=128 through a different kernel template
    than head_dim=64, so this guards against dispatch-time regressions
    that the main matrix (head_dim=64 only) would miss.
    """
    pytest.importorskip("flashinfer")

    R, C = 2, 2
    P = R * C
    if torch.cuda.device_count() < P:
        pytest.skip(f"needs {P} CUDA devices, have {torch.cuda.device_count()}")

    L = ((max(64, 8 * P) + P - 1) // P) * P
    head_dim = 128
    dtype_name = "bfloat16"
    seed = 0x5EED
    num_heads, num_kv_heads = 4, 4

    args = (P, R, C, L, num_heads, num_kv_heads, head_dim, dtype_name, seed)
    with MPIPoolExecutor(max_workers=P) as ex:
        results = ex.map(_entrypoint, *zip(*[args] * P))
        for r in results:
            assert r is True


def test_attn2d_flashinfer_float16():
    """Verify backend correctness with float16 dtype.

    The merge_states + LSE accumulator path is dtype-sensitive; this
    guards against float16-only bugs that the main matrix (bfloat16
    only) would miss.
    """
    pytest.importorskip("flashinfer")

    R, C = 2, 2
    P = R * C
    if torch.cuda.device_count() < P:
        pytest.skip(f"needs {P} CUDA devices, have {torch.cuda.device_count()}")

    L = ((max(64, 8 * P) + P - 1) // P) * P
    head_dim = 64
    dtype_name = "float16"
    seed = 0x5EED
    num_heads, num_kv_heads = 4, 4

    args = (P, R, C, L, num_heads, num_kv_heads, head_dim, dtype_name, seed)
    with MPIPoolExecutor(max_workers=P) as ex:
        results = ex.map(_entrypoint, *zip(*[args] * P))
        for r in results:
            assert r is True


# ---------------------------------------------------------------------------
# Multi-request batch test
# ---------------------------------------------------------------------------
#
# The backend's forward iterates over ``zip(seq_lens, total_lens)`` and
# runs the full set of collectives once per request, concatenating the
# outputs along the token axis.  The single-request matrix above never
# enters the iteration loop's second pass, so this test drives a 3-request
# batch with mixed divisible / uneven lengths to verify the per-request
# dispatch stays in lockstep across ranks and the output concatenation
# preserves request order.


@torch.inference_mode()
def _run_attn2d_check_multi(
    world_size: int,
    rank: int,
    R: int,
    C: int,
    L_list,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_name: str,
    seed: int,
) -> None:
    # Dispatch mode is selected by the caller via ``TLLM_DISABLE_MPI``;
    # see ``_run_attn2d_check`` for details.  Hand-built ``cp_pg`` below
    # is required in both PG and MPI modes for the K/V mesh-transpose.

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
    hidden = num_heads * head_dim
    kv_hidden = num_kv_heads * head_dim

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    _init_torch_dist(rank, world_size)

    try:
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
            cp_rank=cp_rank,
            R=R,
            C=C,
            cp_pg=cp_pg,
            row_pg=row_pgs[row_idx],
            col_pg=col_pgs[col_idx],
        )

        # Build per-request global tensors with distinct seeds so the
        # reference oracle differs across requests and order-mismatch
        # bugs are detectable.
        q_globals, k_globals, v_globals = [], [], []
        q_locals, k_locals, v_locals = [], [], []
        L_locals = []
        for i, L in enumerate(L_list):
            gen = torch.Generator().manual_seed(seed + i)
            q_full = torch.randn(L, hidden, dtype=dtype, generator=gen).to(device)
            k_full = torch.randn(L, kv_hidden, dtype=dtype, generator=gen).to(device)
            v_full = torch.randn(L, kv_hidden, dtype=dtype, generator=gen).to(device)
            q_globals.append(q_full)
            k_globals.append(k_full)
            v_globals.append(v_full)

            q_loc = q_full[cp_rank::P].contiguous()
            k_loc = k_full[cp_rank::P].contiguous()
            v_loc = v_full[cp_rank::P].contiguous()
            q_locals.append(q_loc)
            k_locals.append(k_loc)
            v_locals.append(v_loc)
            L_locals.append(q_loc.shape[0])

        q_cat = torch.cat(q_locals, dim=0)
        k_cat = torch.cat(k_locals, dim=0)
        v_cat = torch.cat(v_locals, dim=0)

        backend = Attn2DFlashInferAttention(
            layer_idx=0, num_heads=num_heads, head_dim=head_dim, num_kv_heads=num_kv_heads
        )
        metadata = Attn2DFlashInferAttentionMetadata(
            max_num_requests=len(L_list),
            max_num_tokens=int(sum(L_locals)),
            seq_lens=torch.tensor(L_locals, dtype=torch.int32),
            num_contexts=len(L_list),
            mapping=mapping,
            total_input_lens=torch.tensor(list(L_list), dtype=torch.int32),
        )
        metadata.prepare()
        forward_args = AttentionForwardArgs(attention_mask=PredefinedAttentionMask.CAUSAL)

        out_cat = backend.forward(q_cat, k_cat, v_cat, metadata, forward_args=forward_args)
        assert out_cat.shape[0] == sum(L_locals)

        # Per-request gather and compare against single-GPU SDPA.
        offset = 0
        for i, L in enumerate(L_list):
            L_local = L_locals[i]
            out_local = out_cat[offset : offset + L_local]
            offset += L_local
            L_max = (L + P - 1) // P
            if L_local < L_max:
                pad = out_local.new_zeros(L_max - L_local, hidden)
                out_local_padded = torch.cat([out_local, pad], dim=0)
            else:
                out_local_padded = out_local
            out_gathered = torch.empty(P, L_max, hidden, dtype=dtype, device=device)
            dist.all_gather_into_tensor(
                out_gathered.view(-1), out_local_padded.contiguous().view(-1)
            )

            if rank == 0:
                out_full = out_gathered.permute(1, 0, 2).reshape(L_max * P, hidden)[:L]
                out_full = out_full.view(L, num_heads, head_dim)
                out_ref = _reference_full_causal_sdpa(
                    q_globals[i],
                    k_globals[i],
                    v_globals[i],
                    num_heads,
                    num_kv_heads,
                    head_dim,
                )
                atol = 2e-2 if dtype == torch.bfloat16 else 1e-3
                rtol = 1e-2 if dtype == torch.bfloat16 else 1e-3
                torch.testing.assert_close(out_full, out_ref, atol=atol, rtol=rtol)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _entrypoint_multi(
    world_size, R, C, L_list, num_heads, num_kv_heads, head_dim, dtype_name, seed
):
    """MPIPoolExecutor entry for the multi-request batch (PG-mode dispatch)."""
    os.environ["TLLM_DISABLE_MPI"] = "1"
    rank = tensorrt_llm.mpi_rank()
    try:
        _run_attn2d_check_multi(
            world_size, rank, R, C, L_list, num_heads, num_kv_heads, head_dim, dtype_name, seed
        )
    except Exception:
        traceback.print_exc()
        raise
    return True


def _entrypoint_mpi_multi(
    world_size, R, C, L_list, num_heads, num_kv_heads, head_dim, dtype_name, seed
):
    """MPIPoolExecutor entry for the multi-request batch (MPI-mode dispatch).

    Mirrors ``_entrypoint_multi`` but unsets ``TLLM_DISABLE_MPI`` so the
    ``attn2d_*`` collective wrappers take the MPI-backed branch.
    """
    os.environ.pop("TLLM_DISABLE_MPI", None)
    rank = tensorrt_llm.mpi_rank()
    try:
        _run_attn2d_check_multi(
            world_size, rank, R, C, L_list, num_heads, num_kv_heads, head_dim, dtype_name, seed
        )
    except Exception:
        traceback.print_exc()
        raise
    return True


# (R, C) coverage for multi-request:
#   (2, 2): square mesh, Q-split path
#   (3, 2): coprime -> custom_mask fallback (verifies multi-request also
#           drives the non-divisible mesh branch in lockstep)
@pytest.mark.parametrize("R,C", [(2, 2), (3, 2)])
def test_attn2d_flashinfer_multi_request_batch(R, C):
    """Backend forward must correctly iterate a multi-request batch.

    Drives a 3-request batch (mixing divisible and uneven total lengths,
    plus one shorter request) through a single ``forward`` call.  The
    backend must run its full set of collectives once per request and
    concatenate the outputs in request order; every per-request output
    is independently verified against single-GPU SDPA on that request's
    own Q/K/V.
    """
    pytest.importorskip("flashinfer")

    P = R * C
    if torch.cuda.device_count() < P:
        pytest.skip(f"needs {P} CUDA devices, have {torch.cuda.device_count()}")

    # Three requests: divisible, uneven (rank 0 gets +1 token), short.
    # Distinct lengths so an output-ordering bug surfaces as a shape
    # mismatch rather than only a value mismatch.
    L_div = ((max(48, 6 * P) + P - 1) // P) * P
    L_uneven = ((max(80, 10 * P) + P - 1) // P) * P + 1
    L_short = ((max(32, 4 * P) + P - 1) // P) * P
    L_list = (L_div, L_uneven, L_short)

    head_dim = 64
    dtype_name = "bfloat16"
    seed = 0xA11CE
    num_heads, num_kv_heads = 8, 2  # GQA

    args = (P, R, C, L_list, num_heads, num_kv_heads, head_dim, dtype_name, seed)
    with MPIPoolExecutor(max_workers=P) as ex:
        results = ex.map(_entrypoint_multi, *zip(*[args] * P))
        for r in results:
            assert r is True


# ---------------------------------------------------------------------------
# MPI-mode coverage
# ---------------------------------------------------------------------------
#
# All tests above force the PG-mode dispatch (``TLLM_DISABLE_MPI=1``).
# Production deployments leave MPI enabled and route the ``attn2d_*``
# wrappers through ``torch.ops.trtllm.allgather`` /
# ``torch.ops.trtllm.alltoall_helix`` (rank-list keyed NCCL comm pool,
# bootstrapped over MPI) instead of ``torch.distributed`` ProcessGroups.
# This was previously uncovered by CI; the tests below exercise that
# code path on a smaller mesh matrix to keep CI cost in check while
# still hitting all three local-attention branches:
#
#   (2, 2): Q-split path (square mesh, s == 1)
#   (4, 2): K-split path (R % C == 0, s == 2; needs 8 GPUs)
#   (3, 2): custom_mask fallback (coprime mesh)
#
# ``_redistribute_kv_to_row_major`` still uses ``dist.batch_isend_irecv``
# on ``cp_pg`` in both modes -- ``MpiTopology`` does not expose
# ``cp_group_pg``, so even MPI-mode runs need a hand-built torch PG for
# the K/V mesh-transpose.  This is a known backend limitation; see the
# note in ``attn2d_flashinfer.py:_redistribute_kv_to_row_major`` about
# the future ``torch.ops.trtllm.permute_send_recv`` op that would lift
# the PG dependency.


@pytest.mark.parametrize("R,C", [(2, 2), (4, 2), (3, 2)])
@pytest.mark.parametrize("L_extra", [0, 1])
def test_attn2d_flashinfer_mpi_mode_matches_full_causal_sdpa(R, C, L_extra):
    """MPI-mode dispatch: backend output must match single-GPU full-causal SDPA.

    Mirror of ``test_attn2d_flashinfer_matches_full_causal_sdpa`` but
    routes the row/col allgathers and the row all_to_all through the
    MPI-backed NCCL comm pool (the production default) instead of
    torch.distributed PGs.
    """
    pytest.importorskip("flashinfer")

    P = R * C
    if torch.cuda.device_count() < P:
        pytest.skip(f"needs {P} CUDA devices, have {torch.cuda.device_count()}")

    L_base = max(64, 8 * P)
    L_base = ((L_base + P - 1) // P) * P
    L = L_base + L_extra
    head_dim = 64
    dtype_name = "bfloat16"
    seed = 0x5EED
    num_heads, num_kv_heads = 8, 2  # GQA — exercises broadcast in SDPA reference

    args = (P, R, C, L, num_heads, num_kv_heads, head_dim, dtype_name, seed)
    with MPIPoolExecutor(max_workers=P) as ex:
        results = ex.map(_entrypoint_mpi, *zip(*[args] * P))
        for r in results:
            assert r is True


@pytest.mark.parametrize("R,C", [(2, 2)])
def test_attn2d_flashinfer_mpi_mode_multi_request_batch(R, C):
    """MPI-mode dispatch: multi-request batch must match per-request SDPA.

    Mirror of ``test_attn2d_flashinfer_multi_request_batch`` running the
    MPI-backed wrappers.  Kept on a single small mesh (2, 2) -- the
    multi-request iteration is exercised by the forward loop itself and
    is independent of the (R, C) branch.
    """
    pytest.importorskip("flashinfer")

    P = R * C
    if torch.cuda.device_count() < P:
        pytest.skip(f"needs {P} CUDA devices, have {torch.cuda.device_count()}")

    L_div = ((max(48, 6 * P) + P - 1) // P) * P
    L_uneven = ((max(80, 10 * P) + P - 1) // P) * P + 1
    L_short = ((max(32, 4 * P) + P - 1) // P) * P
    L_list = (L_div, L_uneven, L_short)

    head_dim = 64
    dtype_name = "bfloat16"
    seed = 0xA11CE
    num_heads, num_kv_heads = 8, 2

    args = (P, R, C, L_list, num_heads, num_kv_heads, head_dim, dtype_name, seed)
    with MPIPoolExecutor(max_workers=P) as ex:
        results = ex.map(_entrypoint_mpi_multi, *zip(*[args] * P))
        for r in results:
            assert r is True
