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
shard of a single (or multi-) request, all-gathers the result via the
TRT-LLM rank-list-keyed wrappers, and compares to a single-GPU
full-causal SDPA computed on the unsharded Q/K/V.  The single-GPU SDPA
serves as the (formerly in-tree) vanilla-torch oracle that was removed
in favor of ``attn2d_flashinfer.py``.

Matches the convention used by the other CP-attention tests
(``test_mha_helix.py`` / ``test_mla_helix.py``): real ``Mapping`` with
``CpType.ATTN2D``, MPI-backed NCCL comm pool, no ``torch.distributed``
ProcessGroup setup.  Every collective in the backend -- including the
K/V mesh-transpose via ``torch.ops.trtllm.permute_send_recv`` -- is
keyed on rank lists drawn from ``mapping``, so this test exercises
exactly the production code path.
"""

import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

import tensorrt_llm
from tensorrt_llm._torch.distributed import cp_allgather
from tensorrt_llm.mapping import CpType, Mapping

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


def _build_mapping(rank: int, P: int, R: int, C: int) -> Mapping:
    """Real ``Mapping`` with ``CpType.ATTN2D`` for a single-(tp=1, pp=1) test."""
    return Mapping(
        world_size=P,
        rank=rank,
        cp_size=P,
        cp_config={
            "cp_type": CpType.ATTN2D,
            "row_size": R,
            "col_size": C,
        },
    )


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


def _gather_full_output(
    out_local: torch.Tensor,
    *,
    L: int,
    L_max: int,
    hidden: int,
    P: int,
    dtype: torch.dtype,
    device: torch.device,
    mapping: Mapping,
) -> torch.Tensor:
    """Pad to ``L_max``, cp-allgather, and reassemble into absolute position order.

    Returns a ``[L, hidden]`` tensor on every rank (caller compares on
    rank 0).  Padding rows carry virtual positions ``>= L`` and are
    discarded by the position slice below.
    """
    L_local = out_local.shape[0]
    if L_local < L_max:
        pad = out_local.new_zeros(L_max - L_local, hidden)
        out_local_padded = torch.cat([out_local, pad], dim=0)
    else:
        out_local_padded = out_local
    # cp_allgather concatenates along dim 0 in cp-rank order, so
    # gathered[r*L_max : (r+1)*L_max] is rank r's contribution.
    gathered = cp_allgather(out_local_padded.contiguous(), mapping=mapping, dim=0)
    # gathered[r, i] is the token at absolute position r + i*P; permute
    # to flat position order, then slice to L real positions.
    return gathered.view(P, L_max, hidden).permute(1, 0, 2).reshape(L_max * P, hidden)[:L]


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

    mapping = _build_mapping(rank, P, R, C)
    cp_rank = mapping.cp_rank

    # Deterministic Q/K/V: every rank seeds the same CPU generator and
    # moves the result to its own device.  Ensures all ranks agree on
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

    out_full = _gather_full_output(
        out_local,
        L=L,
        L_max=L_max,
        hidden=hidden,
        P=P,
        dtype=dtype,
        device=device,
        mapping=mapping,
    )

    if rank == 0:
        out_full = out_full.view(L, num_heads, head_dim)
        out_ref = _reference_full_causal_sdpa(
            q_full, k_full, v_full, num_heads, num_kv_heads, head_dim
        )
        # FlashInfer FA2 + LSE-merge in bf16 vs torch SDPA accumulates
        # a few bits of noise on the unsharded path.  Tolerance chosen
        # empirically -- tighten if the kernel changes.
        atol = 2e-2 if dtype == torch.bfloat16 else 1e-3
        rtol = 1e-2 if dtype == torch.bfloat16 else 1e-3
        torch.testing.assert_close(out_full, out_ref, atol=atol, rtol=rtol)


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
# Single-request tests
# ---------------------------------------------------------------------------


# (R, C) coverage (capped at <= 8 GPUs to match the convention used by
# every other test in tests/unittest/_torch/multi_gpu/ -- the shared
# ``mpi_pool_executor`` fixture in tests/unittest/conftest.py:333
# parametrizes worker counts as [2, 4, 8]).  Configurations needing more
# than 8 GPUs (s=3 with full mesh, s=4 with full mesh) belong in
# tests/integration/ once attn2d is wired into a model end-to-end and
# can be exercised via the multi-node test lists.
#
# Backend constraint: R % C == 0 or C % R == 0 -- non-divisible meshes
# (e.g. (3, 2)) are rejected by an assertion in forward().
#
#   (2, 2): square, Q-split with s=1 (both causal + strict-causal tiles)
#   (2, 4), (4, 2): Q-split s=2 and K-split s=2
#   (1, 4): R==1 fast path (skips K/V mesh-transpose and col all-gather);
#           hits Q-split with k_sorted = k, v_sorted = v (degenerate s=4)
#   (4, 1): C==1 fast path (skips Q row all-gather and the row-group
#           all_to_all + LSE merge); output returns straight from the
#           K-split branch (degenerate s=4)
#
# Known gaps deferred to integration tests:
#   - Q-split s=3 (would be (2, 6), 12 GPUs)
#   - Q-split s=4 with non-trivial K/V mesh-transpose (would be (2, 8))
#   - K-split s=4 with full row all_to_all + LSE merge tail (would be
#     (8, 2))
#   - All-strict-causal tile pattern (subset of s=3 / large-s coverage)
#
# L_extra coverage:
#   0: L divisible by P (uniform shard sizes)
#   1: L = L_base + 1 -> rank 0 has one extra token (uneven sharding,
#      pad/trim path exercised)
@pytest.mark.parametrize(
    "R,C",
    [(2, 2), (2, 4), (4, 2), (1, 4), (4, 1)],
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

    mapping = _build_mapping(rank, P, R, C)
    cp_rank = mapping.cp_rank

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
        out_full = _gather_full_output(
            out_local,
            L=L,
            L_max=L_max,
            hidden=hidden,
            P=P,
            dtype=dtype,
            device=device,
            mapping=mapping,
        )

        if rank == 0:
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


def _entrypoint_multi(
    world_size, R, C, L_list, num_heads, num_kv_heads, head_dim, dtype_name, seed
):
    """MPIPoolExecutor entry for the multi-request batch."""
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
#   (4, 2): K-split path (verifies multi-request also drives the K-split
#           merge_states tail in lockstep across ranks)
@pytest.mark.parametrize("R,C", [(2, 2), (4, 2)])
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
# Chunked-prefill test: exercise the cache write + read + chunked compute
# paths in a single forward sequence (two consecutive forwards on the same
# request, second forward reads chunk 1's K/V from the cache and merges with
# chunk 2's new K/V).
# ---------------------------------------------------------------------------


class _MockKVCacheManager:
    """Minimal per-rank KV cache manager satisfying the backend's API.

    The backend pulls four things off ``metadata.kv_cache_manager``:
        * ``tokens_per_block``        -> page size
        * ``blocks_in_primary_pool``  -> max pages on this rank
        * ``get_buffers(layer_idx, kv_layout)``  -> NHD cache buffer
        * ``get_batch_cache_indices(request_ids)`` -> per-request page lists

    Pages are pre-allocated per request via ``allocate_pages``; this is a
    test harness, not a production allocator, so we just hand out
    contiguous IDs.  NHD layout matches what flashinfer.page.append_paged_kv_cache
    and the backend's ``_materialize_kv_from_pages`` expect:
    ``(max_pages, 2, page_size, num_kv_heads, head_dim)``.
    """

    def __init__(
        self,
        *,
        page_size: int,
        max_pages: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.tokens_per_block = page_size
        self.blocks_in_primary_pool = max_pages
        self._buffer = torch.zeros(
            (max_pages, 2, page_size, num_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        self._req_pages = {}
        self._next_page = 0

    def get_buffers(self, layer_idx: int, kv_layout: str = "NHD") -> torch.Tensor:
        assert kv_layout == "NHD", "test mock only supports NHD layout"
        return self._buffer

    def get_batch_cache_indices(self, request_ids):
        return [self._req_pages[req_id] for req_id in request_ids]

    def allocate_pages(self, req_id: int, num_pages: int) -> None:
        assert self._next_page + num_pages <= self.blocks_in_primary_pool, (
            f"out of pages: tried to allocate {num_pages} after {self._next_page}, "
            f"have {self.blocks_in_primary_pool}"
        )
        self._req_pages[req_id] = list(range(self._next_page, self._next_page + num_pages))
        self._next_page += num_pages


@torch.inference_mode()
def _run_attn2d_chunked_check(
    world_size: int,
    rank: int,
    R: int,
    C: int,
    L: int,
    L_chunk_1: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_name: str,
    seed: int,
) -> None:
    from tensorrt_llm._torch.attention_backend.attn2d_flashinfer import (
        Attn2DFlashInferAttention,
        Attn2DFlashInferAttentionMetadata,
    )
    from tensorrt_llm._torch.attention_backend.interface import (
        AttentionForwardArgs,
        PredefinedAttentionMask,
    )
    from tensorrt_llm._torch.metadata import KVCacheParams

    dtype = getattr(torch, dtype_name)
    P = R * C
    assert world_size == P
    L_chunk_2 = L - L_chunk_1
    assert L_chunk_1 > 0 and L_chunk_2 > 0, (
        f"both chunks must be non-empty: L={L}, L_chunk_1={L_chunk_1}"
    )

    # Per-rank cyclic counts: when L_chunk_1 mod P != 0, the per-rank chunk
    # split is uneven; chunk 2's L_prev = L_chunk_1 is then not a multiple of
    # P either, so the backend's first_q[row_idx] formula picks up a non-zero
    # rotation and the kernel-call diff (natural_shift - required_shift) goes
    # through the full classification (instead of staying = 0 for the
    # P-aligned case).  When L mod P != 0, ranks r < L%P have one extra token;
    # _gather_full_output handles this by padding each rank's output to
    # L_max = ceil(L/P) before the cp_allgather.
    L_local = (L - rank + P - 1) // P  # cyclic count on this rank
    if rank < L_chunk_1:
        L_local_chunk_1 = (L_chunk_1 - rank + P - 1) // P
    else:
        L_local_chunk_1 = 0
    L_local_chunk_2 = L_local - L_local_chunk_1
    assert L_local_chunk_2 >= 0
    hidden = num_heads * head_dim
    kv_hidden = num_kv_heads * head_dim

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    mapping = _build_mapping(rank, P, R, C)
    cp_rank = mapping.cp_rank

    # Deterministic global Q/K/V (same on every rank).
    gen = torch.Generator().manual_seed(seed)
    q_full = torch.randn(L, hidden, dtype=dtype, generator=gen).to(device)
    k_full = torch.randn(L, kv_hidden, dtype=dtype, generator=gen).to(device)
    v_full = torch.randn(L, kv_hidden, dtype=dtype, generator=gen).to(device)

    # Cyclic shard.
    q_local_full = q_full[cp_rank::P].contiguous()
    k_local_full = k_full[cp_rank::P].contiguous()
    v_local_full = v_full[cp_rank::P].contiguous()

    # Per-chunk slices of THIS rank's cyclic shard (in cyclic order, which
    # = sorted absolute-position order, so chunk 1 is the first
    # L_local_chunk_1 entries of the shard and chunk 2 is the rest).
    q_local_c1 = q_local_full[:L_local_chunk_1].contiguous()
    k_local_c1 = k_local_full[:L_local_chunk_1].contiguous()
    v_local_c1 = v_local_full[:L_local_chunk_1].contiguous()
    q_local_c2 = q_local_full[L_local_chunk_1:].contiguous()
    k_local_c2 = k_local_full[L_local_chunk_1:].contiguous()
    v_local_c2 = v_local_full[L_local_chunk_1:].contiguous()

    # Mock cache manager: 1 page = L_local tokens for simplicity (one page
    # per rank per request).  Allocate enough pages for ceil(L_local / page_size).
    page_size = max(8, L_local)
    num_pages = (L_local + page_size - 1) // page_size
    max_pages = max(num_pages * 4, 8)
    req_id = 0
    cache_mgr = _MockKVCacheManager(
        page_size=page_size,
        max_pages=max_pages,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )
    cache_mgr.allocate_pages(req_id, num_pages)

    backend = Attn2DFlashInferAttention(
        layer_idx=0,
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
    )

    def _build_metadata(seq_lens_val: int, total: int, chunk: int, cached_local: int):
        md = Attn2DFlashInferAttentionMetadata(
            max_num_requests=1,
            max_num_tokens=L_local_chunk_1 + L_local_chunk_2,
            seq_lens=torch.tensor([seq_lens_val], dtype=torch.int32),
            num_contexts=1,
            mapping=mapping,
            kv_cache_manager=cache_mgr,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=[cached_local],
            ),
            total_input_lens=torch.tensor([total], dtype=torch.int32),
            chunk_input_lens=torch.tensor([chunk], dtype=torch.int32),
        )
        md.request_ids = [req_id]
        md.prepare()
        return md

    forward_args = AttentionForwardArgs(attention_mask=PredefinedAttentionMask.CAUSAL)

    # --- Forward 1: chunk 1 (L_prev = 0). ---
    md1 = _build_metadata(
        seq_lens_val=L_local_chunk_1,
        total=L_chunk_1,
        chunk=L_chunk_1,
        cached_local=0,
    )
    out_c1 = backend.forward(q_local_c1, k_local_c1, v_local_c1, md1, forward_args=forward_args)
    assert out_c1.shape[0] == L_local_chunk_1

    # --- Forward 2: chunk 2 (L_prev = L_chunk_1).  Cache now holds chunk 1
    # K/V; chunk 2's forward writes new K/V then reads (cached + new) for
    # the mesh comm.
    md2 = _build_metadata(
        seq_lens_val=L_local_chunk_2,
        total=L,  # cumulative K range = L_chunk_1 + L_chunk_2 = L
        chunk=L_chunk_2,
        cached_local=L_local_chunk_1,
    )
    out_c2 = backend.forward(q_local_c2, k_local_c2, v_local_c2, md2, forward_args=forward_args)
    assert out_c2.shape[0] == L_local_chunk_2

    # Concatenate chunk outputs in cyclic order (chunk 1 positions precede
    # chunk 2 positions in absolute order, so cyclic order is preserved).
    out_local = torch.cat([out_c1, out_c2], dim=0)
    assert out_local.shape[0] == L_local

    # Gather and compare against single-GPU full-causal SDPA.
    # Use ceil(L/P) as L_max so that ranks with L_local < ceil(L/P) are
    # padded correctly when L % P != 0.
    out_full = _gather_full_output(
        out_local,
        L=L,
        L_max=(L + P - 1) // P,
        hidden=hidden,
        P=P,
        dtype=dtype,
        device=device,
        mapping=mapping,
    )
    if rank == 0:
        out_full = out_full.view(L, num_heads, head_dim)
        out_ref = _reference_full_causal_sdpa(
            q_full, k_full, v_full, num_heads, num_kv_heads, head_dim
        )
        atol = 2e-2 if dtype == torch.bfloat16 else 1e-3
        rtol = 1e-2 if dtype == torch.bfloat16 else 1e-3
        torch.testing.assert_close(out_full, out_ref, atol=atol, rtol=rtol)


def _entrypoint_chunked(
    world_size, R, C, L, L_chunk_1, num_heads, num_kv_heads, head_dim, dtype_name, seed
):
    rank = tensorrt_llm.mpi_rank()
    try:
        _run_attn2d_chunked_check(
            world_size,
            rank,
            R,
            C,
            L,
            L_chunk_1,
            num_heads,
            num_kv_heads,
            head_dim,
            dtype_name,
            seed,
        )
    except Exception:
        traceback.print_exc()
        raise
    return True


# Chunked-prefill coverage:
#   (2, 2): square mesh, Q-split degenerate s=1 (most cases).
#   (4, 2): K-split with s=2 (verifies LSE-merge across u under chunked prefill).
#   (2, 4): Q-split with s=2 (verifies per-t shift derivation under chunked prefill).
#
# chunk_offset axis: amount added to L // 2 to pick the chunk-1 boundary.
#   0: P-aligned -- uniform per-rank chunk counts, L_prev mod P == 0 for chunk 2.
#   1: non-aligned by 1 -- exercises variable-size allgatherv across the row /
#      col groups (per-rank chunk-1 counts differ by 1) and a non-zero L_prev
#      mod P in chunk 2 (so the backend's first_q[row_idx] rotation kicks in
#      and the kernel diff classification is non-trivial).
#
# L_extra axis: whether the total sequence length is P-divisible.
#   0: L divisible by P -- uniform per-rank L_local; baseline case.
#   1: L = L_base + 1 -- L mod P != 0; ranks r < 1 have one extra token.
#      Stresses the global-total derivation in model_engine.py (the last-chunk
#      snap to total_input_len_cp) and the per-rank padding in _gather_full_output.
@pytest.mark.parametrize("R,C", [(2, 2), (4, 2), (2, 4)])
@pytest.mark.parametrize("chunk_offset", [0, 1])
@pytest.mark.parametrize("L_extra", [0, 1])
def test_attn2d_flashinfer_chunked_prefill(R, C, chunk_offset, L_extra):
    """Two-chunk prefill (cache write + read) matches single-shot SDPA.

    The first forward processes chunk 1 with empty cache; new K/V is
    appended.  The second forward processes chunk 2: cache write appends
    chunk 2's K/V, cache read materializes the full (chunk 1 + chunk 2)
    K/V for this rank, mesh comm operates on it, and the kernel call uses
    ``kv_len > qo_len`` (chunk 2 attends to both chunks).  Concatenating
    the two chunks' outputs in cyclic-shard order must match the
    single-shot ATTN2D / full-causal SDPA reference.

    chunk_offset=0 picks a P-aligned boundary; chunk_offset=1 picks a
    boundary off by one token, which stresses variable-size allgatherv +
    a non-zero L_prev mod P in the chunk 2 forward.

    L_extra=0 keeps L divisible by P (uniform per-rank shard sizes);
    L_extra=1 makes L mod P == 1, so rank 0 has one extra token and the
    backend must derive a consistent global total via the last-chunk snap
    rather than multiplying a local count by P.
    """
    pytest.importorskip("flashinfer")
    P = R * C
    if torch.cuda.device_count() < P:
        pytest.skip(f"needs {P} CUDA devices, have {torch.cuda.device_count()}")

    L_base = ((max(64, 8 * P) + P - 1) // P) * P
    L = L_base + L_extra
    L_chunk_1 = L // 2 + chunk_offset
    head_dim = 64
    dtype_name = "bfloat16"
    seed = 0xC04ED + chunk_offset + L_extra * 0x100
    num_heads, num_kv_heads = 4, 4

    args = (P, R, C, L, L_chunk_1, num_heads, num_kv_heads, head_dim, dtype_name, seed)
    with MPIPoolExecutor(max_workers=P) as ex:
        results = ex.map(_entrypoint_chunked, *zip(*[args] * P))
        for r in results:
            assert r is True
