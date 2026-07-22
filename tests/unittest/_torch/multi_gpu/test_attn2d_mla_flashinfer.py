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
"""Multi-GPU correctness test for Attn2DFlashInferAttention in MLA mode.

Companion to ``test_attn2d_flashinfer.py`` (the MHA/GQA suite).  Exercises
the absorbed (MQA-over-latent) MLA path: ``q`` is the fused query of width
``kv_lora_rank + qk_rope_head_dim``, the K/V for the new tokens is the
single compressed-latent head ``[compressed_kv | k_pe]`` delivered via
``forward_args.latent_cache``, and V is the leading ``kv_lora_rank`` slice
of that latent (no separate V collective, single-tensor mesh gather).

Spawns ``R*C`` ranks via MPI, runs the 2D-mesh CP backend on a cyclic
shard, all-gathers the result, and compares to a single-GPU absorbed-MLA
reference (causal MQA over the unsharded latent, head_dim_qk != head_dim_vo,
sm_scale = 1/sqrt(qk_nope+qk_rope)).  This is the backend-level absorbed
output (before the module's v_b_proj up-projection).
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


# MLA dims for the test.  D_qk (latent width) = kv_lora_rank + qk_rope_head_dim;
# D_vo (absorbed output width) = kv_lora_rank; sm_scale uses qk_nope+qk_rope.
KV_LORA_RANK = 128
QK_ROPE_HEAD_DIM = 64
QK_NOPE_HEAD_DIM = 64
Q_LORA_RANK = 256  # unused by the ATTN2D backend, only satisfies create_attention
Q_SCALING = 1.0
D_QK = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 192
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM  # 128


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


def _mla_sm_scale() -> float:
    import math

    return 1.0 / (math.sqrt(QK_HEAD_DIM) * Q_SCALING)


def _reference_absorbed_mla(
    fused_q: torch.Tensor,
    latent: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    """Single-GPU absorbed-MLA reference on the unsharded fused_q / latent.

    ``fused_q``: ``[L, num_heads * D_qk]``; ``latent``: ``[L, D_qk]`` (single
    latent head).  Causal MQA: every query head attends over the one latent
    head; V is the leading ``kv_lora_rank`` slice.  Returns
    ``[L, num_heads, kv_lora_rank]``.
    """
    L = fused_q.shape[0]
    q = fused_q.view(L, num_heads, D_QK).transpose(0, 1).contiguous()  # [H, L, D_qk]
    k = (
        latent.view(L, 1, D_QK).expand(L, num_heads, D_QK).transpose(0, 1).contiguous()
    )  # [H, L, D_qk]
    v = (
        latent[:, :KV_LORA_RANK]
        .view(L, 1, KV_LORA_RANK)
        .expand(L, num_heads, KV_LORA_RANK)
        .transpose(0, 1)
        .contiguous()
    )  # [H, L, kv_lora_rank]
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=True, scale=_mla_sm_scale()
    )  # [H, L, kv_lora_rank]
    return out.transpose(0, 1).contiguous()  # [L, H, kv_lora_rank]


def _gather_full_output(
    out_local: torch.Tensor,
    *,
    L: int,
    L_max: int,
    hidden: int,
    P: int,
    mapping: Mapping,
) -> torch.Tensor:
    """Pad to ``L_max``, cp-allgather, reassemble into absolute-position order.

    Returns a ``[L, hidden]`` tensor on every rank (caller compares on rank 0).
    """
    L_local = out_local.shape[0]
    if L_local < L_max:
        pad = out_local.new_zeros(L_max - L_local, hidden)
        out_local_padded = torch.cat([out_local, pad], dim=0)
    else:
        out_local_padded = out_local
    gathered = cp_allgather(out_local_padded.contiguous(), mapping=mapping, dim=0)
    return gathered.view(P, L_max, hidden).permute(1, 0, 2).reshape(L_max * P, hidden)[:L]


def _make_mla_backend(num_heads: int):
    from tensorrt_llm._torch.attention_backend.utils import create_attention

    return create_attention(
        "ATTN2D",
        layer_idx=0,
        num_heads=num_heads,
        head_dim=D_QK,
        num_kv_heads=1,
        q_scaling=Q_SCALING,
        is_mla_enable=True,
        q_lora_rank=Q_LORA_RANK,
        kv_lora_rank=KV_LORA_RANK,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        v_head_dim=KV_LORA_RANK,  # absorbed output width for standard MLA
    )


# ---------------------------------------------------------------------------
# Single-request (single-shot, no cache) check
# ---------------------------------------------------------------------------


@torch.inference_mode()
def _run_attn2d_mla_check(
    world_size: int,
    rank: int,
    R: int,
    C: int,
    L: int,
    num_heads: int,
    dtype_name: str,
    seed: int,
) -> None:
    from tensorrt_llm._torch.attention_backend.attn2d_flashinfer import (
        Attn2DFlashInferAttentionMetadata,
    )
    from tensorrt_llm._torch.attention_backend.interface import (
        AttentionForwardArgs,
        PredefinedAttentionMask,
    )

    dtype = getattr(torch, dtype_name)
    P = R * C
    assert world_size == P
    L_local = (L - rank + P - 1) // P
    L_max = (L + P - 1) // P
    out_hidden = num_heads * KV_LORA_RANK

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    mapping = _build_mapping(rank, P, R, C)
    cp_rank = mapping.cp_rank

    # Deterministic fused_q / latent (same on every rank via a CPU generator).
    gen = torch.Generator().manual_seed(seed)
    fused_q_full = torch.randn(L, num_heads * D_QK, dtype=dtype, generator=gen).to(device)
    latent_full = torch.randn(L, D_QK, dtype=dtype, generator=gen).to(device)

    # Cyclic shard: rank r owns positions {p: p%P == r}.
    q_local = fused_q_full[cp_rank::P].contiguous()
    latent_local = latent_full[cp_rank::P].contiguous()
    assert q_local.shape[0] == L_local

    backend = _make_mla_backend(num_heads)
    metadata = Attn2DFlashInferAttentionMetadata(
        max_num_requests=1,
        max_num_tokens=L_local,
        seq_lens=torch.tensor([L_local], dtype=torch.int32),
        num_contexts=1,
        mapping=mapping,
        total_input_lens=torch.tensor([L], dtype=torch.int32),
    )
    metadata.prepare()
    forward_args = AttentionForwardArgs(
        attention_mask=PredefinedAttentionMask.CAUSAL,
        latent_cache=latent_local,
    )

    out_local = backend.forward(q_local, None, None, metadata, forward_args=forward_args)
    assert out_local.shape[0] == L_local
    assert out_local.shape[1] == out_hidden

    out_full = _gather_full_output(
        out_local, L=L, L_max=L_max, hidden=out_hidden, P=P, mapping=mapping
    )

    if rank == 0:
        out_full = out_full.view(L, num_heads, KV_LORA_RANK)
        out_ref = _reference_absorbed_mla(fused_q_full, latent_full, num_heads)
        atol = 2e-2 if dtype == torch.bfloat16 else 1e-3
        rtol = 1e-2 if dtype == torch.bfloat16 else 1e-3
        torch.testing.assert_close(out_full, out_ref, atol=atol, rtol=rtol)


def _entrypoint(world_size, R, C, L, num_heads, dtype_name, seed):
    rank = tensorrt_llm.mpi_rank()
    try:
        _run_attn2d_mla_check(world_size, rank, R, C, L, num_heads, dtype_name, seed)
    except Exception:
        traceback.print_exc()
        raise
    return True


# (R, C) coverage mirrors the MHA suite (capped at <= 8 GPUs):
#   (2, 2) square, (2, 4)/(4, 2) Q-split/K-split s=2, (1, 4)/(4, 1) fast paths.
@pytest.mark.parametrize("R,C", [(2, 2), (2, 4), (4, 2), (1, 4), (4, 1)])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("L_extra", [0, 1])
def test_attn2d_mla_flashinfer_matches_absorbed_reference(R, C, num_heads, L_extra):
    """MLA-mode backend output across R*C GPUs matches single-GPU absorbed MLA."""
    pytest.importorskip("flashinfer")
    P = R * C
    if torch.cuda.device_count() < P:
        pytest.skip(f"needs {P} CUDA devices, have {torch.cuda.device_count()}")

    L_base = ((max(64, 8 * P) + P - 1) // P) * P
    L = L_base + L_extra
    dtype_name = "bfloat16"
    seed = 0x5EED

    args = (P, R, C, L, num_heads, dtype_name, seed)
    with MPIPoolExecutor(max_workers=P) as ex:
        results = ex.map(_entrypoint, *zip(*[args] * P))
        for r in results:
            assert r is True


# ---------------------------------------------------------------------------
# Chunked-prefill (latent cache write + read) check
# ---------------------------------------------------------------------------


class _MockLatentKVCacheManager:
    """Minimal per-rank MLA latent cache manager (kv_factor=1, single head).

    NHD latent layout matches ``append_mla_latent_cache`` and the backend's
    ``_materialize_latent_from_pages``: ``(max_pages, 1, page_size, 1, D_qk)``.
    """

    def __init__(
        self,
        *,
        page_size: int,
        max_pages: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.tokens_per_block = page_size
        self.blocks_in_primary_pool = max_pages
        self._buffer = torch.zeros(
            (max_pages, 1, page_size, 1, head_dim),
            dtype=dtype,
            device=device,
        )
        self._req_pages = {}
        self._next_page = 0

    def get_buffers(self, layer_idx: int, kv_layout: str = "NHD") -> torch.Tensor:
        assert kv_layout == "NHD", "test mock only supports NHD layout"
        return self._buffer

    def get_batch_cache_indices(self, request_ids, layer_idx=None):
        return [self._req_pages[req_id] for req_id in request_ids]

    def allocate_pages(self, req_id: int, num_pages: int) -> None:
        assert self._next_page + num_pages <= self.blocks_in_primary_pool
        self._req_pages[req_id] = list(range(self._next_page, self._next_page + num_pages))
        self._next_page += num_pages


@torch.inference_mode()
def _run_attn2d_mla_chunked_check(
    world_size: int,
    rank: int,
    R: int,
    C: int,
    L: int,
    L_chunk_1: int,
    num_heads: int,
    dtype_name: str,
    seed: int,
) -> None:
    from tensorrt_llm._torch.attention_backend.attn2d_flashinfer import (
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
    assert L_chunk_1 > 0 and L_chunk_2 > 0

    L_local = (L - rank + P - 1) // P
    if rank < L_chunk_1:
        L_local_chunk_1 = (L_chunk_1 - rank + P - 1) // P
    else:
        L_local_chunk_1 = 0
    L_local_chunk_2 = L_local - L_local_chunk_1
    assert L_local_chunk_2 >= 0
    out_hidden = num_heads * KV_LORA_RANK

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    mapping = _build_mapping(rank, P, R, C)
    cp_rank = mapping.cp_rank

    gen = torch.Generator().manual_seed(seed)
    fused_q_full = torch.randn(L, num_heads * D_QK, dtype=dtype, generator=gen).to(device)
    latent_full = torch.randn(L, D_QK, dtype=dtype, generator=gen).to(device)

    q_local_full = fused_q_full[cp_rank::P].contiguous()
    latent_local_full = latent_full[cp_rank::P].contiguous()

    q_local_c1 = q_local_full[:L_local_chunk_1].contiguous()
    latent_local_c1 = latent_local_full[:L_local_chunk_1].contiguous()
    q_local_c2 = q_local_full[L_local_chunk_1:].contiguous()
    latent_local_c2 = latent_local_full[L_local_chunk_1:].contiguous()

    page_size = max(8, L_local)
    num_pages = (L_local + page_size - 1) // page_size
    max_pages = max(num_pages * 4, 8)
    req_id = 0
    cache_mgr = _MockLatentKVCacheManager(
        page_size=page_size,
        max_pages=max_pages,
        head_dim=D_QK,
        dtype=dtype,
        device=device,
    )
    cache_mgr.allocate_pages(req_id, num_pages)

    backend = _make_mla_backend(num_heads)

    def _build_metadata(seq_lens_val, total, chunk, cached_local):
        md = Attn2DFlashInferAttentionMetadata(
            max_num_requests=1,
            max_num_tokens=L_local,
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

    # --- Forward 1: chunk 1 (L_prev = 0). ---
    md1 = _build_metadata(L_local_chunk_1, L_chunk_1, L_chunk_1, 0)
    fa1 = AttentionForwardArgs(
        attention_mask=PredefinedAttentionMask.CAUSAL, latent_cache=latent_local_c1
    )
    out_c1 = backend.forward(q_local_c1, None, None, md1, forward_args=fa1)
    assert out_c1.shape[0] == L_local_chunk_1

    # --- Forward 2: chunk 2 (L_prev = L_chunk_1); cache holds chunk 1. ---
    md2 = _build_metadata(L_local_chunk_2, L, L_chunk_2, L_local_chunk_1)
    fa2 = AttentionForwardArgs(
        attention_mask=PredefinedAttentionMask.CAUSAL, latent_cache=latent_local_c2
    )
    out_c2 = backend.forward(q_local_c2, None, None, md2, forward_args=fa2)
    assert out_c2.shape[0] == L_local_chunk_2

    out_local = torch.cat([out_c1, out_c2], dim=0)
    assert out_local.shape[0] == L_local

    out_full = _gather_full_output(
        out_local, L=L, L_max=(L + P - 1) // P, hidden=out_hidden, P=P, mapping=mapping
    )
    if rank == 0:
        out_full = out_full.view(L, num_heads, KV_LORA_RANK)
        out_ref = _reference_absorbed_mla(fused_q_full, latent_full, num_heads)
        atol = 2e-2 if dtype == torch.bfloat16 else 1e-3
        rtol = 1e-2 if dtype == torch.bfloat16 else 1e-3
        torch.testing.assert_close(out_full, out_ref, atol=atol, rtol=rtol)


def _entrypoint_chunked(world_size, R, C, L, L_chunk_1, num_heads, dtype_name, seed):
    rank = tensorrt_llm.mpi_rank()
    try:
        _run_attn2d_mla_chunked_check(
            world_size, rank, R, C, L, L_chunk_1, num_heads, dtype_name, seed
        )
    except Exception:
        traceback.print_exc()
        raise
    return True


@pytest.mark.parametrize("R,C", [(2, 2), (4, 2), (2, 4)])
@pytest.mark.parametrize("chunk_offset", [0, 1])
@pytest.mark.parametrize("L_extra", [0, 1])
def test_attn2d_mla_flashinfer_chunked_prefill(R, C, chunk_offset, L_extra):
    """Two-chunk MLA prefill (latent cache write + read) matches single-shot."""
    pytest.importorskip("flashinfer")
    P = R * C
    if torch.cuda.device_count() < P:
        pytest.skip(f"needs {P} CUDA devices, have {torch.cuda.device_count()}")

    L_base = ((max(64, 8 * P) + P - 1) // P) * P
    L = L_base + L_extra
    L_chunk_1 = L // 2 + chunk_offset
    dtype_name = "bfloat16"
    seed = 0xC04ED + chunk_offset + L_extra * 0x100
    num_heads = 8

    args = (P, R, C, L, L_chunk_1, num_heads, dtype_name, seed)
    with MPIPoolExecutor(max_workers=P) as ex:
        results = ex.map(_entrypoint_chunked, *zip(*[args] * P))
        for r in results:
            assert r is True
