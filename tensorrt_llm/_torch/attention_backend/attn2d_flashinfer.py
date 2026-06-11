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
"""FlashInfer-based ATTN2D context-parallel attention backend.

Implements the 2D-mesh attention from the Attention2D paper
(https://arxiv.org/abs/2503.15758), adapted for LLM prefill with a
causal mask. Tokens are distributed cyclically across the flat cp_size,
and the cp_group is interpreted as a row_size x col_size mesh in
column-major order (cp_rank = col_idx * R + row_idx).

For each request, this backend:

  1. all-gathers Q within the row group (size C) -> [L/R, H_q, D]
  2. permutes K/V across cp_group from column-major to row-major cyclic
     so the subsequent col-gather produces a clean stride-C partition
     (col=col_idx owns positions ``{p: p%C == col_idx}``)
  3. all-gathers K/V within the col group (size R) -> [L/C, H_kv, D]
     (K and V are packed into a single fused collective)
  4. computes local attention via flashinfer with log-sum-exp output,
     using a position-sorted layout and ``causal=True`` (no custom
     mask) -- see mesh-shape constraint below
  5. scatters the L/R partials back to their original owners via
     all_to_all_single within the row group, then LSE-merges the C
     partials this rank receives -> final output for L/P tokens

Mesh aspect is constrained to ``R % C == 0`` or ``C % R == 0``;
non-divisible meshes (e.g. ``R=3, C=2``, ``R=4, C=6``) are rejected.
Re-sorting Q and K/V into ascending absolute position gives positions
``q_pos[m] = m*R + row_idx`` and ``k_pos[k] = k*C + col_idx``, so the
``k_pos <= q_pos`` mask becomes ``(k - m) * C <= row_idx - col_idx``
(square mesh) or a stair-step that the divisibility constraint
decomposes cleanly:

  * **Square mesh** (``R == C``): ``k <= m`` for ``col_idx <= row_idx``
    (causal) and ``k < m`` for ``col_idx > row_idx`` (strict causal).
  * **R < C, with ``C % R == 0``** (s = C/R, "Q-split"): split sorted
    Q into ``s`` stride-s sub-tensors.  Each sub-tensor sees the full
    K and produces a clean causal or strict-causal pattern.  Outputs
    cover disjoint Q positions, so no LSE merge is needed across
    sub-tensors -- they assemble directly into the source-rank-grouped
    layout used by the row-group all_to_all.
  * **R > C, with ``R % C == 0``** (s = R/C, "K-split"): split sorted
    K into ``s`` stride-s sub-tensors.  Each call runs against the
    full Q and produces a (strict-)causal pattern, but each output is
    a *partial* attention over a K-shard, so the ``s`` partials are
    LSE-merged via ``flashinfer.merge_states``.

Strict causal (``k < i``) on equal-length tensors is expressed by
appending one dummy Q row so ``qo_len = L_q + 1`` and ``kv_len = L_q``;
FlashInfer's bottom-right ``causal=True`` then applies ``k <= i - 1``
on the real rows, and the dummy output / LSE are sliced off.

KV cache, generation phase, fused QKV, sparse / sliding-window masks,
and mixed-dtype quantization are out of scope for v0.  Inputs are
assumed to be cyclic-sharded by the caller.

For requests whose total length is not divisible by ``cp_size``, the
backend pads each rank's local shard up to ``ceil(L_total / cp_size)``
with zero rows so every collective stays uniform-shape.  Padding tokens
land at absolute positions ``>= L_total``, so the causal mask
``p_k <= p_q`` excludes them from every real-Q output (both Q-split and
K-split paths handle this naturally).  Padding Q outputs are sliced off
at exit.  The total length is read from ``metadata.total_input_lens``;
if absent the backend assumes divisibility.
"""

from dataclasses import dataclass, field
from itertools import accumulate
from typing import List, Optional, Tuple

import flashinfer
import torch

from tensorrt_llm._torch.distributed import (
    attn2d_col_allgather,
    attn2d_row_allgather,
    attn2d_row_alltoall,
    permute_send_recv,
)
from tensorrt_llm.models.modeling_utils import QuantConfig

from .interface import (
    AttentionBackend,
    AttentionForwardArgs,
    AttentionMetadata,
    PredefinedAttentionMask,
    merge_attention_forward_args,
)


@dataclass(kw_only=True)
class Attn2DFlashInferAttentionMetadata(AttentionMetadata):
    """Metadata for the FlashInfer-based Attn2D backend.

    Each rank receives a cyclic shard of every request's tokens:
    rank ``cp_rank`` owns positions ``{p: p % cp_size == cp_rank}``.
    ``seq_lens`` is the per-request shard length on this rank for the
    new (current-chunk) tokens; ``cached_lens_local`` is the per-rank
    count of previously cached tokens for the same request.

    ``total_input_lens`` is the per-request cumulative K-range length
    (``L_total = L_prev + L_chunk``, same value on every rank), NOT the
    full conversation length.  For initial-chunk prefill, L_prev = 0 so
    L_total equals the prompt length; for chunked / multi-turn prefill
    on iteration N, L_total = (prior chunks total) + (this chunk size).
    The kernel mask only cares about positions in [0, L_total), and the
    cyclic counts derived from L_total must match this rank's actual
    K (cached + new) shard -- using the full conversation length here
    would leak future-chunk positions into the size math.

    ``chunk_input_lens`` is the per-request current-chunk length
    (``L_chunk``, same value on every rank).  ``L_chunk == L_total``
    means initial-chunk prefill (no cached K/V); ``L_chunk < L_total``
    means chunked / multi-turn prefill with ``L_prev = L_total -
    L_chunk`` cached K/V tokens globally.  The FlashInfer kernel sees
    ``kv_len > qo_len`` for chunked-prefill calls and the bottom-right
    causal mask absorbs the ``L_prev`` shift cleanly.  If
    ``chunk_input_lens`` is ``None`` the backend treats every request
    as initial-chunk (``L_chunk == L_total``).

    ``total_input_lens`` must be set via ``update_attn2d_param`` before
    ``forward()``; ``None`` triggers an assertion error.  A per-rank fallback
    (``local_len * P``) would silently produce different ``L_total`` values
    on different ranks whenever ``L_total % cp_size != 0``, breaking
    the mesh-comm size derivation.

    Paged-KV state (``paged_kv_indices``, ``paged_kv_indptr``,
    ``paged_kv_last_page_len``, ``workspace_buffer``) is allocated in
    ``__post_init__`` when ``kv_cache_manager`` is set, and populated
    in ``prepare()`` from the manager's per-request block IDs.  These
    fields are not consumed by the compute path yet -- chunked /
    multi-turn prefill arrives in a later phase -- but the metadata is
    plumbed end-to-end so the engine can route ATTN2D requests through
    the cached path.
    """

    total_input_lens: Optional[torch.Tensor] = None

    # Per-request current-chunk length (global L_chunk, same on every rank).
    # ``None`` means initial-chunk prefill (L_chunk == L_total).
    chunk_input_lens: Optional[torch.Tensor] = None

    # Per-request count of cached tokens on this rank (cyclic-shard view).
    # Derived in ``prepare()`` from ``kv_cache_params.num_cached_tokens_per_seq``.
    cached_lens_local: Optional[torch.Tensor] = None

    # FlashInfer paged-KV workspace; sized once in ``__post_init__``.
    workspace_buffer: Optional[torch.Tensor] = None

    # Stable buffers populated in ``prepare()`` from ``kv_cache_manager``.
    # Allocated in ``__post_init__`` when a cache manager is configured;
    # left unset when ``kv_cache_manager is None`` (v0 no-cache path).
    # ``repr=False`` so ``__repr__`` doesn't crash before allocation.
    paged_kv_indices: torch.Tensor = field(init=False, repr=False)
    paged_kv_indptr: torch.Tensor = field(init=False, repr=False)
    paged_kv_last_page_len: torch.Tensor = field(init=False, repr=False)
    num_blocks_per_request: List[int] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kv_cache_manager is None:
            return
        max_num_pages = self.kv_cache_manager.blocks_in_primary_pool
        self.paged_kv_indices = torch.empty((max_num_pages,), dtype=torch.int32, device="cuda")
        self.paged_kv_indptr = torch.empty(
            (self.max_num_requests + 1,), dtype=torch.int32, device="cuda"
        )
        self.paged_kv_last_page_len = torch.empty(
            (self.max_num_requests,), dtype=torch.int32, device="cuda"
        )
        if self.workspace_buffer is None:
            # 256 MB; matches the conservative end of FlashInfer's prefill
            # workspace recommendation.  Tighten if profiling shows headroom.
            self.workspace_buffer = torch.empty(
                (256 * 1024 * 1024,), dtype=torch.uint8, device="cuda"
            )

    def prepare(self) -> None:
        super().prepare()
        if self.kv_cache_manager is None:
            # No cache configured: v0 prefill path.  Leave paged-KV fields
            # untouched (they were not allocated); cached_lens_local stays None
            # so the backend treats every request as initial-chunk.
            self.cached_lens_local = None
            self.num_blocks_per_request = []
            return

        n = self.num_seqs
        use_cache = self.kv_cache_params is not None and self.kv_cache_params.use_cache
        cached_per_seq = (
            list(self.kv_cache_params.num_cached_tokens_per_seq)
            if use_cache and self.kv_cache_params.num_cached_tokens_per_seq is not None
            else [0] * n
        )
        assert len(cached_per_seq) == n, (
            f"num_cached_tokens_per_seq has {len(cached_per_seq)} entries, expected {n} (num_seqs)"
        )
        self.cached_lens_local = torch.tensor(cached_per_seq, dtype=torch.int32)

        seq_lens_list = self.seq_lens.tolist() if self.seq_lens is not None else [0] * n
        # kv_lens_local = this rank's cached + new K/V count for each request.
        kv_lens_local = [c + s for c, s in zip(cached_per_seq, seq_lens_list)]

        page_size = self.kv_cache_manager.tokens_per_block
        self.num_blocks_per_request = [
            (kv_len + page_size - 1) // page_size for kv_len in kv_lens_local
        ]

        # paged_kv_indices: flatten the per-request block lists in batch order.
        # block_ids_per_seq[i] is rank-local: each rank's cache manager only
        # tracks its own slice of the cyclic-sharded cache.
        assert self.request_ids is not None, (
            "request_ids must be set on metadata before prepare() when a KV "
            "cache manager is configured"
        )
        block_ids_per_seq = self.kv_cache_manager.get_batch_cache_indices(self.request_ids)
        indices_list: List[int] = []
        for i, block_ids in enumerate(block_ids_per_seq):
            indices_list.extend(block_ids[: self.num_blocks_per_request[i]])
        if indices_list:
            indices_tensor = torch.tensor(indices_list, dtype=torch.int32)
            self.paged_kv_indices[: indices_tensor.size(0)].copy_(indices_tensor, non_blocking=True)

        # paged_kv_indptr: cumsum of per-request block counts (prefix 0).
        indptr_list = [0]
        for nb in self.num_blocks_per_request:
            indptr_list.append(indptr_list[-1] + nb)
        indptr_tensor = torch.tensor(indptr_list, dtype=torch.int32)
        self.paged_kv_indptr[: indptr_tensor.size(0)].copy_(indptr_tensor, non_blocking=True)

        # paged_kv_last_page_len: tokens in the last page per request.
        last_page_lens = [
            (kv_len - (nb - 1) * page_size) if nb > 0 else 0
            for kv_len, nb in zip(kv_lens_local, self.num_blocks_per_request)
        ]
        last_page_tensor = torch.tensor(last_page_lens, dtype=torch.int32)
        self.paged_kv_last_page_len[: last_page_tensor.size(0)].copy_(
            last_page_tensor, non_blocking=True
        )

    def update_attn2d_param(
        self,
        total_input_lens: Optional[List[int]],
        chunk_input_lens: Optional[List[int]] = None,
    ) -> None:
        """Set per-request total / chunk input lengths for the ATTN2D backend.

        Called from the model engine's prepare-inputs path once per
        batch, alongside ``seq_lens`` (mirrors ``update_helix_param``).

        Args:
            total_input_lens: per-request total (un-sharded) input
                length, one entry per request in the batch.  When
                ``None`` (or omitted by the engine) the backend falls
                back to the divisible-L assumption.
            chunk_input_lens: per-request current-chunk length (global
                L_chunk).  When ``None`` the backend treats every
                request as initial-chunk (``L_chunk == L_total``).
        """
        if total_input_lens is None:
            self.total_input_lens = None
        else:
            self.total_input_lens = torch.tensor(total_input_lens, dtype=torch.int32)

        if chunk_input_lens is None:
            self.chunk_input_lens = None
        else:
            self.chunk_input_lens = torch.tensor(chunk_input_lens, dtype=torch.int32)


def _cyclic_count(N: int, P: int, r: int) -> int:
    """Count of positions ``{p : p%P == r, 0 <= p < N}``."""
    if r >= N or N <= 0:
        return 0
    return (N - r + P - 1) // P


def _first_chunk_pos(L_prev: int, P: int, r: int) -> int:
    """Smallest position p with ``p%P == r`` and ``p >= L_prev``."""
    return L_prev + ((r - L_prev) % P)


def _build_q_sort_idx(
    L_chunk: int,
    L_prev: int,
    R: int,
    C: int,
    row_idx: int,
    new_counts_in_row: List[int],
    device: torch.device,
) -> torch.Tensor:
    """Build sort_idx[m] = flat row-gather index for sorted-Q absolute position m.

    Row-gather output is rank-major: rank ``c`` (cp_rank = c*R + row_idx)
    contributes ``new_counts_in_row[c]`` entries.  Sorted Q is in absolute
    position order with stride R starting at first_q = L_prev + ((row_idx -
    L_prev) mod R).  This helper precomputes the permutation that maps
    sorted index m to the flat row-gather index where that token lives.

    Returns a CUDA int64 tensor of shape ``(sum(new_counts_in_row),)``.
    """
    P = R * C
    L_new_per_row = sum(new_counts_in_row)
    if L_new_per_row == 0:
        return torch.empty(0, dtype=torch.int64, device=device)
    first_q = L_prev + ((row_idx - L_prev) % R)
    m = torch.arange(L_new_per_row, dtype=torch.int64)
    p = first_q + m * R
    c = ((p % P) - row_idx) // R
    cp_rank_c = c * R + row_idx
    first_pos_c = L_prev + ((cp_rank_c - L_prev) % P)
    j = (p - first_pos_c) // P
    offsets = torch.tensor([0] + list(new_counts_in_row[:-1]), dtype=torch.int64).cumsum(0)
    return (offsets[c] + j).to(device)


def _build_k_sort_idx(
    L_total: int,
    R: int,
    C: int,
    col_idx: int,
    total_counts_in_col: List[int],
    device: torch.device,
) -> torch.Tensor:
    """Build sort_idx[k] = flat col-gather index for sorted-K absolute position k.

    Post-redistribute col-gather output is rank-major in the col group:
    rank at row_idx_in_col contributes total_counts_in_col[row_idx_in_col]
    entries (= L_local_total at source rank ``row_idx_in_col*C + col_idx``).
    Sorted K is in absolute position order with stride C starting at col_idx.

    Returns a CUDA int64 tensor of shape ``(sum(total_counts_in_col),)``.
    """
    P = R * C
    L_k_per_col = sum(total_counts_in_col)
    if L_k_per_col == 0:
        return torch.empty(0, dtype=torch.int64, device=device)
    k = torch.arange(L_k_per_col, dtype=torch.int64)
    p = col_idx + k * C
    # After redistribute, rank at row_idx_in_col holds positions with
    # p%P == row_idx_in_col*C + col_idx.  Solve for row_idx_in_col:
    row_idx_in_col = ((p % P) - col_idx) // C
    # Within that rank, local index j: positions are
    # row_idx_in_col*C + col_idx + j*P, so j = (p - that_first_pos) // P.
    first_pos = row_idx_in_col * C + col_idx
    j = (p - first_pos) // P
    offsets = torch.tensor([0] + list(total_counts_in_col[:-1]), dtype=torch.int64).cumsum(0)
    return (offsets[row_idx_in_col] + j).to(device)


def _redistribute_kv_to_row_major(
    kv: torch.Tensor,
    *,
    R: int,
    C: int,
    cp_rank: int,
    cp_group: List[int],
    recv_count: Optional[int] = None,
) -> torch.Tensor:
    """Permute packed K/V across cp_group: column-major -> row-major cyclic.

    Input ``kv`` is the packed tensor of shape ``(L_local, 2, H_kv, D)``
    with ``kv[:, 0]`` = K and ``kv[:, 1]`` = V.  The varying dim is dim 0
    so asymmetric P2P (different shard counts at paired ranks for chunked
    prefill) can override the recv tensor's dim 0 via ``recv_count``.
    Returns the permuted packed tensor (shape ``(recv_count, 2, H_kv, D)``
    when set, else same shape as ``kv``) -- either ``kv`` itself when the
    permutation is the identity, or a freshly allocated buffer filled
    via P2P from the source peer.

    Initial state: rank ``r`` holds positions ``{p: p%P == r}`` (cp_rank
    is laid out as ``r = col_idx*R + row_idx``).  After this permutation,
    rank ``r`` holds positions ``{p: p%P == (r%R)*C + (r//R)}``, i.e.
    each rank holds the K/V it would have under a row-major cyclic
    distribution.  The subsequent col-gather then yields the clean
    stride-C partition ``K_{col=col_idx} = {p: p%C == col_idx}``,
    mirroring Q's stride-R partition from the row-gather and producing a
    regular causal mask pattern that optimized kernels can exploit.

    Each rank participates in at most one send + recv pair via the
    rank-list-keyed ``permute_send_recv`` op (NCCL group send/recv
    bootstrapped through the TRT-LLM comm pool over MPI).  The
    permutation is a mesh-transpose, so for R != C it is not an
    involution; ``source = (r%R)*C + (r//R)`` is the inverse of
    ``target = (r%C)*R + (r//C)``.

    Main-diagonal ranks (``target == cp_rank``, e.g.
    ``col_idx == row_idx`` for ``R == C``) still call the op with self
    as both peer ranks -- ``ncclSend``/``ncclRecv`` to self inside a
    ``ncclGroupStart/End`` block executes as a memcpy.  This is
    required for collective correctness: ``getComm(cp_group)`` is a
    collective bootstrap that needs every rank in ``cp_group`` to
    participate on the first call, even if a particular rank's
    permutation is the identity.  Skipping the call on diagonal ranks
    leaves the comm half-bootstrapped and deadlocks the others.

    The exception is ``C == 1`` (and equivalently any mesh where every
    rank is diagonal): no rank calls the op, the comm is never built,
    nothing else in the backend uses ``cp_group``, so no participation
    is needed.
    """
    target = (cp_rank % C) * R + (cp_rank // C)
    source = (cp_rank % R) * C + (cp_rank // R)
    if C == 1:
        # Every rank is diagonal -- skip the op entirely so no comm gets
        # bootstrapped.  Safe because nothing else in this backend uses
        # cp_group.
        return kv
    return permute_send_recv(
        kv.contiguous(),
        target_rank=cp_group[target],
        source_rank=cp_group[source],
        group=cp_group,
        recv_count=recv_count,
    )


class Attn2DFlashInferAttention(AttentionBackend[Attn2DFlashInferAttentionMetadata]):
    """FlashInfer-based 2D-mesh CP attention backend (prefill / causal only)."""

    Metadata = Attn2DFlashInferAttentionMetadata

    def __init__(
        self,
        layer_idx: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: Optional[int] = None,
        quant_config: Optional[QuantConfig] = None,
        **kwargs,
    ):
        super().__init__(
            layer_idx,
            num_heads,
            head_dim,
            num_kv_heads=num_kv_heads,
            quant_config=quant_config,
            **kwargs,
        )
        # Lazily allocated on first use (needs CUDA device).
        self._ragged_prefill_wrapper: Optional[flashinfer.BatchPrefillWithRaggedKVCacheWrapper] = (
            None
        )

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: Attn2DFlashInferAttentionMetadata,
        forward_args: Optional[AttentionForwardArgs] = None,
        **kwargs,
    ) -> torch.Tensor:
        forward_args = merge_attention_forward_args(forward_args, kwargs)

        if forward_args.attention_mask != PredefinedAttentionMask.CAUSAL:
            raise NotImplementedError(
                f"Attn2DFlashInferAttention only supports CAUSAL mask, got "
                f"{forward_args.attention_mask}."
            )
        if k is None or v is None:
            raise NotImplementedError(
                "Attn2DFlashInferAttention requires separate K and V inputs "
                "(fused QKV not supported in v0)."
            )

        mapping = metadata.mapping
        assert mapping is not None and mapping.has_cp_attn2d(), (
            "Attn2DFlashInferAttention requires CP type ATTN2D."
        )
        R = mapping.attn2d_row_size
        C = mapping.attn2d_col_size
        assert R % C == 0 or C % R == 0, (
            f"Attn2DFlashInferAttention requires R % C == 0 or C % R == 0, "
            f"got R={R}, C={C}.  Non-divisible meshes (e.g. 3x2, 4x6) are "
            f"not supported -- pick a mesh where one dim divides the other."
        )
        P = mapping.cp_size
        cp_rank = mapping.cp_rank
        row_idx = mapping.attn2d_row_rank
        col_idx = mapping.attn2d_col_rank

        H_q = self.num_heads
        H_kv = self.num_kv_heads
        D = self.head_dim

        q = q.view(-1, H_q, D)
        k = k.view(-1, H_kv, D)
        v = v.view(-1, H_kv, D)

        if metadata.kv_cache_manager is not None:
            self._append_new_kv_to_cache(k, v, metadata)

        seq_lens = metadata.seq_lens.tolist()
        assert metadata.total_input_lens is not None, (
            "Attn2DFlashInferAttentionMetadata.total_input_lens must be set "
            "before forward() -- call update_attn2d_param() from the engine's "
            "prepare-inputs path."
        )
        total_lens = metadata.total_input_lens.tolist()
        assert len(total_lens) == len(seq_lens)
        if metadata.chunk_input_lens is not None:
            chunk_lens = metadata.chunk_input_lens.tolist()
        else:
            chunk_lens = list(total_lens)
        assert len(chunk_lens) == len(seq_lens)
        if metadata.cached_lens_local is not None:
            cached_lens_list = metadata.cached_lens_local.tolist()
        else:
            cached_lens_list = [0] * len(seq_lens)

        has_cache = metadata.kv_cache_manager is not None
        B = len(seq_lens)
        device = q.device

        # ------------------------------------------------------------------ #
        # Per-request cyclic-count metadata (CPU, cheap).                     #
        # ------------------------------------------------------------------ #
        total_counts_all: List[List[int]] = []  # [B][P]
        sizes_row_all: List[List[int]] = []  # [B][C]  new-Q counts in row group
        sizes_col_all: List[List[int]] = []  # [B][R]  total-K counts in col group
        L_q_per_row_all: List[int] = []  # [B]  sum(sizes_row_all[i])
        L_k_per_col_all: List[int] = []  # [B]  sum(sizes_col_all[i])
        L_prev_all: List[int] = []  # [B]
        L_local_total_all: List[int] = []  # [B]
        local_new_len_all: List[int] = []  # [B]

        for req_idx in range(B):
            local_new_len = seq_lens[req_idx]
            total_len = total_lens[req_idx]
            chunk_len = chunk_lens[req_idx]
            L_local_cached = cached_lens_list[req_idx]
            L_prev = total_len - chunk_len
            L_local_total = L_local_cached + local_new_len

            total_counts = [_cyclic_count(total_len, P, r) for r in range(P)]
            cached_counts = [_cyclic_count(L_prev, P, r) for r in range(P)]
            new_counts = [total_counts[r] - cached_counts[r] for r in range(P)]
            sizes_row = [new_counts[c * R + row_idx] for c in range(C)]
            sizes_col = [total_counts[rr * C + col_idx] for rr in range(R)]

            assert total_counts[cp_rank] == L_local_total
            assert new_counts[cp_rank] == local_new_len

            total_counts_all.append(total_counts)
            sizes_row_all.append(sizes_row)
            sizes_col_all.append(sizes_col)
            L_q_per_row_all.append(sum(sizes_row))
            L_k_per_col_all.append(sum(sizes_col))
            L_prev_all.append(L_prev)
            L_local_total_all.append(L_local_total)
            local_new_len_all.append(local_new_len)

        # ------------------------------------------------------------------ #
        # Collect K/V from cache or input args (per-request, then cat).       #
        # ------------------------------------------------------------------ #
        k_list: List[torch.Tensor] = []
        v_list: List[torch.Tensor] = []
        kv_offset = 0
        for req_idx in range(B):
            local_new_len = local_new_len_all[req_idx]
            L_local_total = L_local_total_all[req_idx]
            if has_cache:
                k_req, v_req = self._materialize_kv_from_pages(metadata, req_idx, L_local_total)
            else:
                k_req = k[kv_offset : kv_offset + local_new_len]
                v_req = v[kv_offset : kv_offset + local_new_len]
                kv_offset += local_new_len
            k_list.append(k_req)
            v_list.append(v_req)

        k_cat = torch.cat(k_list, dim=0) if B > 1 else k_list[0]
        v_cat = torch.cat(v_list, dim=0) if B > 1 else v_list[0]
        # q is already concatenated across requests.

        # ------------------------------------------------------------------ #
        # Batch collective sizes.                                              #
        # ------------------------------------------------------------------ #
        # Row-gather: batched sizes = sum of per-request sizes for each rank c.
        sizes_row_batch = [sum(sizes_row_all[i][c] for i in range(B)) for c in range(C)]
        # Col-gather: batched sizes = sum of per-request sizes for each row_idx_in_col.
        sizes_col_batch = [sum(sizes_col_all[i][rr] for i in range(B)) for rr in range(R)]
        # Redistribute recv: all requests share the same source rank mapping.
        source = (cp_rank % R) * C + (cp_rank // R)
        recv_count_batch = sum(total_counts_all[i][source] for i in range(B))

        # ------------------------------------------------------------------ #
        # 1) ONE batched row-gather Q.                                         #
        # Requests are already cat'd in q; sizes_row_batch[c] tells the        #
        # allgather how many tokens rank c contributes for the whole batch.    #
        # ------------------------------------------------------------------ #
        if C > 1:
            q_gathered = attn2d_row_allgather(q.contiguous(), mapping, dim=0, sizes=sizes_row_batch)
        else:
            q_gathered = q
        # q_gathered: (sum_i(L_q_per_row_i), H_q, D), rank-major across batch.
        # Rank c's slice is q_gathered[sum(sizes_row_batch[:c]) : sum(sizes_row_batch[:c+1])].
        # Within rank c's slice, requests appear in order with sizes sizes_row_all[i][c].

        # ------------------------------------------------------------------ #
        # 2-3) ONE batched redistribute + col-gather K/V.                     #
        # ------------------------------------------------------------------ #
        if R > 1:
            kv_send = torch.stack([k_cat, v_cat], dim=1).contiguous()
            kv_redistributed = _redistribute_kv_to_row_major(
                kv_send,
                R=R,
                C=C,
                cp_rank=cp_rank,
                cp_group=mapping.cp_group,
                recv_count=recv_count_batch,
            )
            kv_recv = attn2d_col_allgather(kv_redistributed, mapping, dim=0, sizes=sizes_col_batch)
            k_col_batch = kv_recv[:, 0].contiguous()
            v_col_batch = kv_recv[:, 1].contiguous()
        else:
            k_col_batch = k_cat
            v_col_batch = v_cat
        # k_col_batch: (sum_i(L_k_per_col_i), H_kv, D), col-rank-major across batch.
        # Col-rank rr's slice starts at sum(sizes_col_batch[:rr]).

        # ------------------------------------------------------------------ #
        # Build per-request start offsets into the gathered Q and K tensors.  #
        # q_req_starts[i][c] = start of req i's tokens from row-rank c.       #
        # k_req_starts[i][rr] = start of req i's tokens from col-rank rr.     #
        # ------------------------------------------------------------------ #
        q_rank_starts = [sum(sizes_row_batch[:c]) for c in range(C)]
        k_rank_starts = [sum(sizes_col_batch[:rr]) for rr in range(R)]

        q_req_starts: List[List[int]] = []
        q_rank_cum = [0] * C
        for i in range(B):
            q_req_starts.append([q_rank_starts[c] + q_rank_cum[c] for c in range(C)])
            for c in range(C):
                q_rank_cum[c] += sizes_row_all[i][c]

        k_req_starts: List[List[int]] = []
        k_rank_cum = [0] * R
        for i in range(B):
            k_req_starts.append([k_rank_starts[rr] + k_rank_cum[rr] for rr in range(R)])
            for rr in range(R):
                k_rank_cum[rr] += sizes_col_all[i][rr]

        # ------------------------------------------------------------------ #
        # 4a) Per-request extraction and sort (no FlashInfer calls yet).      #
        # ------------------------------------------------------------------ #
        sorted_q_list: List[torch.Tensor] = []
        sorted_k_list: List[torch.Tensor] = []
        sorted_v_list: List[torch.Tensor] = []
        q_sort_idx_list: List[torch.Tensor] = []
        first_q_row_list: List[int] = []

        for i in range(B):
            L_q_i = L_q_per_row_all[i]
            L_k_i = L_k_per_col_all[i]

            q_parts = [
                q_gathered[q_req_starts[i][c] : q_req_starts[i][c] + sizes_row_all[i][c]]
                for c in range(C)
            ]
            q_full_i = torch.cat(q_parts, dim=0) if C > 1 else q_parts[0]

            k_parts = [
                k_col_batch[k_req_starts[i][rr] : k_req_starts[i][rr] + sizes_col_all[i][rr]]
                for rr in range(R)
            ]
            v_parts = [
                v_col_batch[k_req_starts[i][rr] : k_req_starts[i][rr] + sizes_col_all[i][rr]]
                for rr in range(R)
            ]
            k_col_i = torch.cat(k_parts, dim=0) if R > 1 else k_parts[0]
            v_col_i = torch.cat(v_parts, dim=0) if R > 1 else v_parts[0]

            q_sort_idx = _build_q_sort_idx(
                chunk_lens[i], L_prev_all[i], R, C, row_idx, sizes_row_all[i], device=device
            )
            k_sort_idx = _build_k_sort_idx(
                total_lens[i], R, C, col_idx, sizes_col_all[i], device=device
            )
            sorted_q = q_full_i[q_sort_idx] if L_q_i > 0 else q_full_i
            sorted_k = k_col_i[k_sort_idx] if L_k_i > 0 else k_col_i
            sorted_v = v_col_i[k_sort_idx] if L_k_i > 0 else v_col_i

            sorted_q_list.append(sorted_q)
            sorted_k_list.append(sorted_k)
            sorted_v_list.append(sorted_v)
            q_sort_idx_list.append(q_sort_idx)
            first_q_row_list.append(_first_chunk_pos(L_prev_all[i], R, row_idx))

        # ------------------------------------------------------------------ #
        # 4b) Batched attention: s rounds of B-request batch_prefill calls.   #
        # Reduces B*s individual single_prefill calls to s batch calls.       #
        # output_sorted_list[i]: (L_q_i, H_q, D); lse_sorted_list[i]: same   #
        # but (L_q_i, H_q) -- both in absolute-position sorted order.        #
        # ------------------------------------------------------------------ #
        output_sorted_list, lse_sorted_list = self._compute_attention_sorted_batch(
            sorted_q_list,
            sorted_k_list,
            sorted_v_list,
            L_q_list=L_q_per_row_all,
            L_k_list=L_k_per_col_all,
            first_q_row_list=first_q_row_list,
            L_prev_list=L_prev_all,
            L_chunk_list=chunk_lens,
            L_total_list=total_lens,
            H_q=H_q,
            D=D,
            R=R,
            C=C,
            row_idx=row_idx,
            col_idx=col_idx,
            device=device,
        )

        # ------------------------------------------------------------------ #
        # 4c) Unsort per request.                                             #
        # output_all[i]: (L_q_per_row_i, H_q, D) in rank-major order.        #
        # lse_all[i]:    (L_q_per_row_i, H_q).                                #
        # ------------------------------------------------------------------ #
        output_all: List[torch.Tensor] = []
        lse_all: List[torch.Tensor] = []

        for i in range(B):
            L_q_i = L_q_per_row_all[i]
            if L_q_i > 0:
                output_i = sorted_q_list[i].new_empty(L_q_i, H_q, D)
                lse_i = sorted_q_list[i].new_empty(L_q_i, H_q, dtype=torch.float32)
                output_i[q_sort_idx_list[i]] = output_sorted_list[i]
                lse_i[q_sort_idx_list[i]] = lse_sorted_list[i]
            else:
                output_i = output_sorted_list[i]
                lse_i = lse_sorted_list[i]
            output_all.append(output_i)
            lse_all.append(lse_i)

        # ------------------------------------------------------------------ #
        # 5) ONE batched row-group all_to_all + per-request LSE merge.        #
        #                                                                      #
        # Each per-rank send chunk contains B sub-chunks (one per request),   #
        # each padded to its per-request max_sz.  The all-to-all is uniform:  #
        # every rank c sends sum_i(max_sz_i) tokens.  After receiving, we     #
        # split per request and truncate + merge.                              #
        # ------------------------------------------------------------------ #
        if C > 1:
            max_sz_all = [max(sizes_row_all[i]) if sizes_row_all[i] else 0 for i in range(B)]
            total_max_sz = sum(max_sz_all)

            if total_max_sz == 0:
                return q.new_zeros(sum(local_new_len_all), H_q * D)

            o_chunks: List[torch.Tensor] = []
            lse_chunks: List[torch.Tensor] = []
            for c in range(C):
                o_c_parts: List[torch.Tensor] = []
                lse_c_parts: List[torch.Tensor] = []
                for i in range(B):
                    sz_i = sizes_row_all[i][c]
                    max_sz_i = max_sz_all[i]
                    # Rank c's portion of request i lives at offset
                    # sum(sizes_row_all[i][:c]) within output_all[i].
                    start_i = sum(sizes_row_all[i][:c])
                    o_ci = output_all[i][start_i : start_i + sz_i]
                    lse_ci = lse_all[i][start_i : start_i + sz_i]
                    if sz_i < max_sz_i:
                        pad = max_sz_i - sz_i
                        o_ci = torch.cat([o_ci, o_ci.new_zeros(pad, H_q, D)], dim=0)
                        lse_ci = torch.cat(
                            [lse_ci, lse_ci.new_full((pad, H_q), float("-inf"))], dim=0
                        )
                    o_c_parts.append(o_ci)
                    lse_c_parts.append(lse_ci)
                o_chunks.append(torch.cat(o_c_parts, dim=0).contiguous())
                lse_chunks.append(torch.cat(lse_c_parts, dim=0).contiguous())

            o_recv, lse_recv = attn2d_row_alltoall(o_chunks + lse_chunks, mapping)
            # o_recv:   (C, total_max_sz, H_q, D)
            # lse_recv: (C, total_max_sz, H_q)

            outputs_final: List[torch.Tensor] = []
            req_start = 0
            for i in range(B):
                real_sz_i = sizes_row_all[i][col_idx]
                max_sz_i = max_sz_all[i]
                o_recv_i = o_recv[:, req_start : req_start + max_sz_i, :, :][:, :real_sz_i]
                lse_recv_i = lse_recv[:, req_start : req_start + max_sz_i, :][:, :real_sz_i]
                req_start += max_sz_i
                v_stack_i = o_recv_i.permute(1, 0, 2, 3).contiguous()
                s_stack_i = lse_recv_i.permute(1, 0, 2).contiguous()
                out_merged_i, _ = flashinfer.merge_states(v_stack_i, s_stack_i)
                outputs_final.append(out_merged_i)

            return torch.cat(outputs_final, dim=0).reshape(-1, H_q * D)

        # C == 1: tokens are already complete per request, just cat.
        return torch.cat(output_all, dim=0).reshape(-1, H_q * D)

    def _get_ragged_prefill_wrapper(
        self,
        device: torch.device,
    ) -> flashinfer.BatchPrefillWithRaggedKVCacheWrapper:
        """Return (and lazily create) the shared ragged-prefill wrapper."""
        if self._ragged_prefill_wrapper is None:
            workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
            self._ragged_prefill_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                workspace, kv_layout="NHD"
            )
        return self._ragged_prefill_wrapper

    def _compute_attention_sorted_batch(
        self,
        sorted_q_list: List[torch.Tensor],
        sorted_k_list: List[torch.Tensor],
        sorted_v_list: List[torch.Tensor],
        *,
        L_q_list: List[int],
        L_k_list: List[int],
        first_q_row_list: List[int],
        L_prev_list: List[int],
        L_chunk_list: List[int],
        L_total_list: List[int],
        H_q: int,
        D: int,
        R: int,
        C: int,
        row_idx: int,
        col_idx: int,
        device: torch.device,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Batched Q/K-split FlashInfer attention over B requests.

        Reduces from B*s individual ``single_prefill_with_kv_cache`` calls to
        s ``BatchPrefillWithRaggedKVCacheWrapper`` calls by grouping all B
        requests into one ragged batch per split shard (one round per u/t).

        Returns ``(output_sorted_list, lse_sorted_list)`` each of length B.
        ``output_sorted_list[i]`` has shape ``(L_q_list[i], H_q, D)`` and
        ``lse_sorted_list[i]`` has shape ``(L_q_list[i], H_q)``, both in
        absolute-position sorted order.
        """
        B = len(sorted_q_list)
        H_kv = self.num_kv_heads

        # Pre-allocate per-request output tensors.
        output_sorted_list: List[torch.Tensor] = []
        lse_sorted_list: List[torch.Tensor] = []
        for i in range(B):
            L_q_i = L_q_list[i]
            if L_q_i == 0 or L_k_list[i] == 0:
                output_sorted_list.append(sorted_q_list[i].new_zeros(L_q_i, H_q, D))
                lse_sorted_list.append(
                    sorted_q_list[i].new_full((L_q_i, H_q), float("-inf"), dtype=torch.float32)
                )
            else:
                output_sorted_list.append(sorted_q_list[i].new_empty(L_q_i, H_q, D))
                lse_sorted_list.append(sorted_q_list[i].new_empty(L_q_i, H_q, dtype=torch.float32))

        wrapper = self._get_ragged_prefill_wrapper(device)

        if C % R == 0:
            # Q-split: s = C/R rounds; full K/V replicated each round.
            s = C // R
            for t in range(s):
                active_reqs: List[int] = []
                q_parts: List[torch.Tensor] = []
                k_parts: List[torch.Tensor] = []
                v_parts: List[torch.Tensor] = []
                qo_lens: List[int] = []
                kv_lens: List[int] = []
                dummy_flags: List[bool] = []
                q_t_counts: List[int] = []

                for i in range(B):
                    L_q_i = L_q_list[i]
                    L_k_i = L_k_list[i]
                    if L_q_i == 0 or L_k_i == 0:
                        continue
                    Q_t = sorted_q_list[i][t::s]
                    Q_t_count = Q_t.shape[0]
                    if Q_t_count == 0:
                        continue
                    required_shift = (first_q_row_list[i] + t * R - col_idx) // C
                    natural_shift = L_k_i - Q_t_count
                    diff = natural_shift - required_shift
                    assert diff in (0, 1), (
                        f"unexpected dummy count {diff} at (row={row_idx}, col={col_idx}, t={t}) "
                        f"for L_prev={L_prev_list[i]}, L_chunk={L_chunk_list[i]}, "
                        f"L_total={L_total_list[i]}; natural={natural_shift}, required={required_shift}"
                    )
                    Q_in = (
                        torch.cat([Q_t.contiguous(), Q_t.new_zeros(1, H_q, D)], dim=0)
                        if diff == 1
                        else Q_t.contiguous()
                    )
                    q_parts.append(Q_in)
                    k_parts.append(sorted_k_list[i])
                    v_parts.append(sorted_v_list[i])
                    qo_lens.append(Q_in.shape[0])
                    kv_lens.append(L_k_i)
                    dummy_flags.append(diff == 1)
                    q_t_counts.append(Q_t_count)
                    active_reqs.append(i)

                if not active_reqs:
                    continue

                q_cat = torch.cat(q_parts, dim=0)
                k_cat = torch.cat(k_parts, dim=0)
                v_cat = torch.cat(v_parts, dim=0)
                qo_indptr = torch.tensor(
                    [0] + list(accumulate(qo_lens)), dtype=torch.int32, device=device
                )
                kv_indptr = torch.tensor(
                    [0] + list(accumulate(kv_lens)), dtype=torch.int32, device=device
                )
                wrapper.plan(
                    qo_indptr,
                    kv_indptr,
                    H_q,
                    H_kv,
                    D,
                    causal=True,
                    q_data_type=q_cat.dtype,
                    kv_data_type=k_cat.dtype,
                )
                out_t_all, lse_t_all = wrapper.run(q_cat, k_cat, v_cat, return_lse=True)

                ptr = 0
                for j, i in enumerate(active_reqs):
                    qo_len = qo_lens[j]
                    out_t = out_t_all[ptr : ptr + qo_len]
                    lse_t = lse_t_all[ptr : ptr + qo_len]
                    ptr += qo_len
                    if dummy_flags[j]:
                        out_t = out_t[:-1]
                        lse_t = lse_t[:-1]
                    output_sorted_list[i][t::s] = out_t
                    lse_sorted_list[i][t::s] = lse_t

        else:
            # K-split: s = R/C rounds; Q replicated each round.
            # After all rounds, LSE-merge the s partial results per request.
            s = R // C
            v_stacks: List[Optional[torch.Tensor]] = [
                sorted_q_list[i].new_empty(L_q_list[i], s, H_q, D)
                if L_q_list[i] > 0 and L_k_list[i] > 0
                else None
                for i in range(B)
            ]
            s_stacks: List[Optional[torch.Tensor]] = [
                sorted_q_list[i].new_empty(L_q_list[i], s, H_q, dtype=torch.float32)
                if L_q_list[i] > 0 and L_k_list[i] > 0
                else None
                for i in range(B)
            ]

            for u in range(s):
                active_reqs = []
                q_parts = []
                k_parts = []
                v_parts = []
                qo_lens = []
                kv_lens = []
                dummy_flags = []

                for i in range(B):
                    L_q_i = L_q_list[i]
                    L_k_i = L_k_list[i]
                    if L_q_i == 0 or L_k_i == 0:
                        continue
                    K_u = sorted_k_list[i][u::s].contiguous()
                    V_u = sorted_v_list[i][u::s].contiguous()
                    K_u_count = K_u.shape[0]
                    if K_u_count == 0:
                        # Zero-fill this shard slot.
                        v_stacks[i][:, u].zero_()  # type: ignore[index]
                        s_stacks[i][:, u].fill_(float("-inf"))  # type: ignore[index]
                        continue
                    required_shift = (first_q_row_list[i] - col_idx - u * C) // R
                    natural_shift = K_u_count - L_q_i
                    diff = natural_shift - required_shift
                    assert diff in (0, 1), (
                        f"unexpected dummy count {diff} at (row={row_idx}, col={col_idx}, u={u}) "
                        f"for L_prev={L_prev_list[i]}, L_chunk={L_chunk_list[i]}, "
                        f"L_total={L_total_list[i]}; natural={natural_shift}, required={required_shift}"
                    )
                    Q_in = (
                        torch.cat([sorted_q_list[i], sorted_q_list[i].new_zeros(1, H_q, D)], dim=0)
                        if diff == 1
                        else sorted_q_list[i]
                    )
                    q_parts.append(Q_in)
                    k_parts.append(K_u)
                    v_parts.append(V_u)
                    qo_lens.append(Q_in.shape[0])
                    kv_lens.append(K_u_count)
                    dummy_flags.append(diff == 1)
                    active_reqs.append(i)

                if not active_reqs:
                    continue

                q_cat = torch.cat(q_parts, dim=0)
                k_cat = torch.cat(k_parts, dim=0)
                v_cat = torch.cat(v_parts, dim=0)
                qo_indptr = torch.tensor(
                    [0] + list(accumulate(qo_lens)), dtype=torch.int32, device=device
                )
                kv_indptr = torch.tensor(
                    [0] + list(accumulate(kv_lens)), dtype=torch.int32, device=device
                )
                wrapper.plan(
                    qo_indptr,
                    kv_indptr,
                    H_q,
                    H_kv,
                    D,
                    causal=True,
                    q_data_type=q_cat.dtype,
                    kv_data_type=k_cat.dtype,
                )
                out_u_all, lse_u_all = wrapper.run(q_cat, k_cat, v_cat, return_lse=True)

                ptr = 0
                for j, i in enumerate(active_reqs):
                    qo_len = qo_lens[j]
                    out_u = out_u_all[ptr : ptr + qo_len]
                    lse_u = lse_u_all[ptr : ptr + qo_len]
                    ptr += qo_len
                    if dummy_flags[j]:
                        out_u = out_u[:-1]
                        lse_u = lse_u[:-1]
                    v_stacks[i][:, u] = out_u  # type: ignore[index]
                    s_stacks[i][:, u] = lse_u  # type: ignore[index]

            # LSE-merge the s K-shards per request.
            for i in range(B):
                if L_q_list[i] == 0 or L_k_list[i] == 0:
                    continue
                if s == 1:
                    output_sorted_list[i] = v_stacks[i][:, 0]  # type: ignore[index]
                    lse_sorted_list[i] = s_stacks[i][:, 0]  # type: ignore[index]
                else:
                    out_i, lse_i = flashinfer.merge_states(v_stacks[i], s_stacks[i])
                    output_sorted_list[i] = out_i
                    lse_sorted_list[i] = lse_i

        return output_sorted_list, lse_sorted_list

    def _materialize_kv_from_pages(
        self,
        metadata: Attn2DFlashInferAttentionMetadata,
        req_idx: int,
        L_local_total: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract this rank's cached + new K/V for one request from paged cache.

        After ``_append_new_kv_to_cache`` has run, the cache contains this
        rank's cyclic-shard slice of [0, L_total) for the request.  This
        method materializes those ``L_local_total`` entries into a pair of
        contiguous tensors for the subsequent mesh comm.

        Returns ``(k, v)`` each of shape ``(L_local_total, H_kv, D)`` in
        absolute-position order (per-rank cyclic order: r, r+P, r+2P, ...).
        """
        if L_local_total == 0:
            H_kv = self.num_kv_heads
            D = self.head_dim
            dev = metadata.paged_kv_indices.device
            # Match dtype of the cache buffer.
            kv_cache_buf = metadata.kv_cache_manager.get_buffers(self.layer_idx, kv_layout="NHD")
            return (
                torch.empty(0, H_kv, D, dtype=kv_cache_buf.dtype, device=dev),
                torch.empty(0, H_kv, D, dtype=kv_cache_buf.dtype, device=dev),
            )

        kv_cache_buf = metadata.kv_cache_manager.get_buffers(self.layer_idx, kv_layout="NHD")
        page_size = metadata.kv_cache_manager.tokens_per_block
        page_start = int(metadata.paged_kv_indptr[req_idx])
        page_end = int(metadata.paged_kv_indptr[req_idx + 1])
        num_pages = page_end - page_start
        last_page_len = int(metadata.paged_kv_last_page_len[req_idx])

        # NHD layout: (num_pages_total, 2, page_size, H_kv, D).  Gather this
        # request's pages, then permute (page, slot) -> (page*page_size, ...).
        page_ids = metadata.paged_kv_indices[page_start:page_end]
        pages = kv_cache_buf[page_ids]  # (num_pages, 2, page_size, H_kv, D)
        # Flatten (num_pages, page_size) into a linear token axis: dim order is
        # (num_pages, page_size, 2, H_kv, D), then reshape.
        flat = (
            pages.permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(num_pages * page_size, 2, *pages.shape[3:])
        )
        # Tokens in the last page may be partially filled.
        valid_tokens = (num_pages - 1) * page_size + last_page_len
        assert valid_tokens == L_local_total, (
            f"page table inconsistent for req {req_idx}: derived "
            f"{valid_tokens} but expected L_local_total={L_local_total}"
        )
        kv = flat[:L_local_total]  # (L_local_total, 2, H_kv, D)
        return kv[:, 0].contiguous(), kv[:, 1].contiguous()

    def _append_new_kv_to_cache(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        metadata: Attn2DFlashInferAttentionMetadata,
    ) -> None:
        """Append this rank's new K/V to the paged cache (batch-level).

        ``k``, ``v`` are this rank's freshly-computed K/V for the current
        chunk, shape ``(total_new_tokens, H_kv, D)``, concatenated across
        all requests in the batch.  Each rank writes its own cyclic-shard
        slice -- the cache manager state on this rank tracks the rank's
        own pages, so writing here uses the per-rank page indices that
        ``prepare()`` already populated in ``metadata.paged_kv_indices``.

        Uses ``flashinfer.page.append_paged_kv_cache`` with the standard
        (batch_indices, positions) pair derived from the post-append
        state via ``get_seq_lens`` + ``get_batch_indices_positions``.
        Matches the pattern in
        ``tensorrt_llm/_torch/attention_backend/flashinfer.py:1537``.
        """
        kv_cache_buf = metadata.kv_cache_manager.get_buffers(self.layer_idx, kv_layout="NHD")
        page_size = metadata.kv_cache_manager.tokens_per_block
        n = metadata.num_seqs

        # qo_indptr: cumsum of per-request rank-local new tokens.
        # Built on the fly because there's only one per-forward call here;
        # if this becomes a hot path, promote to a stable buffer in
        # __post_init__ and populate in prepare().
        seq_lens_cuda = metadata.seq_lens_cuda
        qo_indptr = torch.zeros(n + 1, dtype=torch.int32, device=seq_lens_cuda.device)
        torch.cumsum(
            seq_lens_cuda.to(torch.int32),
            dim=0,
            dtype=torch.int32,
            out=qo_indptr[1:],
        )

        # seq_lens_total: post-append total per request, recovered from the
        # page table that prepare() set up for the (cached + new) state.
        seq_lens_total = flashinfer.get_seq_lens(
            metadata.paged_kv_indptr[: n + 1],
            metadata.paged_kv_last_page_len[:n],
            page_size,
        )

        num_new_tokens = k.shape[0]
        batch_indices, positions = flashinfer.get_batch_indices_positions(
            qo_indptr, seq_lens_total, num_new_tokens
        )

        flashinfer.page.append_paged_kv_cache(
            append_key=k.contiguous(),
            append_value=v.contiguous(),
            batch_indices=batch_indices,
            positions=positions,
            paged_kv_cache=kv_cache_buf,
            kv_indices=metadata.paged_kv_indices,
            kv_indptr=metadata.paged_kv_indptr[: n + 1],
            kv_last_page_len=metadata.paged_kv_last_page_len[:n],
            kv_layout="NHD",
        )

    def _compute_attention_sorted(
        self,
        sorted_q: torch.Tensor,
        sorted_k: torch.Tensor,
        sorted_v: torch.Tensor,
        *,
        L_q: int,
        L_k: int,
        H_q: int,
        D: int,
        first_q_row: int,
        col_idx: int,
        L_prev: int,
        L_chunk: int,
        L_total: int,
        row_idx: int,
        R: int,
        C: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Q/K-split FlashInfer kernel calls on sorted Q and K/V.

        Returns ``(output_sorted, lse_sorted)`` in absolute-position sorted
        order, shapes ``(L_q, H_q, D)`` and ``(L_q, H_q)``.  The caller is
        responsible for un-sorting back to row-gather rank-major layout.
        """
        if L_q == 0 or L_k == 0:
            output_sorted = sorted_q.new_zeros(L_q, H_q, D)
            lse_sorted = sorted_q.new_full((L_q, H_q), float("-inf"), dtype=torch.float32)
            return output_sorted, lse_sorted

        if C % R == 0:
            # Q-split: Q_t at stride s = C/R; full K per sub-call.
            s = C // R
            output_sorted = sorted_q.new_empty(L_q, H_q, D)
            lse_sorted = sorted_q.new_empty(L_q, H_q, dtype=torch.float32)
            for t in range(s):
                Q_t = sorted_q[t::s]
                Q_t_count = Q_t.shape[0]
                if Q_t_count == 0:
                    continue
                required_shift = (first_q_row + t * R - col_idx) // C
                natural_shift = L_k - Q_t_count
                diff = natural_shift - required_shift
                assert diff in (0, 1), (
                    f"unexpected dummy count {diff} at (row={row_idx}, "
                    f"col={col_idx}, t={t}) for L_prev={L_prev}, L_chunk={L_chunk}, "
                    f"L_total={L_total}; natural={natural_shift}, required={required_shift}"
                )
                Q_in = torch.cat([Q_t, Q_t.new_zeros(1, H_q, D)], dim=0) if diff == 1 else Q_t
                out_t, lse_t = flashinfer.single_prefill_with_kv_cache(
                    Q_in, sorted_k, sorted_v, causal=True, kv_layout="NHD", return_lse=True
                )
                if diff == 1:
                    out_t = out_t[:-1]
                    lse_t = lse_t[:-1]
                output_sorted[t::s] = out_t
                lse_sorted[t::s] = lse_t
            return output_sorted, lse_sorted

        # K-split: full Q; K_u at stride s = R/C.  LSE-merge across u.
        s = R // C
        v_stack = sorted_q.new_empty(L_q, s, H_q, D)
        s_stack = sorted_q.new_empty(L_q, s, H_q, dtype=torch.float32)
        for u in range(s):
            K_u = sorted_k[u::s]
            V_u = sorted_v[u::s]
            K_u_count = K_u.shape[0]
            if K_u_count == 0:
                v_stack[:, u].zero_()
                s_stack[:, u].fill_(float("-inf"))
                continue
            required_shift = (first_q_row - col_idx - u * C) // R
            natural_shift = K_u_count - L_q
            diff = natural_shift - required_shift
            assert diff in (0, 1), (
                f"unexpected dummy count {diff} at (row={row_idx}, "
                f"col={col_idx}, u={u}) for L_prev={L_prev}, L_chunk={L_chunk}, "
                f"L_total={L_total}; natural={natural_shift}, required={required_shift}"
            )
            Q_in = (
                torch.cat([sorted_q, sorted_q.new_zeros(1, H_q, D)], dim=0)
                if diff == 1
                else sorted_q
            )
            out_u, lse_u = flashinfer.single_prefill_with_kv_cache(
                Q_in, K_u, V_u, causal=True, kv_layout="NHD", return_lse=True
            )
            if diff == 1:
                out_u = out_u[:-1]
                lse_u = lse_u[:-1]
            v_stack[:, u] = out_u
            s_stack[:, u] = lse_u
        output_sorted, lse_sorted = flashinfer.merge_states(v_stack, s_stack)
        return output_sorted, lse_sorted

    def _forward_single_request(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        L_total: int,
        L_chunk: int,
        L_local_cached: int,
        R: int,
        C: int,
        P: int,
        cp_rank: int,
        row_idx: int,
        col_idx: int,
        mapping,
    ) -> torch.Tensor:
        """Process one request's cyclic shard. Returns ``[L_local_new, H_q, D]``.

        Kept for unit-test compatibility.  Production ``forward()`` uses batched
        collectives (one per phase across all requests in the batch) and calls
        ``_compute_attention_sorted`` directly inside a per-request loop.
        """
        L_local_new = q.shape[0]
        L_local_total = k.shape[0]
        assert L_local_total == L_local_cached + L_local_new, (
            f"L_local_total ({L_local_total}) != cached ({L_local_cached}) + new ({L_local_new})"
        )
        L_prev = L_total - L_chunk
        H_q, D = q.shape[1], q.shape[2]
        device = q.device

        total_counts = [_cyclic_count(L_total, P, r) for r in range(P)]
        cached_counts = [_cyclic_count(L_prev, P, r) for r in range(P)]
        new_counts = [total_counts[r] - cached_counts[r] for r in range(P)]
        assert total_counts[cp_rank] == L_local_total
        assert new_counts[cp_rank] == L_local_new

        sizes_row = [new_counts[c * R + row_idx] for c in range(C)]
        L_q_per_row = sum(sizes_row)
        sizes_col = [total_counts[row_idx_in_col * C + col_idx] for row_idx_in_col in range(R)]
        L_k_per_col = sum(sizes_col)

        if C > 1:
            q_full = attn2d_row_allgather(q.contiguous(), mapping, dim=0, sizes=sizes_row)
        else:
            q_full = q

        if R > 1:
            kv_send = torch.stack([k, v], dim=1).contiguous()
            source = (cp_rank % R) * C + (cp_rank // R)
            recv_count = total_counts[source]
            kv_send = _redistribute_kv_to_row_major(
                kv_send,
                R=R,
                C=C,
                cp_rank=cp_rank,
                cp_group=mapping.cp_group,
                recv_count=recv_count,
            )
            kv_recv = attn2d_col_allgather(kv_send, mapping, dim=0, sizes=sizes_col)
            k_col = kv_recv[:, 0].contiguous()
            v_col = kv_recv[:, 1].contiguous()
        else:
            k_col = k
            v_col = v

        q_sort_idx = _build_q_sort_idx(L_chunk, L_prev, R, C, row_idx, sizes_row, device=device)
        k_sort_idx = _build_k_sort_idx(L_total, R, C, col_idx, sizes_col, device=device)
        sorted_q = q_full[q_sort_idx] if L_q_per_row > 0 else q_full
        sorted_k = k_col[k_sort_idx] if L_k_per_col > 0 else k_col
        sorted_v = v_col[k_sort_idx] if L_k_per_col > 0 else v_col

        first_q_row = _first_chunk_pos(L_prev, R, row_idx)
        output_sorted, lse_sorted = self._compute_attention_sorted(
            sorted_q,
            sorted_k,
            sorted_v,
            L_q=L_q_per_row,
            L_k=L_k_per_col,
            H_q=H_q,
            D=D,
            first_q_row=first_q_row,
            col_idx=col_idx,
            L_prev=L_prev,
            L_chunk=L_chunk,
            L_total=L_total,
            row_idx=row_idx,
            R=R,
            C=C,
        )

        if L_q_per_row > 0:
            output = sorted_q.new_empty(L_q_per_row, H_q, D)
            lse = sorted_q.new_empty(L_q_per_row, H_q, dtype=torch.float32)
            output[q_sort_idx] = output_sorted
            lse[q_sort_idx] = lse_sorted
        else:
            output = output_sorted
            lse = lse_sorted

        if C > 1:
            max_sz = max(sizes_row) if sizes_row else 0
            if max_sz == 0:
                return output
            o_chunks: List[torch.Tensor] = []
            lse_chunks: List[torch.Tensor] = []
            cum = 0
            for c in range(C):
                sz = sizes_row[c]
                o_c = output[cum : cum + sz]
                lse_c = lse[cum : cum + sz]
                if sz < max_sz:
                    pad = max_sz - sz
                    o_c = torch.cat([o_c, o_c.new_zeros(pad, H_q, D)], dim=0)
                    lse_c = torch.cat([lse_c, lse_c.new_full((pad, H_q), float("-inf"))], dim=0)
                o_chunks.append(o_c.contiguous())
                lse_chunks.append(lse_c.contiguous())
                cum += sz
            o_recv, lse_recv = attn2d_row_alltoall(o_chunks + lse_chunks, mapping)
            real_sz = sizes_row[col_idx]
            o_recv = o_recv[:, :real_sz]
            lse_recv = lse_recv[:, :real_sz]
            v_stack = o_recv.permute(1, 0, 2, 3).contiguous()
            s_stack = lse_recv.permute(1, 0, 2).contiguous()
            out_merged, _ = flashinfer.merge_states(v_stack, s_stack)
            return out_merged

        return output
