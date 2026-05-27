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
  4. computes local attention via flashinfer with log-sum-exp output;
     square meshes use a position-sorted layout and ``causal=True``
     (no custom mask), non-square meshes fall back to an absolute-
     position custom mask
  5. scatters the L/R partials back to their original owners via
     all_to_all_single within the row group, then LSE-merges the C
     partials this rank receives -> final output for L/P tokens

Mask shape after the gathers depends on the mesh aspect ratio.  Re-
sorting Q and K/V into ascending absolute position gives positions
``q_pos[m] = m*R + row_idx`` and ``k_pos[k] = k*C + col_idx``, so the
``k_pos <= q_pos`` mask becomes ``(k - m) * C <= row_idx - col_idx``
(square mesh) or a stair-step with slope ``C/R`` otherwise.

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
  * Otherwise (e.g. ``R=4, C=6``): ``custom_mask=(k_pos <= q_pos)``.

Strict causal (``k < i``) on equal-length tensors is expressed by
appending one dummy Q row so ``qo_len = L_q + 1`` and ``kv_len = L_q``;
FlashInfer's bottom-right ``causal=True`` then applies ``k <= i - 1``
on the real rows, and the dummy output / LSE are sliced off.

FA2/FA3 public APIs do not expose ``custom_mask`` (and a user-defined
mask rules out FA4), so FlashInfer is the only optimized kernel option
for the fallback path.  KV cache, generation phase, fused QKV, sparse /
sliding-window masks, and mixed-dtype quantization are out of scope for
v0.  Inputs are assumed to be cyclic-sharded by the caller.

For requests whose total length is not divisible by ``cp_size``, the
backend pads each rank's local shard up to ``ceil(L_total / cp_size)``
with zero rows so every collective stays uniform-shape.  Padding tokens
land at absolute positions ``>= L_total``, so the causal mask
``p_k <= p_q`` excludes them from every real-Q output (Q-split,
K-split, and custom_mask paths all handle this naturally).  Padding Q
outputs are sliced off at exit.  The total length is read from
``metadata.total_input_lens``; if absent the backend assumes
divisibility.
"""

from dataclasses import dataclass
from typing import List, Optional

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
    ``seq_lens`` is the per-request shard length on this rank.

    ``total_input_lens`` is the per-request total length (same value on
    every rank).  When ``L_total % cp_size != 0``, ranks ``r < L_total %
    cp_size`` get one extra token; the backend pads every rank up to
    ``ceil(L_total / cp_size)`` so all collectives stay uniform-shape
    and the natural causal mask excludes padding positions (>= L_total)
    from real-Q attention outputs.  Padding Q outputs are sliced off
    at exit.

    If ``total_input_lens`` is ``None``, the backend falls back to the
    divisible-L assumption and ``L_total = seq_lens * cp_size``.
    """

    total_input_lens: Optional[torch.Tensor] = None

    def prepare(self) -> None:
        super().prepare()

    def update_attn2d_param(
        self,
        total_input_lens: Optional[List[int]],
    ) -> None:
        """Set per-request total input lengths for the ATTN2D backend.

        Called from the model engine's prepare-inputs path once per
        batch, alongside ``seq_lens`` (mirrors ``update_helix_param``).

        Args:
            total_input_lens: per-request total (un-sharded) input
                length, one entry per request in the batch.  When
                ``None`` (or omitted by the engine) the backend falls
                back to the divisible-L assumption.
        """
        if total_input_lens is None:
            self.total_input_lens = None
            return
        self.total_input_lens = torch.tensor(total_input_lens, dtype=torch.int32)


def _redistribute_kv_to_row_major(
    kv: torch.Tensor,
    *,
    R: int,
    C: int,
    cp_rank: int,
    cp_group: List[int],
) -> torch.Tensor:
    """Permute packed K/V across cp_group: column-major -> row-major cyclic.

    Input ``kv`` is the packed tensor of shape ``(2, L_local, H_kv, D)``
    with ``kv[0] = K`` and ``kv[1] = V``.  Returns the permuted packed
    tensor (same shape and dtype) -- either ``kv`` itself when the
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
        if metadata.kv_cache_manager is not None:
            raise NotImplementedError(
                "Attn2DFlashInferAttention does not support KV cache (prefill only in v0)."
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

        seq_lens = metadata.seq_lens.tolist()
        # Per-request total length (same on every rank).  When None, fall
        # back to the divisibility assumption L_total = L_local * P.
        if metadata.total_input_lens is not None:
            total_lens = metadata.total_input_lens.tolist()
        else:
            total_lens = [local_len * P for local_len in seq_lens]
        assert len(total_lens) == len(seq_lens)

        outputs = []
        offset = 0
        for local_len, total_len in zip(seq_lens, total_lens):
            q_local = q[offset : offset + local_len]
            k_local = k[offset : offset + local_len]
            v_local = v[offset : offset + local_len]

            outputs.append(
                self._forward_single_request(
                    q_local,
                    k_local,
                    v_local,
                    L_total=total_len,
                    R=R,
                    C=C,
                    P=P,
                    cp_rank=cp_rank,
                    row_idx=row_idx,
                    col_idx=col_idx,
                    mapping=mapping,
                )
            )
            offset += local_len

        return torch.cat(outputs, dim=0).reshape(-1, H_q * D)

    def _forward_single_request(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        L_total: int,
        R: int,
        C: int,
        P: int,
        cp_rank: int,
        row_idx: int,
        col_idx: int,
        mapping,
    ) -> torch.Tensor:
        """Process one request's cyclic shard. Returns [L_local, H_q, D].

        ``L_total`` is the request's total (un-sharded) length.  When
        ``L_total % P != 0``, ranks ``r < L_total % P`` already hold one
        extra real token; the rest pad up to ``ceil(L_total / P)`` with
        one zero row.  Padding rows take "virtual" positions ``>= L_total``
        so the natural causal mask excludes them from real-Q outputs,
        and we slice them off the final output below.
        """
        L_local = q.shape[0]
        L_local_padded = (L_total + P - 1) // P  # ceil(L_total / P)
        pad_rows = L_local_padded - L_local
        assert pad_rows in (0, 1), (
            f"unexpected pad_rows={pad_rows} for L_total={L_total}, P={P}, "
            f"L_local={L_local}; cyclic sharding can differ by at most 1."
        )
        H_q, D = q.shape[1], q.shape[2]
        H_kv = k.shape[1]
        device = q.device

        if pad_rows:
            q = torch.cat([q, q.new_zeros(pad_rows, H_q, D)], dim=0)
            k = torch.cat([k, k.new_zeros(pad_rows, H_kv, D)], dim=0)
            v = torch.cat([v, v.new_zeros(pad_rows, H_kv, D)], dim=0)
        # Below, ``L_local`` refers to the padded length; the original
        # is preserved as ``L_local_unpadded`` for the final slice.
        L_local_unpadded = L_local
        L_local = L_local_padded

        # --- 1) Row-gather Q: [L_local, H_q, D] -> [L_q = C*L_local, H_q, D].
        if C > 1:
            q_full = attn2d_row_allgather(q.contiguous(), mapping, dim=0)
        else:
            q_full = q

        # --- 2-3) Redistribute K/V (column-major -> row-major cyclic) and
        # col-gather, packed end-to-end.  Result is source-rank-grouped
        # K/V; step 4 re-layouts per path.
        L_q = C * L_local
        L_k = R * L_local

        if R > 1:
            kv_send = torch.stack([k, v], dim=0).contiguous()
            kv_send = _redistribute_kv_to_row_major(
                kv_send, R=R, C=C, cp_rank=cp_rank, cp_group=mapping.cp_group
            )
            # attn2d_col_allgather concatenates along dim 0; output is
            # rank-major, so .view(R, 2, ...) yields the same layout that
            # all_gather_into_tensor into a (R, 2, L_local, H_kv, D) buffer
            # would have produced.
            kv_recv = attn2d_col_allgather(kv_send, mapping, dim=0).view(R, 2, L_local, H_kv, D)
        # R == 1: no redistribute or col-gather needed; k/v are already the
        # sorted tensors used by the Q-split branch below.

        # --- 4) Local attention.  K/V layout is path-specific:
        #   Q-split    -> full sorted (k_sorted, v_sorted) via one permute
        #   K-split    -> per-u contiguous (K_u, V_u) views into one
        #                 permute -- no second copy in the per-u loop
        #   fallback   -> source-rank-grouped (k_full, v_full)
        if C % R == 0:
            # Q-split path (includes square mesh as s == 1).
            # Each Q_t is the sorted Q at stride s, with positions
            # advancing by C per index -- matching K.  Per-Q_t mask is
            #   (k - m) * C <= t*R + row_idx - col_idx
            # which is causal (RHS >= 0) or strict-causal (RHS < 0).
            # Outputs cover disjoint Q positions -> no LSE merge across t.
            if R > 1:
                # (R, 2, L_local, H_kv, D) -> (2, L_local, R, H_kv, D)
                # -> (2, L_k, H_kv, D).  Flat L_k index l*R + r maps to
                # position (l*R + r)*C + col_idx; slicing dim 0 at 0/1
                # gives contiguous K / V.
                kv_sorted = kv_recv.permute(1, 2, 0, 3, 4).contiguous().view(2, L_k, H_kv, D)
                k_sorted = kv_sorted[0]
                v_sorted = kv_sorted[1]
            else:
                k_sorted = k
                v_sorted = v
            s = C // R
            output_send = q.new_empty(R, s, L_local, H_q, D)
            lse_send = q.new_empty(R, s, L_local, H_q, dtype=torch.float32)
            q_full_view = q_full.view(C, L_local, H_q, D)
            for t in range(s):
                # q_full_view[t::s] selects R blocks at strided c indices;
                # transpose puts the slow dim (L_local) before the stride
                # dim (R), giving sorted Q_t order in the flat layout.
                Q_t = q_full_view[t::s].transpose(0, 1).contiguous().view(L_k, H_q, D)
                strict = (t * R + row_idx) < col_idx
                if strict:
                    Q_in = torch.cat([Q_t, Q_t.new_zeros(1, H_q, D)], dim=0)
                else:
                    Q_in = Q_t
                out_t, lse_t = flashinfer.single_prefill_with_kv_cache(
                    Q_in,
                    k_sorted,
                    v_sorted,
                    causal=True,
                    kv_layout="NHD",
                    return_lse=True,
                )
                if strict:
                    out_t = out_t[:-1]
                    lse_t = lse_t[:-1]
                # Write directly into the source-rank-grouped layout.
                # out_t at sorted-Q_t index m' = i*R + k goes to
                # output_send[k, t, i].
                output_send[:, t] = out_t.view(L_local, R, H_q, D).transpose(0, 1)
                lse_send[:, t] = lse_t.view(L_local, R, H_q).transpose(0, 1)
            output = output_send.view(C * L_local, H_q, D)
            lse = lse_send.view(C * L_local, H_q)
        elif R % C == 0:
            # K-split path (R == C falls into the Q-split branch above,
            # so here s >= 2 and R > 1).  Each K_u is the sorted K at
            # stride s, with positions advancing by R per index --
            # matching Q.  Per-K_u mask is
            #   (k - m) * R <= row_idx - col_idx - u*C
            # again clean causal / strict-causal.  Each call is a partial
            # attention over a K-shard, so partials are LSE-merged
            # across u.
            s = R // C
            # Fuse "split into K/V" with "interleave to sorted K_u order"
            # into ONE permute+contiguous, then slice per-u contiguous
            # views.  View as (C, s, ...) splits the R axis as (c_in_R,
            # u) so r = c_in_R * s + u; permuting (u, t, l, c_in_R, h, d)
            # and flattening (l, c_in_R) -> L_q gives sorted K_u order
            # k'_u = l*C + c_in_R.  kv_by_u[u, t] is a contiguous (L_q,
            # H_kv, D) view -- no extra copy in the per-u loop.
            kv_by_u = (
                kv_recv.view(C, s, 2, L_local, H_kv, D)
                .permute(1, 2, 3, 0, 4, 5)
                .contiguous()
                .view(s, 2, L_q, H_kv, D)
            )
            q_sorted = (
                q_full.view(C, L_local, H_q, D).transpose(0, 1).contiguous().view(L_q, H_q, D)
            )
            v_stack = q.new_empty(L_q, s, H_q, D)
            s_stack = q.new_empty(L_q, s, H_q, dtype=torch.float32)
            for u in range(s):
                strict = row_idx < (col_idx + u * C)
                if strict:
                    Q_in = torch.cat([q_sorted, q_sorted.new_zeros(1, H_q, D)], dim=0)
                else:
                    Q_in = q_sorted
                out_u, lse_u = flashinfer.single_prefill_with_kv_cache(
                    Q_in,
                    kv_by_u[u, 0],
                    kv_by_u[u, 1],
                    causal=True,
                    kv_layout="NHD",
                    return_lse=True,
                )
                if strict:
                    out_u = out_u[:-1]
                    lse_u = lse_u[:-1]
                v_stack[:, u] = out_u
                s_stack[:, u] = lse_u
            output_sorted, lse_sorted = flashinfer.merge_states(v_stack, s_stack)
            # Inverse interleave both back to source-rank-grouped order
            # so the row-group all_to_all below picks them up unchanged.
            output = (
                output_sorted.view(L_local, C, H_q, D)
                .transpose(0, 1)
                .contiguous()
                .view(L_q, H_q, D)
            )
            lse = lse_sorted.view(L_local, C, H_q).transpose(0, 1).contiguous().view(L_q, H_q)
        else:
            # Custom-mask fallback: neither R % C == 0 nor C % R == 0
            # (e.g. R = 3, C = 2, or R = 4, C = 6).  Build the
            # absolute-position mask directly on the source-rank-grouped
            # K/V.  R > 1 is guaranteed (R == 1 always hits Q-split).
            k_full = kv_recv[:, 0].reshape(L_k, H_kv, D)
            v_full = kv_recv[:, 1].reshape(L_k, H_kv, D)
            # Row-gather concatenates source ranks in row-group order
            # (col_idx_src = 0..C-1); source rank col_idx_src has cp_rank
            # = col_idx_src*R + row_idx, so its local token at index i
            # maps to position col_idx_src*R + row_idx + i*P.
            col_src = torch.arange(C, device=device).repeat_interleave(L_local)
            local_i = torch.arange(L_local, device=device).repeat(C)
            q_pos = col_src * R + row_idx + local_i * P  # [L_q]

            # After the row-major redistribution, source rank at
            # row_idx_src in col_pg holds positions
            # {p: p%P == row_idx_src*C + col_idx}, so its local token at
            # index j maps to position col_idx + row_idx_src*C + j*P.
            row_src = torch.arange(R, device=device).repeat_interleave(L_local)
            local_j = torch.arange(L_local, device=device).repeat(R)
            k_pos = col_idx + row_src * C + local_j * P  # [L_k]

            custom_mask = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
            output, lse = flashinfer.single_prefill_with_kv_cache(
                q_full,
                k_full,
                v_full,
                custom_mask=custom_mask,
                kv_layout="NHD",
                return_lse=True,
            )
        # output: [L_q, H_q, D] in source-rank-grouped order;
        # lse:    [L_q, H_q] (natural log).

        # --- 5) Row-group all_to_all + LSE-merge of the C partials.
        if C > 1:
            # output: [C*L_local, H_q, D] grouped by destination col_idx
            # in the row group.  After all_to_all, o_recv[c, i] is the
            # partial from source col_idx=c for this rank's local token i.
            # attn2d_row_alltoall takes a multi-list input (C tensors per
            # list); we pass [o_chunks..., lse_chunks...] and unpack two
            # outputs.
            o_chunks = list(output.view(C, L_local, H_q, D).contiguous().unbind(0))
            lse_chunks = list(lse.view(C, L_local, H_q).contiguous().unbind(0))
            o_recv, lse_recv = attn2d_row_alltoall(o_chunks + lse_chunks, mapping)
            # o_recv:   (C, L_local, H_q, D)
            # lse_recv: (C, L_local, H_q)

            # merge_states wants (seq_len, num_states, ...).  C is the
            # state count; L_local is the seq_len for this rank.
            v_stack = o_recv.permute(1, 0, 2, 3).contiguous()
            s_stack = lse_recv.permute(1, 0, 2).contiguous()
            out_merged, _ = flashinfer.merge_states(v_stack, s_stack)
            # Slice off padded rows: those carried virtual positions
            # >= L_total and were ignored by the causal mask.
            return out_merged[:L_local_unpadded]

        # C == 1: this rank's L_local tokens are already complete.
        return output[:L_local_unpadded]
