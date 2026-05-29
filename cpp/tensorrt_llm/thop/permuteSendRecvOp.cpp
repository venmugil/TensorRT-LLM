/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <iterator>
#include <set>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{
#if ENABLE_MULTI_DEVICE

namespace
{

// Translate a world rank to its index in the sorted ``group`` set, which
// is the local rank within the NCCL comm returned by getComm(group).
int worldRankToLocal(std::set<int> const& group, int worldRank)
{
    auto it = group.find(worldRank);
    TLLM_CHECK_WITH_INFO(it != group.end(), "rank %d is not a member of the group", worldRank);
    return static_cast<int>(std::distance(group.begin(), it));
}

class PermuteSendRecvOp
{
public:
    PermuteSendRecvOp(std::set<int> group)
        : mGroup(std::move(group))
    {
    }

    ~PermuteSendRecvOp() = default;

    int initialize()
    {
        TLLM_LOG_TRACE("%s start for rank %d", __PRETTY_FUNCTION__, COMM_SESSION.getRank());
        TLLM_CHECK_WITH_INFO(mGroup.size() > 0, "group size should be greater than 0");
        mNcclComm = getComm(mGroup);
        TLLM_LOG_TRACE("%s stop for rank %d", __PRETTY_FUNCTION__, COMM_SESSION.getRank());
        return 0;
    }

    torch::Tensor run(torch::Tensor input, int targetWorldRank, int sourceWorldRank, c10::optional<int64_t> recvCount)
    {
        TLLM_CHECK_WITH_INFO(mNcclComm.get() != nullptr, "mNcclComm should be initialized before used");
        TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

        int const targetLocal = worldRankToLocal(mGroup, targetWorldRank);
        int const sourceLocal = worldRankToLocal(mGroup, sourceWorldRank);

        // When ``recvCount`` is unset, the receive buffer matches the send
        // buffer (today's symmetric P2P).  When it's set, dim 0 of the output
        // is overridden -- supports the asymmetric mesh-transpose used by
        // ATTN2D's chunked / multi-turn prefill, where paired ranks hold
        // different cyclic-shard counts.
        torch::Tensor output;
        if (recvCount.has_value())
        {
            auto sizes = input.sizes().vec();
            TORCH_CHECK(!sizes.empty(), "input must have at least 1 dimension when recv_count is set");
            sizes[0] = recvCount.value();
            output = torch::empty(sizes, input.options());
        }
        else
        {
            output = torch::empty_like(input);
        }

        auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
        auto type = tensorrt_llm::runtime::TorchUtils::dataType(input.scalar_type());
        auto ncclType = (*getDtypeMap())[type];

        ncclGroupStart();
        ncclSend(input.data_ptr(), input.numel(), ncclType, targetLocal, *mNcclComm, stream);
        ncclRecv(output.mutable_data_ptr(), output.numel(), ncclType, sourceLocal, *mNcclComm, stream);
        NCCLCHECK_THROW(ncclGroupEnd());
        return output;
    }

private:
    std::set<int> mGroup;
    std::shared_ptr<ncclComm_t> mNcclComm;
};

} // namespace

#endif // ENABLE_MULTI_DEVICE

// Each rank in ``group`` sends ``input`` to world rank ``target_rank`` and
// receives a same-dtype tensor from world rank ``source_rank``.
// ``target_rank`` and ``source_rank`` must both be members of ``group``;
// arbitrary peer pairings are allowed (it is the caller's responsibility to
// ensure the global send/recv pattern is well-formed -- i.e. each rank's
// source is some other rank's target).
//
// When ``recv_count`` is unset, the receive buffer has the same shape as the
// input (symmetric P2P).  When set, dim 0 of the output is overridden to
// ``recv_count`` while the trailing dims match the input.  This supports the
// asymmetric mesh-transpose required by ATTN2D's chunked / multi-turn
// prefill, where paired ranks may hold different cyclic-shard counts; the
// caller is responsible for ensuring the sender's element count equals the
// receiver's expected element count (the op only orchestrates the NCCL
// send/recv pair -- it cannot detect a size mismatch ahead of time).
torch::Tensor permute_send_recv(torch::Tensor input, int64_t target_rank, int64_t source_rank,
    torch::List<int64_t> group_, c10::optional<int64_t> recv_count)
{
#if ENABLE_MULTI_DEVICE
    std::set<int> group;
    for (int64_t rank : group_)
    {
        group.insert(static_cast<int>(rank));
    }
    PermuteSendRecvOp op(group);
    op.initialize();
    return op.run(input, static_cast<int>(target_rank), static_cast<int>(source_rank), recv_count);
#else
    return input;
#endif // ENABLE_MULTI_DEVICE
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "permute_send_recv(Tensor input, int target_rank, int source_rank, int[] group, int? recv_count=None) -> "
        "Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("permute_send_recv", &tensorrt_llm::torch_ext::permute_send_recv);
}
