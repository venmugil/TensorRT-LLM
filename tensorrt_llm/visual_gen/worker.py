# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Worker entry point for rank > 0 in multi-node VisualGen distributed inference.

Invoked by trtllm-visualgen-launch on every rank except 0.  Rank 0 runs the
user's command (trtllm-serve or a user script) while all other ranks run this
module.  Requests are broadcast from rank 0's diffusion worker via
torch.distributed; no ZMQ sockets are used here.

Required environment variables (set by SLURM/torchrun automatically):
    MASTER_ADDR   Hostname or IP of the rank-0 node.
    SLURM_PROCID / RANK        Global rank of this process.
    SLURM_LOCALID / LOCAL_RANK Local rank within the node.
    SLURM_NTASKS / WORLD_SIZE  Total number of ranks.
"""

import argparse
import os

from tensorrt_llm._torch.visual_gen.executor import run_diffusion_worker
from tensorrt_llm.logger import logger
from tensorrt_llm.visual_gen.args import VisualGenArgs


def main():
    parser = argparse.ArgumentParser(description="VisualGen distributed worker (rank > 0)")
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a VisualGen YAML config file (parallel settings, teacache, etc.)",
    )
    parser.add_argument("--log-level", default="info", help="Logging level")
    args = parser.parse_args()

    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", 0)))
    local_rank = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", rank)))
    world_size = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", 1)))
    master_addr = os.environ.get("MASTER_ADDR")
    master_port = int(os.environ.get("MASTER_PORT", 29500))

    if master_addr is None:
        raise RuntimeError(
            "MASTER_ADDR must be set for multi-node VisualGen workers. "
            "Add to your sbatch script:\n"
            "  export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1)"
        )

    visual_gen_args = VisualGenArgs.from_yaml(args.config, checkpoint_path=args.model)

    logger.info(
        f"VisualGen worker: rank={rank}/{world_size}, local_rank={local_rank}, "
        f"master={master_addr}:{master_port}"
    )

    run_diffusion_worker(
        rank=rank,
        world_size=world_size,
        master_addr=master_addr,
        master_port=master_port,
        request_queue_addr=None,
        response_queue_addr=None,
        diffusion_args=visual_gen_args,
        log_level=args.log_level,
        req_hmac_key=None,
        resp_hmac_key=None,
        local_rank=local_rank,
    )


if __name__ == "__main__":
    main()
