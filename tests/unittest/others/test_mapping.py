# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import unittest

from tensorrt_llm.mapping import CpType, Mapping


class TestMapping(unittest.TestCase):

    def test_mapping(self):
        m = Mapping(world_size=8, rank=0, tp_size=8)
        self.assertEqual(len(m.tp_groups), 1)
        self.assertEqual(len(m.pp_groups), 8)
        self.assertEqual(m.tp_group, [0, 1, 2, 3, 4, 5, 6, 7])

        m = Mapping(world_size=8, rank=0, tp_size=4, pp_size=2)
        self.assertEqual(len(m.tp_groups), 2)
        self.assertEqual(len(m.pp_groups), 4)
        self.assertEqual(m.tp_group, [0, 1, 2, 3])
        self.assertEqual(m.pp_group, [0, 4])
        self.assertTrue(m.is_first_pp_rank())
        self.assertFalse(m.is_last_pp_rank())
        self.assertEqual(m.prev_pp_rank(), 4)
        self.assertEqual(m.next_pp_rank(), 4)

        m = Mapping(world_size=8, rank=6, tp_size=2, pp_size=4)
        self.assertEqual(len(m.tp_groups), 4)
        self.assertEqual(len(m.pp_groups), 2)
        self.assertEqual(m.tp_group, [6, 7])
        self.assertEqual(m.pp_group, [0, 2, 4, 6])
        self.assertFalse(m.is_first_pp_rank())
        self.assertTrue(m.is_last_pp_rank())
        self.assertEqual(m.prev_pp_rank(), 4)
        self.assertEqual(m.next_pp_rank(), 0)

        m = Mapping(world_size=2, rank=0, cp_size=2)
        self.assertEqual(len(m.tp_groups), 2)
        self.assertEqual(len(m.pp_groups), 2)
        self.assertEqual(len(m.cp_groups), 1)
        self.assertEqual(m.tp_group, [0])
        self.assertEqual(m.pp_group, [0])
        self.assertEqual(m.cp_group, [0, 1])

        m = Mapping(world_size=8, rank=3, tp_size=2, pp_size=2, cp_size=2)
        self.assertEqual(len(m.tp_groups), 4)
        self.assertEqual(len(m.pp_groups), 4)
        self.assertEqual(len(m.cp_groups), 4)
        self.assertEqual(m.tp_group, [1, 3])
        self.assertEqual(m.pp_group, [3, 7])
        self.assertEqual(m.cp_group, [2, 3])
        self.assertTrue(m.is_first_pp_rank())
        self.assertFalse(m.is_last_pp_rank())
        self.assertFalse(m.is_first_cp_rank())
        self.assertTrue(m.is_last_cp_rank())
        self.assertEqual(m.prev_pp_rank(), 7)
        self.assertEqual(m.next_pp_rank(), 7)
        self.assertEqual(m.prev_cp_rank(), 2)
        self.assertEqual(m.next_cp_rank(), 2)

        m = Mapping(world_size=16, rank=9, tp_size=2, pp_size=2, cp_size=4)
        self.assertEqual(m.tp_group, [9, 13])
        self.assertEqual(m.pp_group, [1, 9])
        self.assertEqual(m.cp_group, [8, 9, 10, 11])
        self.assertFalse(m.is_first_pp_rank())
        self.assertTrue(m.is_last_pp_rank())
        self.assertFalse(m.is_first_cp_rank())
        self.assertFalse(m.is_last_cp_rank())
        self.assertEqual(m.prev_pp_rank(), 1)
        self.assertEqual(m.next_pp_rank(), 1)
        self.assertEqual(m.prev_cp_rank(), 8)
        self.assertEqual(m.next_cp_rank(), 10)


def _make_attn2d(rank,
                 world_size,
                 tp_size,
                 cp_size,
                 row_size,
                 col_size,
                 enable_adp=False):
    """Helper: build an ATTN2D Mapping (pp=1) and return it."""
    return Mapping(
        world_size=world_size,
        rank=rank,
        tp_size=tp_size,
        cp_size=cp_size,
        cp_config={
            "cp_type": CpType.ATTN2D,
            "row_size": row_size,
            "col_size": col_size,
        },
        enable_attention_dp=enable_adp,
    )


class TestAttn2dMoeMapping(unittest.TestCase):
    """CPU-only tests for the ATTN2D CP→MoE-EP mapping derivation.

    ATTN2D repurposes the CP dimension as MoE EP.  Four cases:

        Case A  tp=1, cp=P (no ADP)   → moe_ep=P,   moe_tp=1  ✅ correct
        Case B  tp=1, cp=P (ADP)      → moe_ep=P,   moe_tp=1  ✅ correct (same: 1*P=P)
        Case C  tp=T, cp=P (no ADP)   → moe_ep=P,   moe_tp=T  ❌ broken comm (see below)
        Case D  tp=T, cp=P (ADP)      → moe_ep=T*P, moe_tp=1  ✅ fixed (was ⚠️ suspect)

    Case C remains broken (see comment in test).

    Case D fix: under ATTN2D+ADP all tp×cp ranks hold mutually-distinct
    tokens (tp_rank selects requests, cp_rank selects positions), so the
    full PP stage is one flat EP group.  Setting moe_ep_size=tp×cp aligns
    the mapping with what DeepEP/NVLinkOneSided already construct
    (Split(pp_rank,…) spans tp×cp; NVLinkOneSided asserts world==ep).
    moe_ep_rank is tp-major, cp-minor (tp_rank*cp_size + cp_rank) to match
    tp_cp_allgather ordering used for the token-count scatter.
    """

    # ------------------------------------------------------------------
    # 1. Size derivation
    # ------------------------------------------------------------------

    def test_attn2d_moe_sizes_tp1(self):
        """Sizes for tp=1, cp=4 (R=2, C=2) without and with ADP."""
        for enable_adp in (False, True):
            with self.subTest(enable_adp=enable_adp):
                m = _make_attn2d(rank=0,
                                 world_size=4,
                                 tp_size=1,
                                 cp_size=4,
                                 row_size=2,
                                 col_size=2,
                                 enable_adp=enable_adp)
                # CP is repurposed as EP; TP does not contribute.
                self.assertEqual(m.moe_ep_size, 4)  # == cp_size
                self.assertEqual(m.moe_tp_size, 1)  # tp=1 so always 1
                self.assertEqual(m.moe_cluster_size, 1)
                # moe_world = ep * tp * cluster
                self.assertEqual(m.moe_tp_cluster_ep_size, 4)

    def test_attn2d_moe_sizes_tp2_noadp(self):
        """Sizes for tp=2, cp=2 (R=2, C=1) without ADP.

        moe_tp=tp_size=2, moe_ep=cp_size=2.
        This config is ❌ broken for MoE communication (moe_tp>1 forces
        AllGatherReduceScatter over tp_group, skipping the EP exchange
        across CP).  Sizes are asserted here to lock in the derivation and
        detect regressions when the comm backends are eventually fixed.
        """
        m = _make_attn2d(rank=0,
                         world_size=4,
                         tp_size=2,
                         cp_size=2,
                         row_size=2,
                         col_size=1)
        self.assertEqual(m.moe_ep_size, 2)  # == cp_size
        self.assertEqual(m.moe_tp_size, 2)  # == tp_size (no ADP)
        self.assertEqual(m.moe_cluster_size, 1)
        self.assertEqual(m.moe_tp_cluster_ep_size, 4)  # tp * ep

    def test_attn2d_moe_sizes_tp2_adp(self):
        """Sizes for tp=2, cp=2 (R=2, C=1) with ADP.

        With ADP, all tp×cp ranks hold distinct tokens, so the full PP stage
        is one flat EP group: moe_ep=tp*cp=4, moe_tp=1.  This aligns with
        what DeepEP/NVLinkOneSided already construct (Split spans the stage).
        """
        m = _make_attn2d(rank=0,
                         world_size=4,
                         tp_size=2,
                         cp_size=2,
                         row_size=2,
                         col_size=1,
                         enable_adp=True)
        self.assertEqual(m.moe_ep_size, 4)  # == tp_size * cp_size
        self.assertEqual(m.moe_tp_size, 1)  # ADP: always 1
        self.assertEqual(m.moe_cluster_size, 1)
        self.assertEqual(m.moe_tp_cluster_ep_size, 4)  # ep=4, tp=1, cluster=1

    # ------------------------------------------------------------------
    # 2. Rank accessors — iterate every rank in a small world
    # ------------------------------------------------------------------

    def test_attn2d_moe_rank_accessors_tp1_noadp(self):
        """moe_ep_rank == cp_rank and moe_tp_rank == 0 for all ranks (tp=1)."""
        P = 4
        for rank in range(P):
            with self.subTest(rank=rank):
                m = _make_attn2d(rank=rank,
                                 world_size=P,
                                 tp_size=1,
                                 cp_size=P,
                                 row_size=2,
                                 col_size=2)
                # For tp=1,pp=1: cp_rank == rank
                self.assertEqual(m.cp_rank, rank)
                self.assertEqual(m.tp_rank, 0)
                self.assertEqual(m.moe_ep_rank, rank)  # == cp_rank
                self.assertEqual(m.moe_tp_rank, 0)  # tp_rank (=0 since tp=1)
                self.assertEqual(m.moe_cluster_rank, 0)

    def test_attn2d_moe_rank_accessors_tp2_noadp(self):
        """moe_ep_rank == cp_rank and moe_tp_rank == tp_rank for all ranks (tp=2, no ADP).

        This is the ❌ broken-comm case; rank accessors are still correct.
        """
        # world=4, tp=2, cp=2, pp=1: rank layout below.
        # tp_rank  = rank // cp_size = rank // 2  → {0:0, 1:0, 2:1, 3:1}
        # cp_rank  = rank  % cp_size = rank  % 2  → {0:0, 1:1, 2:0, 3:1}
        expected = {
            #  rank: (cp_rank, tp_rank, moe_ep_rank, moe_tp_rank)
            0: (0, 0, 0, 0),
            1: (1, 0, 1, 0),
            2: (0, 1, 0, 1),
            3: (1, 1, 1, 1),
        }
        for rank, (cp_rank, tp_rank, ep_rank, moe_tp_rank) in expected.items():
            with self.subTest(rank=rank):
                m = _make_attn2d(rank=rank,
                                 world_size=4,
                                 tp_size=2,
                                 cp_size=2,
                                 row_size=2,
                                 col_size=1)
                self.assertEqual(m.cp_rank, cp_rank)
                self.assertEqual(m.tp_rank, tp_rank)
                self.assertEqual(m.moe_ep_rank, ep_rank)  # == cp_rank
                self.assertEqual(m.moe_tp_rank,
                                 moe_tp_rank)  # == tp_rank (no ADP)
                self.assertEqual(m.moe_cluster_rank, 0)

    def test_attn2d_moe_rank_accessors_tp2_adp(self):
        """moe_ep_rank == tp_rank*cp_size + cp_rank for all ranks (tp=2, ADP).

        ADP fix: moe_ep spans the full PP stage (tp×cp), so moe_ep_rank is
        tp-major, cp-minor to match tp_cp_allgather ordering.
        For world=4, tp=2, cp=2, pp=1:
          tp_rank  = rank // cp_size = rank // 2  → {0:0, 1:0, 2:1, 3:1}
          cp_rank  = rank  % cp_size = rank  % 2  → {0:0, 1:1, 2:0, 3:1}
          moe_ep_rank = tp_rank*cp_size + cp_rank  → {0:0, 1:1, 2:2, 3:3}
        """
        expected = {
            #  rank: (cp_rank, tp_rank, moe_ep_rank)
            0: (0, 0, 0),
            1: (1, 0, 1),
            2: (0, 1, 2),
            3: (1, 1, 3),
        }
        for rank, (cp_rank, tp_rank, moe_ep_rank) in expected.items():
            with self.subTest(rank=rank):
                m = _make_attn2d(rank=rank,
                                 world_size=4,
                                 tp_size=2,
                                 cp_size=2,
                                 row_size=2,
                                 col_size=1,
                                 enable_adp=True)
                self.assertEqual(m.cp_rank, cp_rank)
                self.assertEqual(m.tp_rank, tp_rank)
                self.assertEqual(m.moe_ep_rank,
                                 moe_ep_rank)  # tp_rank*cp_size + cp_rank
                self.assertEqual(m.moe_tp_rank, 0)  # ADP: always 0
                self.assertEqual(m.moe_cluster_rank, 0)
                # all_rank_num_tokens_rank must match moe_ep_rank for ATTN2D+ADP
                self.assertEqual(m.all_rank_num_tokens_rank, moe_ep_rank)

    # ------------------------------------------------------------------
    # 3. Group membership — verify delegation to cp_group / tp_group
    # ------------------------------------------------------------------

    def test_attn2d_moe_ep_group_equals_cp_group_noadp(self):
        """Without ADP, moe_ep_group must equal cp_group for ATTN2D."""
        # tp=2, cp=2: cp_groups = [[0,1], [2,3]] (one per TP rank).
        # rank 0 (tp_rank=0): cp_group=[0,1]; rank 2 (tp_rank=1): cp_group=[2,3].
        for rank, expected_cp in [(0, [0, 1]), (1, [0, 1]), (2, [2, 3]),
                                  (3, [2, 3])]:
            with self.subTest(rank=rank):
                m = _make_attn2d(rank=rank,
                                 world_size=4,
                                 tp_size=2,
                                 cp_size=2,
                                 row_size=2,
                                 col_size=1)
                self.assertEqual(m.cp_group, expected_cp)
                self.assertEqual(m.moe_ep_group, m.cp_group)

    def test_attn2d_moe_tp_group_noadp(self):
        """moe_tp_group == tp_group (not ADP) for ATTN2D."""
        # tp=2, cp=2: tp_groups = [[0,2], [1,3]] (stride-cp_size).
        for rank, expected_tp in [(0, [0, 2]), (1, [1, 3]), (2, [0, 2]),
                                  (3, [1, 3])]:
            with self.subTest(rank=rank):
                m = _make_attn2d(rank=rank,
                                 world_size=4,
                                 tp_size=2,
                                 cp_size=2,
                                 row_size=2,
                                 col_size=1)
                self.assertEqual(m.tp_group, expected_tp)
                self.assertEqual(m.moe_tp_group, m.tp_group)

    def test_attn2d_moe_tp_group_adp(self):
        """moe_tp_group == [rank] (singleton) with ADP for ATTN2D."""
        for rank in range(4):
            with self.subTest(rank=rank):
                m = _make_attn2d(rank=rank,
                                 world_size=4,
                                 tp_size=2,
                                 cp_size=2,
                                 row_size=2,
                                 col_size=1,
                                 enable_adp=True)
                self.assertEqual(m.moe_tp_group, [rank])

    def test_attn2d_moe_tp1_ep_group_is_full_cp_group(self):
        """For tp=1, moe_ep_group is the full CP group (all P ranks)."""
        P = 4
        for rank in range(P):
            with self.subTest(rank=rank):
                m = _make_attn2d(rank=rank,
                                 world_size=P,
                                 tp_size=1,
                                 cp_size=P,
                                 row_size=2,
                                 col_size=2)
                self.assertEqual(m.moe_ep_group, list(range(P)))
                self.assertEqual(m.moe_tp_group, [rank])  # tp_group singleton

    # ------------------------------------------------------------------
    # 4. No stale moe group lists: _init_parallel_groups early-returns for ATTN2D
    # ------------------------------------------------------------------

    def test_attn2d_moe_group_lists_are_empty(self):
        """MoE group lists stay empty because _init_parallel_groups early-returns for ATTN2D.

        The MPI path (MpiTopology) builds pp/cp/tp group lists but returns
        early before building moe_tp_groups / moe_ep_groups / moe_cluster_groups.
        The accessor properties delegate to cp_group / tp_group instead.
        """
        m = _make_attn2d(rank=0,
                         world_size=4,
                         tp_size=2,
                         cp_size=2,
                         row_size=2,
                         col_size=1)
        # These lists are only populated for non-ATTN2D mappings.
        self.assertEqual(m.moe_tp_groups, [])
        self.assertEqual(m.moe_ep_groups, [])
        self.assertEqual(m.moe_cluster_groups, [])
        # pp/cp/tp lists ARE populated.
        self.assertGreater(len(m.tp_groups), 0)
        self.assertGreater(len(m.cp_groups), 0)

    # ------------------------------------------------------------------
    # 5. Constructor guards
    # ------------------------------------------------------------------

    def test_attn2d_requires_row_col_in_cp_config(self):
        """Constructing ATTN2D without row_size / col_size raises ValueError."""
        with self.assertRaises(ValueError):
            Mapping(world_size=4,
                    rank=0,
                    cp_size=4,
                    cp_config={"cp_type": CpType.ATTN2D})

    def test_attn2d_row_col_product_must_equal_cp_size(self):
        """row_size * col_size != cp_size raises ValueError."""
        with self.assertRaises(ValueError):
            Mapping(world_size=4,
                    rank=0,
                    cp_size=4,
                    cp_config={
                        "cp_type": CpType.ATTN2D,
                        "row_size": 3,
                        "col_size": 2
                    })

    def test_attn2d_rejects_moe_cluster_size_gt1(self):
        """moe_cluster_size > 1 with ATTN2D raises ValueError."""
        with self.assertRaises(ValueError):
            Mapping(world_size=4,
                    rank=0,
                    cp_size=4,
                    cp_config={
                        "cp_type": CpType.ATTN2D,
                        "row_size": 2,
                        "col_size": 2
                    },
                    moe_cluster_size=2)

    def test_attn2d_rejects_mismatched_explicit_moe_ep(self):
        """Explicit moe_ep_size != cp_size raises ValueError for ATTN2D."""
        with self.assertRaises(ValueError):
            Mapping(
                world_size=4,
                rank=0,
                tp_size=1,
                cp_size=4,
                cp_config={
                    "cp_type": CpType.ATTN2D,
                    "row_size": 2,
                    "col_size": 2
                },
                moe_ep_size=2,  # wrong: should equal cp_size=4
                moe_tp_size=1)

    def test_attn2d_rejects_mismatched_explicit_moe_tp_noadp(self):
        """Explicit moe_tp_size != tp_size (no ADP) raises ValueError for ATTN2D."""
        with self.assertRaises(ValueError):
            Mapping(
                world_size=4,
                rank=0,
                tp_size=2,
                cp_size=2,
                cp_config={
                    "cp_type": CpType.ATTN2D,
                    "row_size": 2,
                    "col_size": 1
                },
                moe_ep_size=2,  # correct
                moe_tp_size=1)  # wrong: no-ADP requires moe_tp==tp_size=2

    def test_attn2d_rejects_mismatched_explicit_moe_tp_adp(self):
        """Explicit moe_tp_size != 1 with ADP raises ValueError for ATTN2D."""
        with self.assertRaises(ValueError):
            Mapping(
                world_size=4,
                rank=0,
                tp_size=2,
                cp_size=2,
                cp_config={
                    "cp_type": CpType.ATTN2D,
                    "row_size": 2,
                    "col_size": 1
                },
                enable_attention_dp=True,
                moe_ep_size=4,  # correct (tp*cp)
                moe_tp_size=2)  # wrong: ADP requires moe_tp=1

    def test_attn2d_rejects_wrong_ep_size_adp(self):
        """ADP: moe_ep_size != tp*cp raises ValueError for ATTN2D."""
        with self.assertRaises(ValueError):
            Mapping(
                world_size=4,
                rank=0,
                tp_size=2,
                cp_size=2,
                cp_config={
                    "cp_type": CpType.ATTN2D,
                    "row_size": 2,
                    "col_size": 1
                },
                enable_attention_dp=True,
                moe_ep_size=2,  # wrong: ADP requires tp*cp=4
                moe_tp_size=1)

    def test_attn2d_accepts_explicit_correct_moe_sizes_tp2_adp(self):
        """Explicitly passing moe_ep_size=tp*cp, moe_tp_size=1 with ADP is accepted."""
        m = Mapping(
            world_size=4,
            rank=0,
            tp_size=2,
            cp_size=2,
            cp_config={
                "cp_type": CpType.ATTN2D,
                "row_size": 2,
                "col_size": 1
            },
            enable_attention_dp=True,
            moe_ep_size=4,  # == tp_size * cp_size ✓
            moe_tp_size=1)  # ADP: always 1 ✓
        self.assertEqual(m.moe_ep_size, 4)
        self.assertEqual(m.moe_tp_size, 1)

    def test_attn2d_accepts_explicit_correct_moe_sizes_tp1(self):
        """Explicitly passing the correct moe sizes for tp=1 ATTN2D is accepted."""
        m = Mapping(
            world_size=4,
            rank=0,
            tp_size=1,
            cp_size=4,
            cp_config={
                "cp_type": CpType.ATTN2D,
                "row_size": 2,
                "col_size": 2
            },
            moe_ep_size=4,  # == cp_size ✓
            moe_tp_size=1)  # == tp_size ✓
        self.assertEqual(m.moe_ep_size, 4)
        self.assertEqual(m.moe_tp_size, 1)

    def test_attn2d_accepts_explicit_correct_moe_sizes_tp2_noadp(self):
        """Explicitly passing the correct moe sizes for tp=2 ATTN2D (no ADP) is accepted."""
        m = Mapping(
            world_size=4,
            rank=0,
            tp_size=2,
            cp_size=2,
            cp_config={
                "cp_type": CpType.ATTN2D,
                "row_size": 2,
                "col_size": 1
            },
            moe_ep_size=2,  # == cp_size ✓
            moe_tp_size=2)  # == tp_size ✓
        self.assertEqual(m.moe_ep_size, 2)
        self.assertEqual(m.moe_tp_size, 2)

    # ------------------------------------------------------------------
    # 6. Comm-strategy documentation (CPU-only; no actual alltoall)
    # ------------------------------------------------------------------

    def test_attn2d_all_rank_num_tokens_rank_noadp(self):
        """Without ADP, all_rank_num_tokens_rank == tp_rank (standard tp_allgather)."""
        for rank in range(4):
            with self.subTest(rank=rank):
                m = _make_attn2d(rank=rank,
                                 world_size=4,
                                 tp_size=2,
                                 cp_size=2,
                                 row_size=2,
                                 col_size=1)
                self.assertEqual(m.all_rank_num_tokens_rank, m.tp_rank)

    def test_attn2d_tp1_moe_tp_is_1_selects_alltoall_path(self):
        """For tp=1 ATTN2D, moe_tp_size==1 so the comm factory skips AllGatherReduceScatter.

        The actual strategy selected (NVLink, DeepEP, or AllGather fallback)
        depends on hardware; this test only verifies the mapping attribute that
        drives the selection.  moe_tp_size==1 is the necessary condition for
        an EP-alltoall strategy to be considered.
        """
        P = 4
        for enable_adp in (False, True):
            with self.subTest(enable_adp=enable_adp):
                m = _make_attn2d(rank=0,
                                 world_size=P,
                                 tp_size=1,
                                 cp_size=P,
                                 row_size=2,
                                 col_size=2,
                                 enable_adp=enable_adp)
                # moe_tp_size == 1 → factory does NOT return AllGatherReduceScatter
                # immediately; it tries NVLink/DeepEP first.
                self.assertEqual(m.moe_tp_size, 1)
                # ATTN2D exempts the early-return in the factory so comm is
                # always created (has_cp_attn2d() is True).
                self.assertTrue(m.has_cp_attn2d())

    def test_attn2d_tp2_noadp_moe_tp2_forces_allgather(self):
        """For tp=2 ATTN2D (no ADP), moe_tp_size==2 forces AllGatherReduceScatter.

        This is the ❌ broken-comm configuration: the comm factory returns
        AllGatherReduceScatter (mapping.py-level equivalent is moe_tp_size != 1),
        which dispatches over mapping.tp_group — but tp_group is NOT the EP group.
        The EP exchange across CP never happens.

        This test documents the broken condition via mapping attributes only.
        """
        m = _make_attn2d(rank=0,
                         world_size=4,
                         tp_size=2,
                         cp_size=2,
                         row_size=2,
                         col_size=1)
        # moe_tp_size > 1 triggers AllGatherReduceScatter in the factory.
        self.assertEqual(m.moe_tp_size, 2)
        # tp_group and moe_ep_group (==cp_group) are DIFFERENT groups:
        # tp_group spans TP ranks (stride-cp_size), ep_group spans CP ranks.
        # AllGatherReduceScatter allgathers over tp_group — wrong for EP.
        self.assertNotEqual(m.tp_group, m.moe_ep_group)

    def test_attn2d_tp2_adp_moe_ep_group_is_full_stage(self):
        """For tp=2 ATTN2D (ADP), moe_ep_group is the full PP stage (all tp×cp ranks).

        ADP fix: moe_ep_size=tp*cp=4, moe_ep_group=[0,1,2,3] for all ranks.
        This matches what DeepEP/NVLinkOneSided already construct:
          - DeepEP Split(color=pp_rank=0, …) → one group of 4 = tp*cp ✓
          - NVLinkOneSided: world_size == moe_ep_size (4==4) ✓
        """
        for rank in range(4):
            with self.subTest(rank=rank):
                m = _make_attn2d(rank=rank,
                                 world_size=4,
                                 tp_size=2,
                                 cp_size=2,
                                 row_size=2,
                                 col_size=1,
                                 enable_adp=True)
                self.assertEqual(m.moe_ep_size, 4)  # tp*cp
                self.assertEqual(m.moe_tp_size, 1)
                self.assertEqual(m.moe_ep_group, [0, 1, 2, 3])  # full PP stage
                # Mapping now agrees with backend group construction.
                self.assertEqual(m.world_size, m.moe_ep_size)  # 4 == 4
