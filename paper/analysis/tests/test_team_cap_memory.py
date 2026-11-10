"""Exact bounded cap counters and deterministic A2c 8/4-thread memory selection."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import a2c_team_cap as team


class TeamCapMemoryTests(unittest.TestCase):
    def test_chunked_counts_match_original_full_vectors(self):
        counts=np.array([2,3,5,50,51,500,1001,20000,2,3,8],np.int64)
        sizes=np.array([2,3,7,50,51,500,1001,30000,250,700,1000000],np.int32)
        offsets=np.r_[np.int64(0),np.cumsum(counts)]
        pairs=counts*(counts-1)//2
        expected={}
        for cap in team.CAPS:
            big=sizes>cap
            expected[cap]={'works_left_out':int(big.sum()),'authorships_left_out':int(counts[big].sum()),
                           'pairs_left_out':int(pairs[big].sum()),'pairs_kept':int(pairs[~big].sum()),
                           'max_researchers_on_a_work':int(counts[big].max()) if big.any() else 0}
        for batch in (1,3,100):
            self.assertEqual(team.work_cap_statistics(offsets,sizes,batch_size=batch),expected)
        empty=team.work_cap_statistics(np.array([0],np.int64),np.array([],np.int32),batch_size=1)
        self.assertTrue(all(value==0 for row in empty.values() for value in row.values()))

    def test_plan_uses_eight_or_four_and_rejects_an_unsafe_shared_graph(self):
        dimensions=dict(n=10_000_000,works=2_000_000,authorships=20_000_000,sources=1000,configured_threads=8,
                        headroom_bytes=100_000_000)
        fast=team.bfs_thread_plan(**dimensions,guard_bytes=1_400_000_000)
        self.assertEqual(fast['threads'],8)
        self.assertEqual(fast['shared_csr_bytes'],264_000_016)
        self.assertEqual(fast['private_bytes_per_worker'],102_000_000)
        self.assertLessEqual(fast['estimated_total_bytes'],fast['guard_bytes'])
        slower=team.bfs_thread_plan(**dimensions,guard_bytes=900_000_000)
        self.assertEqual(slower['threads'],4)
        self.assertLessEqual(slower['estimated_total_bytes'],slower['guard_bytes'])
        with self.assertRaisesRegex(MemoryError,'above'):
            team.bfs_thread_plan(**dimensions,guard_bytes=600_000_000)
        limited=team.bfs_thread_plan(**{**dimensions,'configured_threads':2},guard_bytes=900_000_000)
        self.assertEqual(limited['threads'],2)
        with self.assertRaisesRegex(ValueError,'Positive'):
            team.bfs_thread_plan(**{**dimensions,'configured_threads':0})


if __name__=='__main__':unittest.main()
