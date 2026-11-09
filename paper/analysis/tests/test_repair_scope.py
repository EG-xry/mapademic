"""Exact repaired-center screening with disconnected sources and defined distance denominators."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from center_cache import (CENTER_SELECTION_VERSION, reusable_center, screening_candidates,
                          source_with_largest_reach, verified_largest_component)
from kernels import _bfs, distance_sums, exact_mean, histograms_distance_sums
from repair_metrics import guarded_summary, group_summary, verify_cached_summary


def paths(sizes):
    adjacency=[]
    start=0
    for size in sizes:
        for i in range(size):
            adjacency.append(([start+i-1] if i else [])+([start+i+1] if i+1<size else []))
        start+=size
    offsets=np.r_[np.int64(0),np.cumsum([len(row) for row in adjacency])]
    return offsets,np.array([v for row in adjacency for v in row],np.int32)


def reach_from(offsets,neighbors,source):
    reach=np.empty(len(offsets)-1,np.int16)
    _bfs(offsets,neighbors,source,reach,np.empty(len(reach),np.int32))
    return reach


class RepairScopeTests(unittest.TestCase):
    def test_disconnected_source_is_excluded_only_from_center_screening(self):
        offsets,neighbors=paths([6,3])
        sources=np.array([6,0,2],np.int64)
        hist,sums,counts=histograms_distance_sums(offsets,neighbors,sources,64,2)
        self.assertEqual(hist.sum(axis=1).tolist(),[2,5,5])
        self.assertEqual(source_with_largest_reach(sources,hist),0)
        reach=reach_from(offsets,neighbors,source_with_largest_reach(sources,hist))
        estimates,best,metadata=screening_candidates(sources,reach,sums,counts,candidates=3)
        independent_sums,independent_counts=distance_sums(offsets,neighbors,np.array([0,2],np.int64),2)
        expected=np.where(independent_counts==2,independent_sums/2,np.inf)
        np.testing.assert_array_equal(estimates,expected)
        np.testing.assert_array_equal(best,np.argsort(expected)[:3])
        self.assertTrue(np.all(best<6))
        self.assertEqual(metadata,{"selection_version":CENTER_SELECTION_VERSION,"sources":2,
                                   "surviving_sources_total":3,"excluded_sources_outside_component":1,"component_size":6})
        # Existing all-source pair weighting keeps the minor component's observations.
        self.assertEqual(guarded_summary(hist,"fixture")["pairs"],12)
        # Computing only included sources (A8c fallback) gives the same screening result.
        fallback,fb_best,fb_meta=screening_candidates(sources,reach,independent_sums,independent_counts,candidates=3)
        np.testing.assert_array_equal(fallback,estimates)
        np.testing.assert_array_equal(fb_best,best)
        self.assertEqual(fb_meta,metadata)

    def test_one_component_matches_previous_estimates_order_and_exact_means(self):
        offsets,neighbors=paths([9])
        sources=np.array([6,0,2],np.int64)
        hist,sums,counts=histograms_distance_sums(offsets,neighbors,sources,64,2)
        reach=reach_from(offsets,neighbors,source_with_largest_reach(sources,hist))
        estimates,best,metadata=screening_candidates(sources,reach,sums,counts,candidates=4)
        old=np.where(counts==len(sources),sums/len(sources),np.inf)
        old_best=np.argsort(old)[:4].astype(np.int64)
        np.testing.assert_array_equal(estimates,old)
        np.testing.assert_array_equal(best,old_best)
        np.testing.assert_array_equal(exact_mean(offsets,neighbors,best),exact_mean(offsets,neighbors,old_best))
        self.assertEqual(metadata["sources"],len(sources))
        self.assertEqual(metadata["excluded_sources_outside_component"],0)

    def test_unproven_component_invalid_counts_and_old_cache_fail(self):
        offsets,neighbors=paths([6,3])
        sources=np.array([6,0,2],np.int64)
        _,sums,counts=histograms_distance_sums(offsets,neighbors,sources,64,2)
        reach=reach_from(offsets,neighbors,0)
        with self.assertRaisesRegex(ValueError,"not more than half"):
            verified_largest_component(reach_from(offsets,neighbors,6))
        with self.assertRaisesRegex(ValueError,"No surviving baseline"):
            screening_candidates(np.array([6]),reach,sums,counts,candidates=3)
        broken=counts.copy();broken[0]=1
        with self.assertRaisesRegex(ValueError,"target counts differ"):
            screening_candidates(sources,reach,sums,broken,candidates=3)
        with self.assertRaisesRegex(ValueError,"finite largest-component candidates"):
            screening_candidates(sources,reach,sums,counts,candidates=7)
        _,_,metadata=screening_candidates(sources,reach,sums,counts,candidates=3)
        result={**metadata,"candidates":3,"estimate_cut":1.,"top20_largest_estimate":1.,
                "most_central":[{"estimate":1.,"mean_distance":2.} for _ in range(3)]}
        record={"signature":{"graph":"fixture"},"result":result}
        self.assertTrue(reusable_center(record,record["signature"],3,9,candidates=3))
        legacy={**result};legacy.pop("selection_version")
        self.assertFalse(reusable_center({**record,"result":legacy},record["signature"],3,9,candidates=3))
        self.assertFalse(reusable_center(record,{"graph":"changed"},3,9,candidates=3))
        self.assertFalse(reusable_center({**record,"result":{**result,"component_size":3}},record["signature"],3,9,candidates=3))

    def test_overflow_and_empty_group_cells_are_rejected_with_counts_retained(self):
        hist=np.array([[0,3,2,0,0],[0,0,1,0,0]],np.int64)
        result=guarded_summary(hist,"fixture")
        self.assertEqual(result["pairs"],6)
        self.assertEqual(result["mean"],1.5)
        verify_cached_summary(result,"fixture",maxd=4)
        bad=hist.copy();bad[0,-1]=1
        with self.assertRaisesRegex(ValueError,"4-bin limit"):
            guarded_summary(bad,"fixture")
        with self.assertRaisesRegex(ValueError,"no reachable"):
            guarded_summary(np.zeros_like(hist),"fixture")
        with self.assertRaisesRegex(ValueError,"cached distances"):
            verify_cached_summary({**result,"pmf":{"4":1.}},"fixture",maxd=4)
        sums=np.array([[1,3],[2,6],[5,2]],np.float64)
        pairs=np.array([[1,2],[1,2],[2,1]],np.int64)
        groups=np.array([0,0,1],np.int64)
        result=group_summary(sums,pairs,groups,["A","B"])
        self.assertEqual(result["mean_distance"],[[1.5,2.25],[2.5,2.]])
        self.assertEqual(result["sources_per_group"],[2,1])
        self.assertEqual(result["reachable_pairs"],[[2,4],[2,1]])
        with self.assertRaisesRegex(ValueError,"No repaired-component sources"):
            group_summary(sums,pairs,np.array([0,0,0]),["A","B"])
        empty=pairs.copy();empty[2,1]=0
        with self.assertRaisesRegex(ValueError,"No reachable non-self target pairs"):
            group_summary(sums,empty,groups,["A","B"])


if __name__=="__main__":
    unittest.main()
