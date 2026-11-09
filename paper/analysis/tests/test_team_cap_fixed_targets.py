"""Fixed-target team-cap effects, including a larger cap that only adds reachable targets."""
from collections import deque
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import a2c_team_cap as team


def fixture():
    n=6
    members=[[0,1],[1,2],[2,3],[0,2],[3,4],[4,5]]
    sizes=np.array([2,2,75,100,500,501],np.int32)
    off=np.arange(0,2*len(members)+1,2,dtype=np.int64)
    authors=np.array([x for pair in members for x in pair],np.int32)
    a_off=np.zeros(n+1,np.int64)
    a_nb=np.zeros(len(authors),np.int32)
    team._author_side_into(n,off,authors,a_off,a_nb)
    return {"a_off":a_off,"a_nb":a_nb,"w_off":off,"w_auth":authors,"w_size":sizes},members


def reference(members,sizes,sources,cap,mask):
    adjacency=[set() for _ in mask]
    for authors,size in zip(members,sizes):
        if size>cap: continue
        for author in authors: adjacency[author].update(set(authors)-{author})
    full=np.zeros((len(sources),team.MAXD+1),np.int64)
    fixed=np.zeros_like(full)
    for row,source in enumerate(sources):
        distances,queue={int(source):0},deque([int(source)])
        while queue:
            node=queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor not in distances:
                    distances[neighbor]=distances[node]+1
                    queue.append(neighbor)
        for target,distance in distances.items():
            full[row,distance]+=1
            if mask[target]: fixed[row,distance]+=1
    return full,fixed


class FixedTargetTests(unittest.TestCase):
    def test_newly_reached_targets_are_separate_from_shortening(self):
        graph,members=fixture()
        sources=np.array([0,1,2],np.int32)
        mask=np.array([True,True,True,False,False,False])
        base=previous=None
        for cap in (50,75,100,500,1_000_000):
            full,fixed=team.run_with_targets(graph['a_off'],graph['a_nb'],graph['w_off'],graph['w_auth'],graph['w_size'],sources,cap,team.MAXD,mask)
            expected=reference(members,graph['w_size'],sources,cap,mask)
            np.testing.assert_array_equal(full,expected[0])
            np.testing.assert_array_equal(fixed,expected[1])
            if cap==50:
                np.testing.assert_array_equal(full,fixed)
                base=previous=fixed
            result=team.fixed_target_summary(fixed,base,previous,int(mask.sum()))
            self.assertEqual(result['target_pairs_per_source'],2)
            self.assertEqual(result['pairs'],6)
            if cap==75:
                self.assertGreater(team.summarize(full)['mean'],team.summarize(base)['mean'])
                np.testing.assert_array_equal(fixed,base)
                self.assertEqual(result['paired_cap_minus_50']['mean'],0.)
                self.assertEqual(result['paired_cap_minus_50']['se'],0.)
            if cap>=100:
                self.assertAlmostEqual(result['mean'],1.)
                self.assertAlmostEqual(result['paired_cap_minus_50']['mean'],-1/3)
                self.assertAlmostEqual(result['paired_cap_minus_50']['se'],1/6)
            previous=fixed
        malformed=base.copy()
        malformed[0,1]-=1
        with self.assertRaisesRegex(AssertionError,'population changed'):
            team.fixed_target_summary(malformed,base,base,3)
        worsening=base.copy()
        worsening[0,1]-=1; worsening[0,3]+=1
        with self.assertRaisesRegex(AssertionError,'CDF worsened'):
            team.fixed_target_summary(worsening,base,base,3)

    def test_checkpoint_stores_both_histograms_and_rejects_changed_targets(self):
        graph,members=fixture()
        sources=np.array([0,1,2],np.int32)
        mask=np.array([True,True,True,False,False,False])
        with tempfile.TemporaryDirectory(prefix='mapademic-fixed-target-test-') as directory:
            cache=Path(directory)
            with mock.patch.object(team,'CACHE',cache),mock.patch.dict(team.os.environ,{'A2C_SOURCE_BATCH':'1'}):
                # The old signature has no fixed target definition and must not be accepted.
                old=team._digest({'graph':'fixture','sources':sources.tolist(),'cap':75,'maxd':team.MAXD,'kernel_version':1})
                path=cache/'a2c_hist_cap75.npz'
                np.savez(path,signature=np.array(old),sources=sources,hist=np.zeros((3,65),np.int64))
                partial=team.checkpointed_run(graph,sources[:2],75,'fixture',sources,mask)
                full=team.checkpointed_run(graph,sources,75,'fixture',sources,mask)
                expected=reference(members,graph['w_size'],sources,75,mask)
                for actual,want in zip(full,expected):np.testing.assert_array_equal(actual,want)
                for actual,want in zip(partial,expected):np.testing.assert_array_equal(actual,want[:2])
                with np.load(path,allow_pickle=False) as saved:
                    self.assertEqual(int(saved['kernel_version']),2)
                    self.assertEqual(str(saved['target_mask_sha256']),hashlib.sha256(mask).hexdigest())
                    np.testing.assert_array_equal(saved['hist_fixed'],expected[1])
                with mock.patch.object(team,'run_with_targets',side_effect=AssertionError('cached BFS repeated')):
                    team.checkpointed_run(graph,sources,75,'fixture',sources,mask)
                expanded=mask.copy();expanded[3]=True
                changed=team.checkpointed_run(graph,sources,75,'fixture',sources,expanded)
                np.testing.assert_array_equal(changed[0],changed[1])
                self.assertTrue(np.all(changed[1].sum(axis=1)==4))


if __name__=='__main__':unittest.main()
