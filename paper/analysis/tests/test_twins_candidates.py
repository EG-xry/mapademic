"""Exact candidate/decoy parity for the bounded script-twin index, using no production data."""
import ast
from array import array
from collections import defaultdict
from pathlib import Path
import random
import unittest

import numpy as np


def helpers():
    # a6_twins.py is a CLI script; extracting pure helpers avoids starting its full-data workflow.
    path = Path(__file__).resolve().parents[1] / "a6_twins.py"
    names = {"candidate_indexes", "usable_candidates", "decoys"}
    selected = [node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name in names]
    env = {"MAXC":5, "defaultdict":defaultdict, "array":array, "log":lambda *args:None}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), env)
    return env


class TwinCandidateParity(unittest.TestCase):
    def test_overlapping_keys_overflow_self_exclusion_and_decoy_rng(self):
        env = helpers()
        rng = random.Random(20261103)
        latin = np.arange(2000, dtype=np.int64)
        names = [str(i) for i in range(2200)]
        keyspace = [(chr(97+i%6),f"surname-{i//6}") for i in range(180)]
        profile_keys = {}
        for i in latin:
            profile_keys[int(i)] = set(rng.sample(keyspace,rng.randrange(1,5)))
        # Rare overlapping keys supply usable unions; common ones exercise early rejection.
        rare = [("rare",f"r-{i}") for i in range(12)]
        for at,key in enumerate(rare):
            for node in range(1500+at,1500+at+at%6+1):
                profile_keys[node].add(key)
        focal = {2000+i:set(rng.sample(keyspace+rare,rng.randrange(1,5))) for i in range(80)}
        focal.update({2100+i:{key} for i,key in enumerate(rare)})
        # Self-only, six total including self (eligible five), and seven including self (ineligible).
        self_only, six_with_self, seven_with_self = ("self","one"),("self","six"),("self","seven")
        profile_keys[0].add(self_only)
        for node in range(6): profile_keys[node].add(six_with_self)
        for node in range(7): profile_keys[node].add(seven_with_self)
        focal[0]={self_only}
        focal[1]={six_with_self}
        focal[2]={seven_with_self}
        needed=set().union(*focal.values())
        needed_s={key[1] for key in needed}
        old_index,old_surnames=defaultdict(list),defaultdict(list)
        for node in latin.tolist():
            ks=profile_keys[node]
            for key in ks&needed: old_index[key].append(node)
            for surname in {key[1] for key in ks}&needed_s: old_surnames[surname].append(node)
        original={f:set().union(*(old_index.get(key,()) for key in ks))-{f} for f,ks in focal.items()}
        old_any=sum(bool(values) for values in original.values())
        old_usable={f:values for f,values in original.items() if 0<len(values)<=5}
        index,surnames,overflow=env["candidate_indexes"](latin,names,lambda name:profile_keys[int(name)],needed,needed_s)
        new_any,new_usable=env["usable_candidates"](focal,index,overflow)
        self.assertEqual(old_any,new_any)
        self.assertEqual(old_usable,new_usable)
        self.assertEqual(list(old_usable),list(new_usable))
        self.assertNotIn(0,new_usable)
        self.assertIn(1,new_usable)
        self.assertNotIn(2,new_usable)
        self.assertTrue(overflow)
        self.assertTrue(all(len(bucket)<=6 for bucket in index.values()))
        self.assertEqual(set(index),set(old_index))
        for key in index:
            self.assertEqual(index[key],old_index[key][:6])
            self.assertEqual(key in overflow,len(old_index[key])>6)
        for surname,pool in surnames.items():
            self.assertIsInstance(pool,array)
            self.assertEqual(pool.typecode,"i")
            self.assertEqual(list(pool),old_surnames[surname])
        for node in new_usable:
            self.assertEqual(list(new_usable[node]),list(old_usable[node]),"candidate tie order changed")
        # Repeat the post-index pruning used by the CLI. Decoy pools are complete and identically ordered.
        usable_keys={node:focal[node] for node in new_usable}
        kept_keys={key for ks in usable_keys.values() for key in ks if key in index}
        kept_surnames={key[1] for key in kept_keys}
        pruned_surnames={name:pool for name,pool in surnames.items() if name in kept_surnames}
        outputs=[]
        for candidate_index,surname_index,usable in ((old_index,old_surnames,old_usable),(kept_keys,pruned_surnames,new_usable)):
            env.update(index=candidate_index,sindex=surname_index,rng=np.random.default_rng(20261024),
                       name_keys=lambda node:frozenset(profile_keys[node]))
            outputs.append({node:env["decoys"](node,candidates,focal[node]) for node,candidates in usable.items()})
        self.assertEqual(outputs[0],outputs[1],"decoy draws or filtering changed")


if __name__ == "__main__":
    unittest.main()
