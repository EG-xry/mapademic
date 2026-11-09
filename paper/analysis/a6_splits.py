"""A6e: profile splits -- one person spread over several OpenAlex profiles.

Profiles with the same normalized name (lowercase, accents and punctuation
dropped) that share coauthors are probably one person: different people who
share a name rarely share collaborators. For every pair within a same-name
group (groups up to MAXG profiles; bigger groups are the most common names and
mostly different people) we count shared coauthors and call the pair a likely
split at SHARED or more. Writes cache/split_pairs.parquet.
"""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numba import njit, prange

from common import CACHE, con, load_csr, log, save_result

MAXG, SHARED = 200, 3


@njit(parallel=True, cache=True)
def sort_rows(offsets, neighbors):
    out = neighbors.copy()
    for u in prange(offsets.shape[0] - 1):
        out[offsets[u]:offsets[u + 1]].sort()
    return out


@njit(parallel=True, cache=True)
def shared(offsets, sn, a, b):
    out = np.zeros(a.shape[0], np.int32)
    for k in prange(a.shape[0]):
        i, ie = offsets[a[k]], offsets[a[k] + 1]
        j, je = offsets[b[k]], offsets[b[k] + 1]
        c = 0
        while i < ie and j < je:
            x, y = sn[i], sn[j]
            if x == y:
                c += 1
                i += 1
                j += 1
            elif x < y:
                i += 1
            else:
                j += 1
        out[k] = c
    return out


c = con()
A = f"read_parquet('{CACHE / 'node_attrs.parquet'}')"
log("same-name groups")
pairs = c.execute(f"""
    WITH n AS (SELECT node_idx, trim(regexp_replace(lower(strip_accents(name)), '[^a-z0-9\\p{{L}} ]', ' ', 'g')) AS key
               FROM {A} WHERE degree > 0),
         g AS (SELECT key, count(*) AS size FROM n GROUP BY 1 HAVING count(*) BETWEEN 2 AND {MAXG})
    SELECT a.node_idx AS a, b.node_idx AS b, g.size
    FROM n a JOIN g USING (key) JOIN n b ON b.key = a.key AND b.node_idx > a.node_idx""").fetchnumpy()
log(f"{len(pairs['a']):,} same-name pairs")
c.close()  # release the name-group join before allocating the sorted graph copy
offsets, neighbors = load_csr()
sn = sort_rows(offsets, neighbors)
del neighbors
k = shared(offsets, sn, pairs["a"].astype(np.int64), pairs["b"].astype(np.int64))
tab = pa.table({"a": pairs["a"], "b": pairs["b"], "group_size": pairs["size"], "shared": k})
pq.write_table(tab, CACHE / "split_pairs.parquet")
likely = k >= SHARED
nodes = np.unique(np.concatenate([pairs["a"][likely], pairs["b"][likely]]))
out = {"same_name_pairs": int(len(k)), "pairs_sharing_1plus": int((k >= 1).sum()),
       "likely_split_pairs": int(likely.sum()), "profiles_in_likely_splits": int(len(nodes)),
       "share_of_mapped_profiles": float(len(nodes) / len(offsets[:-1])),
       "shared_hist": {int(t): int((k == t).sum()) for t in range(0, 11)}}
save_result("a6_splits", out)
log(out)
del sn, offsets
c = con()
c.register("t", tab)
print(c.execute(f"""SELECT x.name, x.country AS ca, y.country AS cb, x.degree AS da, y.degree AS db, t.shared
                    FROM t JOIN {A} x ON x.node_idx = t.a JOIN {A} y ON y.node_idx = t.b
                    WHERE t.shared >= {SHARED} ORDER BY t.shared DESC LIMIT 25""").fetchdf().to_string())
