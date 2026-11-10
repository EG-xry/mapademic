"""B2b: the same seeded sample of the longest 1% of map links, without full edge-table copies.

Shared verified float32 tie features are built by ties_cache.py. The 99th percentile is exact;
selected edge row IDs retain original order, then the original RNG chooses 100,000 links.
"""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from common import CACHE, G, log, save_result
from ties_cache import BATCH, close_features, load_features, read_edge_rows, selected_values

SAMPLE = 100_000


def summarize_long_links(arrays, edges, scratch, sample_size=SAMPLE, batch_size=BATCH):
    length, weight, comm, country, x, y = (arrays[key] for key in ("length", "weights", "community", "country", "x", "y"))
    with selected_values(length,lambda a,b:~np.isnan(length[a:b]),scratch,batch_size) as values:
        p99=np.quantile(values,.99,overwrite_input=True)
    pieces=[]
    for start in range(0,len(length),batch_size):
        pieces.append(np.flatnonzero(length[start:start+batch_size]>=p99)+start)
    long_idx=np.concatenate(pieces)
    del pieces
    pick=np.sort(np.random.default_rng(20261025).choice(long_idx,sample_size,replace=False))
    # One streaming pass obtains both complete long-link endpoints and the sample, preserving each order.
    src,dst=read_edge_rows(edges,np.concatenate((long_idx,pick)),batch_size)
    long_src,long_dst=src[:len(long_idx)],dst[:len(long_idx)]
    src,dst=src[len(long_idx):],dst[len(long_idx):]
    left,right=comm[long_src],comm[long_dst]
    pairs=pd.DataFrame({"a":np.minimum(left,right),"b":np.maximum(left,right)}).value_counts(normalize=True).head(25)
    frame=pd.DataFrame({"x1":x[src],"y1":y[src],"x2":x[dst],"y2":y[dst],"weight":weight[pick],"length":length[pick],
                        "comm1":comm[src],"comm2":comm[dst],"same_community":comm[src]==comm[dst],
                        "same_country":np.where((country[src]>=0)&(country[dst]>=0),country[src]==country[dst],pd.NA)})
    result={"p99_length":float(p99),"long_links":int(len(long_idx)),
            "top_community_pairs":[[int(a),int(b),float(value)] for (a,b),value in pairs.items()]}
    return result,frame,pick


def main():
    arrays,manifest=load_features()
    result,frame,_=summarize_long_links(arrays,G / "edges_int32.parquet",CACHE / "ties/selection_scratch.npy")
    result["provenance"]={"feature_cache":str(CACHE / "ties/manifest.json"),
                          "feature_identity_sha256":manifest["fingerprint"],
                          "feature_code_sha256":manifest["identity"]["feature_code_sha256"]}
    close_features(arrays)
    target=CACHE / "b2_long_links.parquet"
    temporary=target.with_name(target.name+".tmp")
    pq.write_table(pa.Table.from_pandas(frame,preserve_index=False),temporary)
    temporary.replace(target)
    save_result("b2b_long_links",result)
    log(f"p99 length {result['p99_length']:.3f}; {result['long_links']:,} long links; sampled {len(frame):,}; "
        f"cross-country among known {1-frame.same_country.dropna().astype(bool).mean():.1%}")
    log(result["top_community_pairs"])


if __name__ == "__main__":
    main()
