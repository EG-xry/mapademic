"""Exact full-array parity for B2/B2b disk-backed summaries and metadata alignment."""
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import b2_ties as ties
import b2b_long_links as long_links
import identities
import ties_cache as cache


def reference_ties(src,dst,weight,comm,cc,fd,x,y,sample_size):
    in_comm=comm[src]==comm[dst]
    both_cc=(cc[src]>=0)&(cc[dst]>=0)
    in_cc=both_cc&(cc[src]==cc[dst])
    both_fd=(fd[src]>=0)&(fd[dst]>=0)
    in_fd=both_fd&(fd[src]==fd[dst])
    length=np.hypot(x[src]-x[dst],y[src]-y[dst])
    on_map=np.isfinite(length)
    out={'links':int(len(weight)),'links_on_map':int(on_map.sum()),'weight_share_on_map':float(weight[on_map].sum()/weight.sum()),
         'weight_quantiles':{str(q):float(np.quantile(weight,q)) for q in (.1,.25,.5,.75,.9,.99)}}
    def describe(mask):
        return {'links':int(mask.sum()),'share_links':float(mask.mean()),'share_weight':float(weight[mask].sum()/weight.sum()),
                'median_weight':float(np.median(weight[mask])),'mean_weight':float(weight[mask].mean()),
                'share_weight_ge_1':float((weight[mask]>=1).mean())}
    for key,mask in (('inside_community',in_comm),('across_communities',~in_comm),('same_country',in_cc),
                     ('across_countries',both_cc&~in_cc),('same_field',in_fd),('across_fields',both_fd&~in_fd)):
        out[key]=describe(mask)
    dec=np.quantile(weight,np.linspace(0,1,11))
    bins=np.clip(np.searchsorted(dec,weight,side='right')-1,0,9)
    out['by_weight_decile']=[{'decile':k,'weight_from':float(dec[k]),'weight_to':float(dec[k+1]),
                             'across_communities':float((~in_comm[bins==k]).mean()),
                             'across_countries':float((both_cc&~in_cc)[bins==k].sum()/max(both_cc[bins==k].sum(),1)),
                             'median_length':float(np.nanmedian(length[bins==k]))} for k in range(10)]
    lq=np.quantile(length[on_map],[.5,.9,.99,.999])
    out['length_quantiles']=dict(zip(('p50','p90','p99','p999'),map(float,lq)))
    wm=weight[on_map].sum()
    for label,mask in (('shortest_half',length<=lq[0]),('longest_10pct',length>=lq[1]),('longest_1pct',length>=lq[2])):
        out[label]={**describe(mask),'share_map_links':float(mask.sum()/on_map.sum()),'share_map_weight':float(weight[mask].sum()/wm),
                    'across_communities':float((~in_comm[mask]).mean()),
                    'across_countries':float((both_cc&~in_cc)[mask].sum()/max(both_cc[mask].sum(),1)),
                    'across_fields':float((both_fd&~in_fd)[mask].sum()/max(both_fd[mask].sum(),1))}
    pick=np.random.default_rng(20261025).choice(len(weight),sample_size,replace=False)
    sample={'rows':pick,'decile':bins[pick].astype(np.int8),'in_community':in_comm[pick],
            'in_country':in_cc[pick],'across_country':both_cc[pick]&~in_cc[pick]}
    return out,sample,length


class TieStreamingParity(unittest.TestCase):
    def assert_nested_equal(self,left,right,path='root'):
        if isinstance(left,dict):
            self.assertEqual(set(left),set(right),path)
            for key in left:self.assert_nested_equal(left[key],right[key],path+'.'+key)
        elif isinstance(left,list):
            self.assertEqual(len(left),len(right),path)
            for at,(a,b) in enumerate(zip(left,right)):self.assert_nested_equal(a,b,path+f'[{at}]')
        elif isinstance(left,float) and math.isnan(left):
            self.assertTrue(math.isnan(right),path)
        else:self.assertEqual(left,right,path)

    def test_full_array_parity_exact_quantiles_samples_and_cache(self):
        rng=np.random.default_rng(127)
        n=180
        ids=rng.permutation(np.arange(10000,10000+n,dtype=np.int64))
        pairs=np.array([(a,b) for a in range(n) for b in range(a+1,n)],np.int32)
        pairs=pairs[rng.choice(len(pairs),5003,replace=False)]
        src,dst=pairs.T
        weight=rng.choice(np.array([.02,.05,.1,.1234567,.2,.5,1.,1.5,2.],np.float32),len(pairs))
        countries=rng.choice(np.array(['US','CN','JP',None],object),n)
        fields=rng.choice(np.array(['Physics','Medicine',None],object),n)
        communities=[int(v) for v in rng.integers(0,8,n)]
        communities[-5]=None  # require pandas-compatible NaN equality and nullable long-link output
        attrs=pa.table({'node_idx':pa.array(np.arange(n),type=pa.int32()),'id':[f'https://openalex.org/A{x}' for x in ids],
                        'community':pa.array(communities,type=pa.int32()),'country':pa.array(countries,type=pa.string()),
                        'field':pa.array(fields,type=pa.string())})
        shown=rng.permutation(np.arange(n))[:150]
        coords=pd.DataFrame({'id':[f'https://openalex.org/A{ids[i]}' for i in shown],
                             'xw':rng.normal(size=len(shown)),'yw':rng.normal(size=len(shown))})
        coords.loc[len(coords)]=['https://openalex.org/A999999',1.,2.]  # ignored, as in the original reindex
        a=attrs.to_pandas()
        xy=coords.set_index('id').reindex(a.id)
        x,y=xy.xw.to_numpy(np.float32),xy.yw.to_numpy(np.float32)
        comm=a.community.to_numpy()
        cc=a.country.astype('category').cat.codes.to_numpy()
        fd=a.field.astype('category').cat.codes.to_numpy()
        with tempfile.TemporaryDirectory(prefix='mapademic-ties-test-') as directory:
            base=Path(directory)
            attrs_path,coords_path,edges_path,ids_path=base/'attrs.parquet',base/'coords.parquet',base/'edges.parquet',base/'ids.npy'
            pq.write_table(attrs,attrs_path,row_group_size=17)
            coords.to_parquet(coords_path,index=False,row_group_size=19)
            pq.write_table(pa.table({'src':src,'dst':dst,'weight':weight}),edges_path,row_group_size=101)
            np.save(ids_path,ids)
            arrays,manifest=cache.load_features(attrs_path,coords_path,edges_path,ids_path,base/'cache',batch_size=37)
            self.assertEqual(len(manifest['identity']['feature_code_sha256']),64)
            np.testing.assert_array_equal(arrays['x'],x)
            np.testing.assert_array_equal(arrays['y'],y)
            np.testing.assert_array_equal(arrays['community'],comm)
            np.testing.assert_array_equal(arrays['weights'],weight)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                expected,sample,length=reference_ties(src,dst,weight,comm,cc,fd,x,y,123)
                actual,new_sample=ties.summarize_ties(arrays,base/'scratch.npy',sample_size=123,batch_size=43)
            self.assert_nested_equal(actual,expected)
            np.testing.assert_array_equal(arrays['length'],length)
            for key in sample:np.testing.assert_array_equal(new_sample[key],sample[key])
            selected_s,selected_d=cache.read_edge_rows(edges_path,sample['rows'],batch_size=59)
            np.testing.assert_array_equal(selected_s,src[sample['rows']])
            np.testing.assert_array_equal(selected_d,dst[sample['rows']])
            long_result,frame,pick=long_links.summarize_long_links(arrays,edges_path,base/'scratch.npy',sample_size=12,batch_size=71)
            p99=np.nanquantile(length,.99)
            long_idx=np.flatnonzero(length>=p99)
            old_pick=np.sort(np.random.default_rng(20261025).choice(long_idx,12,replace=False))
            np.testing.assert_array_equal(pick,old_pick)
            ls,ld=comm[src[long_idx]],comm[dst[long_idx]]
            ranked=pd.DataFrame({'a':np.minimum(ls,ld),'b':np.maximum(ls,ld)}).value_counts(normalize=True).head(25)
            expected_result={'p99_length':float(p99),'long_links':len(long_idx),
                             'top_community_pairs':[[int(a),int(b),float(value)] for (a,b),value in ranked.items()]}
            self.assert_nested_equal(long_result,expected_result)
            ss,dd=src[old_pick],dst[old_pick]
            expected_frame=pd.DataFrame({'x1':x[ss],'y1':y[ss],'x2':x[dd],'y2':y[dd],'weight':weight[old_pick],'length':length[old_pick],
                                          'comm1':comm[ss],'comm2':comm[dd],'same_community':comm[ss]==comm[dd],
                                          'same_country':np.where((cc[ss]>=0)&(cc[dd]>=0),cc[ss]==cc[dd],pd.NA)})
            pd.testing.assert_frame_equal(frame,expected_frame)
            cache.close_features(arrays)
            with mock.patch.object(cache,'metadata_arrays',side_effect=AssertionError('valid cache recomputed')):
                reused,reused_manifest=cache.load_features(attrs_path,coords_path,edges_path,ids_path,base/'cache',batch_size=37)
            self.assertEqual(reused_manifest,manifest)
            cache.close_features(reused)

    def test_feature_cache_survives_moving_the_project_folder(self):
        n=12
        ids=np.arange(100,100+n,dtype=np.int64)
        attrs=pa.table({'node_idx':pa.array(np.arange(n),type=pa.int32()),'id':[f'https://openalex.org/A{x}' for x in ids],
                        'community':pa.array([i%3 for i in range(n)],type=pa.int32()),
                        'country':pa.array(['US','CN',None]*4,type=pa.string()),
                        'field':pa.array(['Physics',None,'Medicine']*4,type=pa.string())})
        coords=pd.DataFrame({'id':[f'https://openalex.org/A{x}' for x in ids[:8]],'xw':np.arange(8.),'yw':np.arange(8.)})
        edges=pa.table({'src':np.array([0,1,2],np.int32),'dst':np.array([1,2,3],np.int32),
                        'weight':np.array([1.,.5,.25],np.float32)})
        names=('attrs.parquet','coords.parquet','edges.parquet','ids.npy')
        with tempfile.TemporaryDirectory() as directory:
            before,after=Path(directory)/'before',Path(directory)/'after'
            before.mkdir()
            pq.write_table(attrs,before/names[0])
            coords.to_parquet(before/names[1],index=False)
            pq.write_table(edges,before/names[2])
            np.save(before/names[3],ids)
            arrays,manifest=cache.load_features(*(before/name for name in names),before/'cache')
            cache.close_features(arrays)
            os.rename(before,after)
            with mock.patch.object(identities,'ROOT',after), \
                    mock.patch.object(cache,'metadata_arrays',side_effect=AssertionError('valid cache recomputed')):
                reused,reused_manifest=cache.load_features(*(after/name for name in names),after/'cache')
            self.assertEqual(reused_manifest,manifest)
            cache.close_features(reused)

    def test_disk_backed_sort_and_overlap_match_independent_neighbor_sets(self):
        adjacency=[[3,2,1],[2,0],[3,1,0],[2,0],[]]
        offsets=np.zeros(len(adjacency)+1,np.int64)
        offsets[1:]=np.cumsum([len(row) for row in adjacency])
        neighbors=np.array([v for row in adjacency for v in row],np.int32)
        with tempfile.TemporaryDirectory(prefix='mapademic-tie-sort-test-') as directory:
            sn=np.lib.format.open_memmap(Path(directory)/'neighbors.npy',mode='w+',dtype=np.int32,shape=neighbors.shape)
            sn[:]=neighbors
            ties.sort_rows_inplace(offsets,sn)
            np.testing.assert_array_equal(sn,ties.sort_rows(offsets,neighbors))
            left,right=np.array([0,0,1,2],np.int64),np.array([1,3,2,3],np.int64)
            actual=ties.overlap(offsets,sn,left,right)
            expected=[]
            for a,b in zip(left,right):
                common=len(set(adjacency[a])&set(adjacency[b]))
                denominator=len(adjacency[a])+len(adjacency[b])-2-common
                expected.append(common/denominator if denominator else 0.)
            np.testing.assert_array_equal(actual,expected)
            sn._mmap.close()


if __name__=='__main__':unittest.main()
