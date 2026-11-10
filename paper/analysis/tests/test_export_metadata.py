"""Independent source-join and nullable-coordinate parity for the expanded local exporter, and its small-file refresh."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import export_deposit as deposit


class ExportMetadataTests(unittest.TestCase):
    def sources(self, base):
        ids = np.array([108,101,100,107,105,102], np.int64)
        url="https://openalex.org/"
        attrs_ids=[url+"A108","A101",url+"A100","A107",url+"A105",url+"A102"]
        attrs_path=base/"attrs.parquet"
        pq.write_table(pa.table({"node_idx":np.arange(len(ids),dtype=np.int32),"id":attrs_ids}),attrs_path)
        names = [deposit.FALLBACK_TOPIC, "Other", None]
        topics = [[{"display_name":names[0]}], [{"display_name":names[1]}], [],
                  [{"display_name":names[1]}, {"display_name":names[0]}], None,
                  [{"display_name":names[0]}], [{"display_name":names[0]}],
                  [{"display_name":names[0]}], [{"display_name":names[0]}],
                  [{"display_name":names[0]}], [{"display_name":names[0]}],
                  [{"display_name":names[0]}], [{"display_name":names[2]}]]
        authors = pa.table({"id":[url+"A100",url+"A100","A101",url+"A102",url+"A105","A107",url+"A777",None,
                                 url+"A108",url+"A101","A105","A0101",url+"A105"],
                            "topics":pa.array(topics,type=pa.list_(pa.struct([("display_name",pa.string())])))})
        author_paths = [base / "authors1.parquet", base / "authors2.parquet"]
        pq.write_table(authors.slice(0,5),author_paths[0],row_group_size=2)
        pq.write_table(authors.slice(5),author_paths[1],row_group_size=2)
        coords = pa.table({"id":["A107","A105","A777","A100","A102","https://openalex.org/A101"],
                           "xw":[np.nextafter(.123456789,1),None,9.,float("nan"),.76543210987,.3333333333333],
                           "yw":[.34,.5,9.,.6,None,.123456789]})
        coord_path=base/"coords.parquet"
        pq.write_table(coords,coord_path,row_group_size=2)
        return ids,attrs_ids,attrs_path,authors,author_paths,coords,coord_path

    def test_exact_first_topic_membership_and_nullable_double_coordinates(self):
        with tempfile.TemporaryDirectory(prefix="mapademic-export-metadata-") as directory:
            base=Path(directory)
            ids,attrs_ids,attrs_path,authors,paths,coords,coord_path=self.sources(base)
            result=deposit.reproducibility_arrays(ids,attrs_path,paths,coord_path,batch_size=2)
            # Independent Python joins match the original string fallback and numeric map-ID rules.
            excluded={r["id"] for r in authors.to_pylist() if r["topics"] and
                      r["topics"][0]["display_name"]==deposit.FALLBACK_TOPIC}
            by_id={int(r["id"].rsplit("A",1)[-1]):r for r in coords.to_pylist()}
            np.testing.assert_array_equal(result["is_fallback_topic"],[value in excluded for value in attrs_ids])
            # URL+URL and short+short match; both alias-mismatch directions remain false.
            np.testing.assert_array_equal(result["is_fallback_topic"],[True,False,True,True,False,False])
            np.testing.assert_array_equal(result["on_map"],[i in by_id for i in ids])
            for key in ("xw","yw"):
                expected=[by_id.get(i,{}).get(key) for i in ids]
                np.testing.assert_array_equal(result[key+"_valid"],[v is not None for v in expected])
                np.testing.assert_array_equal(result[key],np.array([np.nan if v is None else v for v in expected]))
                self.assertEqual(result[key].dtype,np.dtype("float64"))
            n=len(ids)
            attrs=pa.table({"node_idx":np.arange(n,dtype=np.int32),"id":attrs_ids,"name":[f"Name {i}" for i in ids],
                            "works_kept":np.full(n,5,np.int32),"works":np.full(n,6,np.int32),
                            "cited":np.zeros(n,np.int32),"degree":np.zeros(n,np.int32),
                            "community":np.zeros(n,np.int32),"country":["US"]*n,
                            "institution":["Example"]*n,"field":["Social Sciences"]*n,"on_map":result["on_map"]})
            attrs_path,target=base/"attrs.parquet",base/"researchers.parquet"
            pq.write_table(attrs,attrs_path,row_group_size=2)
            deposit.export_researchers(attrs_path,target,ids,np.arange(n),np.zeros(n,np.uint8),np.array([],np.int64),
                                       network="min5",batch_size=2,supplementary=result)
            exported=pq.read_table(target)
            self.assertEqual(exported.schema.field("is_fallback_topic").type,pa.bool_())
            self.assertEqual(exported.column("is_fallback_topic").null_count,0)
            self.assertEqual(exported.column("field").to_pylist(),["Social Sciences"]*n)
            for key in ("xw","yw"):
                self.assertEqual(exported.schema.field(key).type,pa.float64())
                values=exported.column(key).to_pylist()
                for node,author_id in enumerate(ids):
                    expected=by_id.get(author_id,{}).get(key)
                    if expected is None:self.assertIsNone(values[node])
                    elif np.isnan(expected):self.assertTrue(np.isnan(values[node]))
                    else:self.assertEqual(values[node],expected)
            # Preserve B2 float32 rounding before subtraction, including finite and missing endpoints.
            x=exported.column("xw").to_numpy().astype(np.float32)
            y=exported.column("yw").to_numpy().astype(np.float32)
            source_x=np.array([by_id.get(i,{}).get("xw",np.nan) for i in ids],np.float32)
            source_y=np.array([by_id.get(i,{}).get("yw",np.nan) for i in ids],np.float32)
            left=np.array([0,1,2,3,4,5]);right=np.array([3,3,3,1,5,4])
            distances=np.hypot(x[left]-x[right],y[left]-y[right])
            self.assertEqual(int(np.isfinite(distances).sum()),2)
            np.testing.assert_array_equal(distances,np.hypot(source_x[left]-source_x[right],source_y[left]-source_y[right]))
            # min15 schema requires no supplementary reads and remains unchanged.
            core=base/"core.parquet"
            deposit.export_researchers(attrs_path,core,ids,np.arange(n),np.zeros(n,np.uint8),np.array([],np.int64),
                                       network="min15",batch_size=2)
            self.assertTrue({"is_fallback_topic","xw","yw","on_map"}.isdisjoint(pq.read_schema(core).names))
            before=target.read_bytes()
            with self.assertRaisesRegex(AssertionError,"map membership"):
                deposit.export_researchers(attrs_path,target,ids,np.arange(n),np.zeros(n,np.uint8),np.array([],np.int64),
                                           network="min5",batch_size=2,
                                           supplementary={**result,"on_map":~result["on_map"]})
            self.assertEqual(target.read_bytes(),before)

    def test_duplicates_and_missing_sources_fail(self):
        with tempfile.TemporaryDirectory(prefix="mapademic-export-metadata-invalid-") as directory:
            base=Path(directory)
            ids,_,attrs_path,_,paths,coords,coord_path=self.sources(base)
            with self.assertRaisesRegex(FileNotFoundError,"authors_extract"):
                deposit.reproducibility_arrays(ids,attrs_path,[],coord_path,batch_size=2)
            pq.write_table(pa.concat_tables([coords,coords.slice(0,1)]),coord_path,row_group_size=2)
            with self.assertRaisesRegex(AssertionError,"duplicate map"):
                deposit.reproducibility_arrays(ids,attrs_path,paths,coord_path,batch_size=2)
            with self.assertRaisesRegex(AssertionError,"duplicate researcher"):
                deposit.reproducibility_arrays(np.array([100,100]),attrs_path,paths,coord_path,batch_size=2)

    def test_small_file_refresh_renames_and_leaves_large_files(self):
        with tempfile.TemporaryDirectory(prefix="mapademic-export-small-") as directory:
            base = Path(directory)
            out, cache = base / "deposit", base / "cache"
            out.mkdir()
            cache.mkdir()
            for name in deposit.PACKAGE_FILES:
                (out / name).write_bytes(name.encode() * 3)
            deposit.write_checksums(out)
            large = [name for name in deposit.PACKAGE_FILES if name not in deposit.SMALL_FILES]
            before = {name: ((out / name).stat().st_mtime_ns, (out / name).read_bytes()) for name in large}
            sums = deposit.checksum_entries(out)
            (cache / "community_labels.json").write_text(json.dumps({"0": "Medicine", "1": "Mixed CJK script"}))
            pd.DataFrame({"community": [0, 1], "size": [5000, 1200], "cohesion": [0.81, 0.97],
                          "top_country": ["US", "JP"], "top_country_share": [0.6, 0.56],
                          "top_fields": ["Medicine 58%", "Social Sciences 93%"], "median_degree": [35.0, 8.0],
                          "median_cited": [255.0, 0.0]}).to_csv(cache / "a5b_communities.csv", index=False)
            readme = base / "public_README.md"
            readme.write_text("# Public description\n")
            deposit.refresh_small_files(out, cache=cache, readme=readme)
            labels = pd.read_csv(out / "communities.csv").set_index("community").label.to_dict()
            self.assertEqual(labels, {0: "Medicine", 1: "Japanese, surname first"})
            self.assertEqual((out / "README.md").read_text(), "# Public description\n")
            self.assertEqual({name: ((out / name).stat().st_mtime_ns, (out / name).read_bytes()) for name in large}, before)
            refreshed = deposit.checksum_entries(out)
            self.assertEqual({name: refreshed[name] for name in large}, {name: sums[name] for name in large})
            self.assertTrue(all(refreshed[name] != sums[name] for name in deposit.SMALL_FILES))
            (out / "links.parquet").write_bytes(b"changed behind the export")
            with self.assertRaisesRegex(RuntimeError, "links.parquet"):
                deposit.refresh_small_files(out, cache=cache, readme=readme)


if __name__=="__main__":
    unittest.main()
