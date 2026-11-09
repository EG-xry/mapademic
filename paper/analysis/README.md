# Analyses for the five-work paper

These scripts reproduce *The shape of science* from the OpenAlex snapshot dated 2026-09-23. The current paper uses researchers with at least five quality-filtered scholarly works: **19,848,920 author records and 525,109,095 undirected links**. This is a snapshot-specific author-profile network, not a census of all researchers.

This directory is released with the Mapademic repository (MIT licence; archived on Zenodo under the concept DOI 10.5281/zenodo.23629696). The network it analyses is deposited separately under CC0 1.0 at 10.5281/zenodo.23681522. The snapshot extracts, caches, results, run records and manuscript files that the scripts read or write under `data/` and `paper/` are local and are not in the repository.

Eligibility uses `works_kept`, rather than OpenAlex's unfiltered `works_count`. The shared filter in `pipeline/works.py` excludes expansion-pack records, paratext and listed nonscholarly/versioned types. Eligibility counts qualifying works regardless of team size; edge construction additionally excludes works with more than 50 authors. Each shared work with `n` authors contributes `1/(n-1)` to a link's weight. The team-size analysis retains the same eligible researchers while changing the cap on connecting works.

## Networks and preserved outputs

Set `NET` explicitly. `common.py` defaults to `min5`.

| Setting | Network | Intermediates | Results |
|---|---|---|---|
| `NET=min5` | `data/2026-09-23/min5/` | `paper/cache/` | `paper/results/` |
| `NET=min15` | `data/2026-09-23/` | `paper/cache_min15/` | `paper/results_min15/` |

The 15-work core has 8,396,458 records and 366,027,181 links. Preserve its saved results, caches, `paper/figures_min15/` and `paper/deposit_README_min15.md` for comparison. `FIGURES` currently points to `paper/figures/` for either setting, so do not regenerate core figures over the current paper's figures.

The public map at mapademic.app retains its 15-work layout. Paper map panels draw that subset at existing coordinates, recoloured with communities estimated on the full five-work network. Map lengths and longest-link results use only edges with coordinates at both endpoints; they do not measure all 525 million links in space.

The five-work deposit adds `is_fallback_topic`, `xw` and `yw` to the researcher rows. Export obtains the indicator from the exact first-topic name test in `authors_extract/*/*.parquet`, using the same exact `node_attrs.id = authors_extract.id` string membership as A9. These snapshot IDs are full OpenAlex URLs; short and URL aliases must not be normalized when assigning the indicator. A DuckDB semi-join capped at 4 GB and two threads streams matching node indices, and closes before coordinate arrays are allocated. Original float64 coordinates from `coords_web.parquet` retain attrs.py's numeric-ID join in source batches of at most 250,000 rows. Original node order is preserved. `on_map` retains coordinate-file membership even if a source coordinate is null; source nulls and NaNs remain distinguishable. The 15-work export schema and saved package are unchanged.

These columns provide the numerical inputs for field exclusions and spatial tie statistics without requiring those two local source files. Field-distance analyses exclude fallback-topic records; B2's across-field statistics retain the original field assignments. B2/B2b convert coordinates to float32 before subtraction and `numpy.hypot`, use finite lengths for spatial summaries, preserve link row order and draw samples with seed `20261025`. Reproduction still requires the analysis code and experiment-specific source identities/parameters; it should not substitute a new random draw for a reported one.

Exact map rendering additionally uses `paper/cache/map/coords_web.parquet` (including ring/rendering flags), `tiles/`, `regions.json`, plotting code and its recorded environment. The team-cap experiment requires `data/2026-09-23/works_authorships/*/*.parquet` and the shared work filter; original work memberships and excluded large-team links cannot be reconstructed from aggregated links. These supplementary local inputs are not in the network package or the repository. The deposit's README (`paper/deposit_README.md` locally) gives the package's reproduction scope.

## Environment and execution

The paper environment is `.tmp/venv-paper/`. Record its actual package versions in each run: the repository's basic pipeline dependencies do not cover all analyses. The environment includes NumPy, pandas, PyArrow, Numba, SciPy, DuckDB, matplotlib, NetworkX, Pillow and transliteration packages. Japanese matching also needs the cached JMnedict readings. Manuscript builds use `paper/build.py`, which is not in the repository, with `pandoc`, LuaLaTeX and the fonts it specifies; the tests that need it are skipped when it is absent.

Run individual scripts from `paper/analysis/`, for example:

```sh
cd paper/analysis
NET=min5 ../../.tmp/venv-paper/bin/python a9c_field_order.py
```

The main run is managed by `paper/analysis/run_min5.py`, with progress, events and status under `paper/runs/min5-2026-11-03/`. Three later rounds reuse its design with their own state and logs: `run_followup.py` (`paper/runs/followup-2026-11-03/`), `run_comments.py` (`paper/runs/comments-2026-11-08/`) and `run_review.py` (`paper/runs/review-2026-11-08/`). Each runner's step list names the scripts it runs. Analysis logs and result JSON files remain in `paper/results/`. Read the runner's `--help` and run status before launching another process. Run one heavy analysis at a time: traversal, contraction and large joins need tens of gigabytes of RAM. Record failures and retries; a completion marker alone does not establish that an output exists and is valid.

The old `.tmp/paper-rerun/chain_*.sh` files are historical orchestration. Their Chain B can mark itself complete after a subprocess failure, and Chain D omits the final deposit-dependent checks. Prefer the current runner and validate its products.

After the main runner has finished every step and the final manuscript build is ready, run the full validation from the repository root with its existing memory watchdog:

```sh
.tmp/venv-paper/bin/python paper/analysis/run_final_validation.py
```

This launches exactly `validate_min5.py --require-manuscript`; there is no quick or reconstruction-skipping mode. It holds the same `runner.lock` for the entire invocation, refuses overlap without truncating the lock, and reuses the runner's owned-process footprint/free-memory/swap watchdog. `--min-free` controls the launch threshold (default 35%); runtime limits remain those in `run_resources.py`. Failure or interruption drains only the process group launched by this invocation.

Each attempt records its command and source hashes, stdout/stderr log, watchdog events, resource receipt and a copy of the successful validation JSON under `paper/runs/min5-2026-11-03/final_validation/<attempt>/`. Success requires a zero exit, no watchdog failure, and a newly written passed full report containing all required checks and the delivery fingerprint. An old passed report cannot establish success. The wrapper leaves the parent runner's `state.json`, `events.jsonl`, progress and completion markers intact; it does not publish or mark the overall project complete.

## Dependency order

1. **Prerequisites:** five-work CSR, weighted edges, Leiden communities, identity/link signals and `attrs.py`. Verify cached dimensions and node order against the five-work network.
2. **Baseline:** `a1_basics.py`, `a2_distances.py`, `a4_hubs.py`, `a1b_clustering.py`, `a3_farthest.py`, `a3b_far_edge.py`, `a9_structure.py`, `a9c_field_order.py`. A2 saves the source identities used by later paired comparisons.
3. **Identity evidence:** `a6_nonperson.py`; completed affiliation/live-work/snapshot reads; `a6_twins.py` for `zh`, `ja`, `ko` and `ru`; `a6_splits.py`. Calibrate each script against its own decoys.
4. **Team-size check:** `a2c_team_cap.py` caches a researcher–work bipartite graph and compares caps of 50, 100, 500 and all team sizes. The 1,000-source output is `a2c_team_cap.json`; the 100-source output is `a2c_team_cap_100.json`. Cap-50 histograms and reached counts must reproduce A2 for the same sources before interpreting other caps.
5. **Flags and description:** `flags.py`, `b3_hubs.py`, `a4b_hubs_analysis.py`, `a5_clusters.py`, `a5b_communities.py`, `b4_countries.py`, `b2_ties.py`, `b2b_long_links.py`, `b1_uncited.py`. Verify `cache/script_communities.json` against current community profiles, not old identifiers.
6. **Robustness:** `a7_remedy.py 25 strict`, `a7b_paired.py peak25_strict`, `a8_repair.py`, `a8b_communities.py`, `a8c_center.py`, `a9b_repaired.py`, `a6_validate.py`, `a6_institutions.py`, `a6_quality.py`.
7. **Map and deposit:** `map_recolour.py`, then `map_labels.py`; review labels and update `paper/deposit_README.md` with final counts; run `export_deposit.py` to stage `data/2026-09-23/min5/deposit/`. Export copies the README and hashes all files. Any later file change requires refreshed checksums.
8. **Final checks and paper:** `a10_checks.py` requires the staged `researchers.parquet`, A7 sets and B4 results, so run it after export. Generate figures, refresh claims/tables and run `paper/build.py`. Once the main runner has finished, `run_final_validation.py` performs the complete manuscript-required validation, including checksums and exact repaired-graph reconstruction, under the same memory watchdog. `quoted_numbers.py` compares selected results to `results_min15/`; it is not a complete manuscript audit.
9. **Review analyses:** `run_review.py` runs `rv_ties.py`, `rv_remoteness.py`, `rv_bipartite_null.py`, `rv_rich_club.py` and `rv_hub_links.py` after the main run and the comments round. They read A2c's researcher–work cache and the A1b, A9, A9c, A10, B3 and B2c results, and each writes `paper/results/rv_*.json`.

The API readers (`a6_churn.py`, `a6_affiliations.py`, `a6_live_works.py`) and snapshot reader (`a6_churn_snapshot.py`) cache their inputs. Reuse verified completed reads on resume, record acquisition dates, and distinguish snapshot counts from live-API counts. Local export and manuscript building do not upload or publish anything.

Repair-center searches explicitly use the unique component containing more than half of the repaired nodes. Only surviving sampled sources in that component screen the 200 exact-distance candidates; source exclusions and component size are reported. The all-source repaired mean still pools reachable targets and is not a fixed-population paired estimate. Distance-bin overflow and empty repaired field/country cells fail before reporting a number. See `paper/review/min5-repair-method-review.md`.

`map_labels.py` uses bounded DuckDB aggregation (4 GB, two threads, at most 8 GB disk spill), exact-string fallback membership and country shares among known-country records in communities with at least 1,000 members. Category ties use original node-row order explicitly; this can change label ordering or which equally common category appears at the three-category cutoff, while preserving counts, proportions and naming thresholds.

After full validation and visual review, run `paper/analysis/record_min5_provenance.py --final` with the paper Python environment. It records package and build-tool versions, source-code and artifact SHA-256 digests, the deposit's validated content hashes, and an explicitly metadata-only inventory for large inputs. This records the final evidence; it does not publish files or mark the analysis runner complete.

## Script-to-paper map

A2c's work and pair counts refer only to qualifying works with at least two
retained researchers. `pairs_kept` and `pairs_left_out` count pair incidences,
including repetitions across works, rather than unique graph links. The legacy
JSON key `max_researchers_on_a_work` is the maximum number of retained researchers
on an **excluded** work for that cap. Its uncapped value is zero because no work
is excluded; it is not the maximum observed team size of the full input. The
full qualifying-work census in A6 quality uses a different work population.

| Paper | Scripts |
|---|---|
| Section 2: network and measurements | `attrs.py`, `a1_basics.py`, `leiden.py`, `a10_checks.py` |
| Section 3: distances, clustering, far edge | `a2_distances.py`, `a1b_clustering.py`, `rv_bipartite_null.py`, `a3_farthest.py`, `a3b_far_edge.py` |
| Section 3: fields, countries, hub removal | `a9_structure.py`, `a9c_field_order.py`, `rv_remoteness.py`, `rv_hub_links.py` |
| Sections 4–5: communities and countries | `a5_clusters.py`, `a5b_communities.py`, `b4_countries.py`, `map_labels.py` |
| Section 6 / Tables 1a–1b: hubs and centre | `a4_hubs.py`, `a4b_hubs_analysis.py`, `b3_hubs.py`, `rv_rich_club.py`, `a8c_center.py` |
| Section 7 / Tables 2a–2b: weak ties, long links | `b2_ties.py`, `b2b_long_links.py`, `b2c_repeat_ties.py`, `rv_ties.py` |
| Section 8: uncited periphery | `b1_uncited.py`, `a6_twins.py`, `a10_checks.py` |
| Section 9 / Tables 3–4: robustness | `a7_remedy.py`, `a7b_paired.py`, `a8_repair.py`, `a8b_communities.py`, `a9b_repaired.py`, `a2c_team_cap.py`; `results_min15/` for threshold comparison |
| Appendix A.1: non-person records | `a6_nonperson.py`, `a6_quality.py`; historical hand-judged sample in `results_min15/` |
| Appendix A.2: merges | `a6_affiliations.py`, `a6_institutions.py`, `a6_validate.py` |
| Appendix A.3: twins | `jmnedict_readings.py`, `a6_twins.py` |
| Appendix A.4: duplicates | `a6_splits.py`, `a8_repair.py` |
| Appendix A.5: release changes | `a6_churn_snapshot.py`, `a6_live_works.py`, `a6_validate.py` |

Appendices A and B are in the paper's Online Resource 1.

`common.py` and `kernels.py` provide shared loaders, BFS, union-find and contraction. `a6_ego.py` and `a6_label_sample.py` are exploratory/hand-review tools; do not claim a new manual validation without actual judgments.

## Reviewable products

- `paper/RESULTS.md`: interpreted results, including the preserved core comparison.
- `paper/runs/min5-2026-11-03/`: progress, commands, exit states and provenance.
- `paper/results/` and `paper/cache/`: result JSON, logs and supporting tables/arrays.
- `paper/review/min5-manuscript-audit.md`: claim/artifact refresh checklist, including unmarked stale claims.
- `paper/figures/`: PDF/PNG previews; `paper/figures/submission/`: TIFF and EPS where appropriate. `figures.py` draws the result figures; `fig_concepts.py` draws Fig. 7.
- `data/2026-09-23/min5/deposit/`: local data package, README and `SHA256SUMS`.
- `paper/draft2.md`, `paper/build/manuscript.docx`, `paper/build/manuscript.pdf`: manuscript and build outputs. A successful build alone does not establish that every scientific claim is checked.
