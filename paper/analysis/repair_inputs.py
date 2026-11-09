"""Read A8's small hub candidate table without materialising all author names."""
import numpy as np


def hub_metadata(connection, targets, nodes, snapshot, identity, full_names):
    """Return in target-row order, preserving missing/zero work-count ratios.

    targets has numeric id and node_idx columns. The source populations stay in
    Parquet; only the selected hub records enter pandas. Identity/full-name
    signals are intentionally still those of the current five-work graph.
    """
    selected = targets.copy()
    selected["target_order"] = np.arange(len(selected))
    connection.register("repair_targets", selected)
    try:
        return connection.execute(f"""
            SELECT t.node_idx, k.key_n, f.full_n,
                   o.works_count AS old_works, s.works_count AS new_works
            FROM repair_targets t
            LEFT JOIN read_parquet('{nodes}') o
              ON CAST(regexp_extract(o.id, 'A([0-9]+)$', 1) AS BIGINT) = t.id
            LEFT JOIN read_parquet('{snapshot}') s ON s.id = t.id
            LEFT JOIN read_parquet('{identity}') k ON k.node_idx = t.node_idx
            LEFT JOIN read_parquet('{full_names}') f ON f.node_idx = t.node_idx
            ORDER BY t.target_order
        """).fetchdf().set_index("node_idx")
    finally:
        connection.unregister("repair_targets")
