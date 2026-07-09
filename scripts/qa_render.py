"""Quick-look renders of coords.parquet - one full view + one zoom, colored by community."""
import argparse
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render(coords: str, out_dir: Path, sample: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = duckdb.sql(
        f"""
        SELECT x, y, community % 40 AS hue
        FROM '{coords}'
        WHERE x IS NOT NULL
        USING SAMPLE {sample} ROWS (reservoir, 42)
        """
    ).df()
    for name, (w, h) in {"overview": (16, 16), "zoom": (16, 16)}.items():
        fig, ax = plt.subplots(figsize=(w, h), facecolor="black")
        d = df
        if name == "zoom":
            cx, cy = df.x.median(), df.y.median()
            sx, sy = df.x.std() * 0.15, df.y.std() * 0.15
            d = df[(df.x.between(cx - sx, cx + sx)) & (df.y.between(cy - sy, cy + sy))]
        ax.scatter(d.x, d.y, s=0.05, c=d.hue, cmap="tab20", alpha=0.5, linewidths=0)
        ax.set_axis_off()
        fig.savefig(out_dir / f"{name}.png", dpi=150, bbox_inches="tight",
                    facecolor="black")
        plt.close(fig)
        print(f"wrote {out_dir / f'{name}.png'} ({len(d):,} points)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("coords")
    ap.add_argument("--out", default="data/qa")
    ap.add_argument("--sample", type=int, default=2_000_000)
    args = ap.parse_args()
    render(args.coords, Path(args.out), args.sample)
