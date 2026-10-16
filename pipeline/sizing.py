"""Citation count -> marker size: one rule for raster tiles, point binaries and viewer.

Chosen 2026-10-10 from side-by-side renders of real crops (dense, medium and
sparse regions at z8-z10) comparing log(1 + c), sqrt(c), and log with knees
at 30, 100 and 300 citations against the old three-tier splats. Plain log
made the median author (358 citations) a mid-size disc, so dense regions read
as uniform blobs with weak hierarchy; sqrt left all but the elite as single
pixels. Log with a 100-citation knee keeps uncited and lightly cited authors
smallest while still growing steadily across 1k-150k citations:

    t(c) = clip(log(1 + c / KNEE) / log(1 + CAP / KNEE), 0, 1)

CAP is about the 99.99th percentile (148,985 in the 2026-09-23 release).

Radius in screen pixels at zoom z: a base dot that scales with the map (so
dense regions never carpet over when zoomed out) plus a citation bonus that
shrinks more slowly (so notable authors stay visible at z8-z9):

    r(z, t) = BASE_PX * 2**(z - 10) + BONUS_PX * t**GAMMA * 2**((z - 10) / 2)

At z10 that runs from 0.6 px (uncited) to 7.6 px (CAP and above). Eric asked
for more contrast than the first cut (1 + 3.5 t, max 4.5 px); of three
renders this "strongest" curve won: GAMMA > 1 keeps typical authors small so
only the top stands out.

Zooms here are TILE zooms of the 256px raster tiles. MapLibre counts zoom in
512px tiles, so the viewer at map zoom z shows tile zoom z + 1: the z10 tiles
are pixel-exact at map zoom 9 and magnified 2^1.25 when the deep-zoom points
take over at map zoom 10.25. The viewer's points (web/dev.html: SIZE_KNEE,
SIZE_CAP, SIZE_BASE_PX, SIZE_BONUS_PX, SIZE_GAMMA, TILE_ZOOM_OFFSET) follow
this formula plus that magnification up to the hand-off, so dots keep their
size when they turn clickable, then grow the bonus at the half rate again.
"""
import numpy as np

KNEE = 100.0
CAP = 150_000.0
BASE_PX = 0.6       # radius of an uncited author at REF_ZOOM
BONUS_PX = 7.0      # extra radius at t = 1, at REF_ZOOM
GAMMA = 1.3         # >1 keeps mid-cited authors small, so the top stands out
REF_ZOOM = 10
MIN_DRAW_PX = 1.2   # raster: below this the author's own density pixel shows it
DISC_MIN_ZOOM = 8   # raster: below this the map is a density picture, no discs


def size_t(cited) -> np.ndarray:
    """Citation counts -> size parameter in [0, 1] (log scale above KNEE)."""
    c = np.maximum(np.asarray(cited, dtype=np.float64), 0.0)
    return np.clip(np.log1p(c / KNEE) / np.log1p(CAP / KNEE), 0.0, 1.0)


def radius_px(zoom: float, t) -> np.ndarray:
    """Marker radius in screen pixels at `zoom` for size parameter `t`."""
    t = np.asarray(t, dtype=np.float64)
    return (BASE_PX * 2.0 ** (zoom - REF_ZOOM)
            + BONUS_PX * t ** GAMMA * 2.0 ** ((zoom - REF_ZOOM) / 2.0))


def min_t_drawn(zoom: float) -> float:
    """Smallest t whose disc reaches MIN_DRAW_PX at `zoom`."""
    room = MIN_DRAW_PX - BASE_PX * 2.0 ** (zoom - REF_ZOOM)
    if room <= 0:
        return 0.0
    return float((room / (BONUS_PX * 2.0 ** ((zoom - REF_ZOOM) / 2.0))) ** (1.0 / GAMMA))


def brightness(t) -> np.ndarray:
    """Marker brightness factor: cited authors glow a little more."""
    return 0.55 + 0.45 * np.asarray(t, dtype=np.float64)


def size_byte(cited) -> np.ndarray:
    """size_t quantized to uint8, as stored in attrs.bin for the viewer."""
    return np.round(size_t(cited) * 255.0).astype(np.uint8)
