# Synthetic benchmark

Full pipeline on rendered videos (VGGT reconstruction, not ground-truth depth). Positions are scored after a similarity alignment of the camera centres to ground truth (the metric scale is estimated by a monocular network, so it is not exact). Reproduce: `uv run python scripts/benchmark.py`.

| scenario | what happens | changes found | position error | must NOT be reported | false changes | null test (false clusters) |
|---|---|---|---|---|---|---|
| standard | chair moved 1.7 m + turned 35°, bin removed, backpack added; after-video on another path/height/exposure | 3/3 | moved 4 cm, removed 14 cm, added 10 cm | – | 0 | 0, 0 |
| hard | chair moved, 1 cm rug removed (appearance only), removals the after-video cannot verify (never filmed / behind a cabinet / never visible before); after-video covers part of the room | 2/3 | moved 10 cm, appearance 16 cm | plant: ✅ not reported, crate: ✅ not reported, bin: ✅ not reported | 0 | 0, 0 |
| nochange | same room filmed twice on different paths | 0/0 | – | – | 0 | 0, 0 |
