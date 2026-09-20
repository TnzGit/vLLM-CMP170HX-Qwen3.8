# Phase 2 attribution — both 126K runs, and which number the table quotes

The 126K trace was captured twice (the first trace was cleared by the 250K leg, so the
context was re-run). Both runs used the identical configuration, and they agree to **0.75%**:

| run | ms/output-token | ms/spec-iteration | accepted/pass | output tok/s | rounds |
| --- | --- | --- | --- | --- | --- |
| 1st (quoted in `phase2-m7-attribution.md`) | **10.935** | **34.637** | 3.1866 | 91.477 | 3 (2 unprofiled) |
| 2nd (the re-run, committed as `ph2_126000.json`) | 11.018 | 34.900 | 3.1866 | 90.771 | 3 (2 unprofiled) |

`docs/phase2-m7-attribution.md` quotes the **first** run because that is the one whose
component split is shown there. The committed JSON is the **second** run, because it is the
one whose trace survived on disk.

Both are legitimate; the difference (0.26 ms/spec-iteration, 0.75%) is inside the run-to-run
spread already recorded for this path (Phase 3 measured round spreads of 1.006–1.17x on the
same engine). Neither is a correction of the other.

**Rule this follows:** a committed artifact must be traceable to the table that cites it. When
a re-run overwrites a result, the mismatch is recorded explicitly rather than left for a
reader to discover — which is how this one was found.
