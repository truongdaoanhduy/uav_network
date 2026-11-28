# V124 lưu nguyên các chú thích source tách ra để đáp ứng 500-line policy

Những dòng comment bỏ khỏi `environment/transition.py` chỉ nhằm giảm độ dài tệp, không thay đổi code chạy. Vị trí dựa trên V124 trước format.

- Dòng 4: `(blank)`
- Dòng 5: `(blank)`
- Dòng 7: `(blank)`
- Dòng 13: `(blank)`
- Dòng 56: `(blank)`
- Dòng 90: `(blank)`
- Dòng 110: `(blank)`
- Dòng 125: `# Use start and midpoint geometry in two equal radio sub-intervals.`
- Dòng 126: `# Save/restore the physical state: the authoritative flight update,`
- Dòng 127: `# safety and energy accounting still happen together below.`
- Dòng 265: `# One infeasible actor must not suppress sensing for every *other* UAV.`
- Dòng 266: `# Pairwise violations affect both participants; GCS/team penalties remain global.`
- Dòng 271: `# V113 P0: CPU previously detected invalid swept motion and still COMMITTED`
- Dòng 272: `# that motion. Match V80 GPU fail-closed terminal physical commit: preserve`
- Dòng 273: `# positions, previous velocities, and battery. Network has already advanced.`
- Dòng 274: `# This is a terminal emergency rollback, not a guaranteed viable-set shield.`
- Dòng 275: `# Legacy safety_failure_terminate=False permits other UAVs to sense.`
- Dòng 276: `# That legacy mode is NOT supported by the canonical Warp scenario;`
- Dòng 277: `# fail-closed physical rollback applies to the active terminal mode.`
- Dòng 307: `# V123 optional pilot: physically derived off-nadir optical cone.`
- Dòng 308: `# Canonical nadir disk and its RNG order remain byte-identical.`
- Dòng 344: `# optional cooldown, before RNG and ledger commit; never`
- Dòng 345: `# count a held observation as newly observed evidence.`
