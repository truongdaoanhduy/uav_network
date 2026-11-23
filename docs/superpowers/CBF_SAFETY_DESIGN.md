# XYZ CBF Safety Design

## Intent
Replace the XYZ dynamic-window safety filter with an acceleration-space second-order CBF safety filter while leaving MASAC/MATD3 responsible for task/search decisions.

## Chosen approach
Use linear acceleration half-spaces derived from a short-horizon second-order barrier condition and solve the minimum-intervention safety problem with deterministic most-violated-constraint (Gauss-Southwell) projections. This avoids a new QP dependency and is suitable for a later Warp port.

For a static obstacle or peer along line-of-sight n, use a braking-horizon acceleration half-space derived from predicted clearance. The continuous-time paper form is adapted to this simulator's semi-implicit Euler update (`v[k+1]=v[k]+a*dt`, `p[k+1]=p[k]+v[k+1]*dt`), so the acceleration displacement coefficient is `0.5*H*(H+dt)` rather than `0.5*H^2`. The horizon H is at least one simulator step and expands to the available stopping horizon. Boundary planes use the analogous one-dimensional constraint. The filter projects nominal MARL accelerations only when a constraint is violated and clips to acceleration limits after each projection.

## Task split
1. Native CPU XYZ CBF core + tests.
2. Remove/migrate DWA configuration, telemetry, checkpoint compatibility, and 2D safety selection.
3. Implement Warp/GPU CBF parity.
4. Full verification and controlled safety ablation before new training.

## Non-goals for Task 1
No APF/ORCA/MPC stack, no learned CBF, no new solver dependency, no Warp implementation, and no launch/retrain.
