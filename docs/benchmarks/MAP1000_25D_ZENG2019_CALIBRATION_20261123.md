# Opt-in Zeng 2019 rotary-wing propulsion — scientific calibration note

## Branch and boundaries

Branch: experiment/map1000-25d-zeng-energy-ablation-20261123
Parent: 546fd6b54664a53d4d3e0a03fd8cb77af56857ac (fully source-pinned 2.5D+AoI notebook)
Config: configs/matd3-nooracle-map1000-25d-aoi-zeng2019-h35-130-vz4-20261123.yaml

This is **a NEW scientific MDP**. Existing corrected MAP1000 and 2.5D quadratic-energy baseline configurations are preserved exactly. Must train from scratch. No claimed trained policy or T4 speed benefit.

## Parameter provenance

UavNetSim's pinned energy module (local git SHA 04daafb815eb377409b40b285574eeb62b9a8d58) implements the Zeng, Xu & Zhang IEEE TWC 2019 forward-flight law, with original defaults weight=100N, omega=400rad/s, radius=0.5m, U_tip=500m/s, v0=7.2m/s, rotor area=0.79m², rho=1.225kg/m³, solidity=.05, profile drag=.012, induced correction=.1, fuselage drag=.3. Those ORIGINAL values imply ~1.37 kW hover, making battery 277.2kJ last only ~202s, inconsistent with the 1500s experiment.

For this NEW opt-in hypothesis, calibrate an *illustrative* 2kg (~20N) rotor with omega=220rad/s and R=.5m. Enforce physical relationships Utip=omega*R=110m/s, v0=sqrt(weight/(2*rho*A))=3.21454m/s. Keep other coefficients above. This is NOT manufacturer-validated or flight-tested and must be disclosed as a simulation assumption.

Derived:
- P0 = (delta/8)*rho*s*A*omega^3*R^3 = 96.606 W.
- Pi = (1+k)*weight^(3/2)/sqrt(2*rho*A) = 70.720 W.
- Parasitic drag coefficient D = 0.5*d0*rho*s*A = 0.007258125 W/(m/s)^3.
- Forward-flight power P(v) = P0*(1+3*v²/Utip²) + Pi*sqrt(sqrt(1+v⁴/(4*v0⁴))-v²/(2*v0²)) + D*v³.
- 2.5D approximation uses horizontal speed v=sqrt(vx²+vy²); positive climb adds W*max(vz,0) watts. No energy recovery from descents; acceleration power/turn losses are *not modeled* in this opt-in steady-flight law. This limitation must be stated in a paper.

Expected power and continuous-flight upper bounds for an idealized battery (277200J):

| Ground speed (m/s) | Power (W) | Approx. endurance (s) |
| --- | ---: | ---: |
| 0 (hover) | 167.325 | 1656.7 |
| 5 | 140.533 | 1972.5 |
| 10 | 128.873 | 2151.0 |
| 12 | 131.493 | 2108.1 |
| 15 | 141.630 | 1957.2 |
| 20 | 175.614 | 1578.5 |

This restores the physically meaningful cruise-vs-hover energy trade-off. The last figure is an idealized constant-horizontal-speed bound, not a mission success estimate.

## What is implemented

- Config gate energy.propulsion_model = quadratic | zeng2019, default quadratic.
- Rotor physical parameter validation, source file model equation and derived constants.
- Both native NumPy energy and Warp XYZ cost/funding use the same Zeng2019 steady-flight law and vertical-climb work when opted in.
- The PHY/MAC, Bayesian target belief, false reports and AoI delivery reward are identical to the 2.5D no-Zeng experiment.
- Tests validate energy curve, native/Warp one-step agreement, and invalid parameters.

## What is NOT established

- This is not the UavNetSim 100N aircraft; translating weights without recalibration is physically wrong.
- The rotary model is calibrated by plausible target endurance, not airframe experimental measurements. True 3D climb dynamics, descent recovery, turning and rotor transients are approximated or omitted.
- No full 20k MATD3/MASAC training, no T4 profiling/validation, no learned role assignment and no beacon yet. Results need 5+ seeds and held-out Val100/Test1000.
- This change should be the *energy-model ablation*, not silently mixed with performance optimization and CBF.
