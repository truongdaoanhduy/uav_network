# Shield-aware reward and policy-safety telemetry

## Intent
CBF remains a hard last-resort safety filter. Training reward and W&B must not treat a trajectory rescued by CBF/boundary fallback as if the raw policy produced it safely.

## Required behavior
- Preserve latest-braking-viability CBF geometry and pass-through behavior for already-safe commands.
- Measure raw one-step unsafe attempts before CBF: boundary exit, obstacle swept collision, peer swept-separation violation.
- Penalize raw unsafe attempts even when CBF/fallback prevents the physical collision.
- Penalize CBF intervention frequency separately so frequent preventive dependence is not free.
- Measure shield correction in normalized **action/acceleration space**, not divided by max flight speed.
- Log raw unsafe-attempt rates, CBF rescue/dependency rates, and reward penalties separately.
- Keep CPU and Warp behavior/telemetry/reward numerically aligned.
- Make training-history aggregates explicit; do not label whole-training means as final-policy performance.
- Canonical 20k config enables the new unsafe-action penalty; legacy configs remain compatible through a zero default.

## Reward semantics
`unsafe_action` is a nonnegative coefficient. Per step, `unsafe_action_step` is the fraction of active UAVs whose raw nominal one-step segment would violate a hard boundary/obstacle/peer safety condition before the shield. Reward adds `-unsafe_action * unsafe_action_step`.

`shield_correction` continues to penalize preventive CBF dependence, but its correction magnitude is the mean active-UAV norm of `(safe_accel - nominal_accel) / max_accel`, making it directly interpretable as fraction of commanded actuation overridden.

`shield_intervention` is a nonnegative coefficient applied to the fraction of active UAVs whose acceleration was modified by CBF on the current step. This penalizes repeated reliance even when each individual correction is small.

## Non-goals
- Do not remove CBF.
- Do not relax stopping-distance safety to one-step-only collision checks.
- Do not store post-CBF actions in replay in this change; raw-action replay remains the MDP action, while telemetry/reward makes shield dependence observable and costly.

## Verified interpretation

CBF remains preventive at the latest braking-viable point; it is not changed into a collision-only alarm. High shield dependence is now explicitly costly and separately visible, so a rescued trajectory cannot be interpreted as equivalent to a policy that was safe without intervention. Short MATD3 microtraining shows the unsafe/CBF rates decrease with additional updates, but full-cohort convergence remains a separate experiment.
