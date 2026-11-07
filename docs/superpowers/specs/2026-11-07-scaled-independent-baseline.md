# Scaled independent baseline specification

User requested two hidden layers of 512 neurons and 35 targets/25 obstacles.
The question about updates16/32 is advisory; keep updates8 until selected.
Create a separate experiment, preserving the delivered20t15o/256 notebooks.

Use the existing independent source recipe for BOTH MASAC and MATD3.
Copy only world target/obstacle counts and both hidden sizes into a separate
profile; keep map1000m,6UAV,1500step cap, battery, reward, CBF, observations,
sensor slots, hyperparameters, seeds and evaluation schedule unchanged.
Use distinct output/group/tags and notebooks labelled35T25O/H512/U8.
Do not launch training. Source archives must be committed and identical
between the matched pair. Existing default builder calls remain compatible.

Explain exact dimensions, update semantics, parameter/memory cost and limits.
Changing width and scene together cannot establish a width effect.
Compare U8/U16/U32 within the same scene/width and transition budgets.
No claim of smoother curves, improved learning, T4 fit or MASAC superiority.
