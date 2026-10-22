# UAV training continuity

When resuming the W&B/Kaggle training audit, CPU replay work, or a previous
session, read [CURRENT_TASK.md](docs/superpowers/CURRENT_TASK.md) before editing
code or launching a job. It contains the current task, verified evidence,
next action, and continuation rules. Read its linked task brief only when
working on CPU replay.

Update the current-task record with the actual commit, verification result,
and next action before handing off or finishing a task. Commit and push
the record on the working branch so a fresh chat can retrieve it.
Keep timestamps on observations from running services; recheck live state
before making operational decisions.
