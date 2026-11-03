# Owner-initiated Kaggle launcher — matched seed-44 study

This launch helper is for the **10 pre-defined experiments** on branch
`experiment/matched-architecture-seed44-20261104`. It submits **one case per invocation**
under the account of the person running it; it does not switch among Kaggle
accounts or pool GPU quotas. The person whose account is being used must
run the command themselves in accordance with the Kaggle service terms.

## Before running

- From the repository root, check out the **published** experiment branch.
- Install the dependencies according to this project's `setup.sh`.
- Keep a local, uncommitted `key.txt` or use `--key-file /path/to/team-only.json`.
  It needs the team's `wandb` (API key, entity, project) and `huggingface`
  (token) sections; **no other person's Kaggle tokens are needed**.
- The account owner generates a Kaggle API token in their own Kaggle settings.
  Never paste it into a notebook, commit it, or send it in chat.
- Kaggle access, available GPU quota, W&B/HF access, private datasets and
  Kaggle internet/GPU permissions are checked by the existing cloud launcher.

## Commands (the owner runs them on their own account)

```bash
# Preview case 04 (MASAC PQL, shared actor and critic) without API or GPU use.
python -m scripts.run_owned_kaggle_study_case --case 4

# Securely set the owner's Kaggle API token only in this terminal session.
read -rsp 'Kaggle API token: ' KAGGLE_API_TOKEN; echo
export KAGGLE_API_TOKEN

# Submit the single chosen case (valid IDs: 1 through 10).
python -m scripts.run_owned_kaggle_study_case --case 4 --submit
unset KAGGLE_API_TOKEN
```

Use `--key-file /path/to/team-only.json` if the credential file is not at
`key.txt`. For the full experimental matrix, each authorized participant
can choose their assigned case from 1–10 and initiate it independently.
The launcher creates a private temporary single-account credential file,
uses the existing Kaggle cloud launch path, then deletes the temporary file.
No source/config YAML under ignored `outputs/` is referenced remotely:
Kaggle checks out the published commit and resolves the canonical YAML
with exactly the selected overrides.

The owner receives Kaggle and W&B URLs. The local submission receipt lives at:

```
outputs/matched-architecture-seed44-20261104/kaggle-submissions/case-NN.json
```

The launcher rejects a second submission for the same case from the same
workspace if the receipt already exists; inspect the existing Kaggle run
before retrying. These receipts are **local** and not committed to GitHub.

## Important scientific limits

All cases use `seed=44`, `deterministic=true`, `collector_overlap=false`,
FP32, 4096 environments, and the canonical 20k-episode reference.
The MASAC PQL shared replica pair is case **04 vs 10**.
The MASAC BASE shared `vectorize_critics` pair is **02 vs 09**. In this code
shared MASAC critic does not use the independent-critic `torch.vmap` fast
path, so **02 vs 09 is NOT a valid speedup demonstration of vmap**.
Use an additional *independent-critic* paired run for a useful GPU vmap
speed experiment, and compare fixed numbers of updates and env steps.

`runtime.deterministic=true` disables concurrent PQL workers. One Kaggle
session can end before completing 20k episodes because the reference also
limits training wall clock. This helper does **not** enable the disabled
GitHub supervisor or promise automatic multi-session continuation.

Tests: `python -m pytest -q tests/test_owned_kaggle_study_case.py
 tests/test_matched_architecture_study.py`.
