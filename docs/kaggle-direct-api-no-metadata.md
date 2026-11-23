# Kaggle direct API launcher — no `kernel-metadata.json`

The cloud launcher submits the generated private notebook through the official
Kaggle `save_kernel` SDK request. It does **not** call `kaggle kernels push`
or write `kernel-metadata.json`. Each run still generates its own `kernel.py`
(source payload) and `launch.json` (receipt).

API fields are constructed automatically from the existing job specification:

- `slug`, `new_title`, `text` (the generated Python script)
- `language=python`, `kernel_type=script`, `is_private=true`
- `enable_gpu` / `machine_shape=NvidiaTeslaT4` for GPU training
- `enable_internet=true`, `dataset_data_sources` for private W&B/HF inputs
- `session_timeout_seconds` from the run specification

The Kaggle API token from the selected account is passed only as the
`KAGGLE_API_TOKEN` environment variable to an isolated child process;
it is not placed in the script or in any API-request metadata field.

## Local requirements

Use the project's declared cloud dependencies (`pip install -e '.[cloud]'`)
or at minimum a Kaggle SDK that supports `machine_shape`:

```bash
python -m pip install --upgrade 'kaggle>=2.2,<3'
```

The installed SDK in the inspected Sandbox was `kaggle==1.7.4.5`, which **does
not** support explicitly selecting T4 ×2 via `ApiSaveKernelRequest.machine_shape`.
The new launcher fails before submitting when that field is missing; it does
not silently fall back to a different GPU. Upgrading Kaggle is required.

Do not use this path to evade a tool-level safety restriction. An API submission
can still be rejected by the platform or blocked by the tool environment.
The tests in `tests/test_kaggle_direct_submit.py` exercise the construction and
response validation with a fake SDK and do **not** create cloud jobs.
