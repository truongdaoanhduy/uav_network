# Optional launcher annotations (no Kaggle metadata file)

**The current launcher uses Kaggle's direct `save_kernel` API.** No
`kernel-metadata.json` is created or submitted, and no arbitrary JSON keys
are sent to Kaggle as metadata. The API request itself includes the actual
needed options: notebook ref, source, private, GPU, Internet, and input datasets.

The optional local file `kaggle-metadata-extras.json` beside `key.txt` can still
supply harmless, per-algorithm *comments* inside the uploaded `kernel.py`.
The file is **not** a Kaggle API metadata file. Its key-value pairs do not
control GPU selection, credentials, source identity, or training behavior.

Use `configs/kaggle-metadata-extras.example.json` as a template:

```json
{
  "common": {"my_label": "optional text"},
  "masac": {"masac_note": 12},
  "matd3": {"matd3_note": 34}
}
```

You can edit these fields without committing code; they are optional. Adding
comments does **not** protect source code from someone with notebook access.

For Kaggle UI imports (`.ipynb`), the separate CLI metadata file is likewise
unnecessary; the UI supplies the server's notebook settings.
