# Kaggle package annotations (not training parameters)

The Kaggle CLI accepts operational fields such as `id`, `title`, `code_file`,
`enable_gpu`, `is_private`, and `dataset_sources`. It **does not transmit** the
arbitrary `x_uav_annotations` or `x_package_*` keys as Kaggle server metadata.
These keys exist in the generated local `kernel-metadata.json`. A corresponding
comment is embedded in uploaded `kernel.py` for readers of that code file.

The launcher creates unique random package keys and comments for **each** run,
and uses different built-in maps for MASAC and MATD3. They cannot affect
algorithm configuration, GPUs, credentials, or replay contents.

To add your own JSON key-value pairs (arbitrary nested JSON objects, arrays,
numbers, strings, and booleans), copy
`configs/kaggle-metadata-extras.example.json` to
`kaggle-metadata-extras.json` **next to the selected `key.txt`**. The launcher
will automatically read this file for subsequent launches, without needing to
modify and commit the Git worktree. You can also pass
`--metadata-extras-file /absolute/path/to/custom.json` to override it.

Structure:

```json
{
  "common": {"my_label": "any value", "nested": {"a": 123}},
  "masac": {"only_masac": [1, 2, 3]},
  "matd3": {"only_matd3": false}
}
```

Only the `common` section plus the selected algorithm section appear in
`x_uav_annotations.custom` and uploaded kernel comments; the other algorithm's
map is not included there. Kaggle operational keys cannot be overridden by
this file. The extra data is limited to 8 KiB and must never contain secrets.

These decorations do **not** hide or encrypt source. Protect unpublished work
using a private repository and private notebooks with access controls.

Note: when manually importing a `.ipynb` through Kaggle's UI, Kaggle does not
consume a separate `kernel-metadata.json`. This feature applies to the
repository's Kaggle CLI submission paths, not to already exported notebooks.
