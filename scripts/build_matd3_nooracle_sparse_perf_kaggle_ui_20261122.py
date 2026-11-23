"""Build an independent V9-SPARSEFAST Kaggle notebook with pinned source hash.

Keeps the 3-cell E4096/U4/B8192, Val100, Test4000, W&B + HF workflow.
It does NOT alter or resubmit the running V9 e079eace notebook.
"""
from pathlib import Path
import argparse
import ast
import nbformat
from scripts.build_matd3_reference_style_kaggle_ui import REFERENCE_NOTEBOOK, build

OLD = "NOORACLE-V9"
NEW = "NOORACLE-V9-SPARSEFAST"


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=REFERENCE_NOTEBOOK)
    opts = parser.parse_args()
    notebook = build(opts.reference, root)
    assert len(notebook.cells) == 3
    assert [c.cell_type for c in notebook.cells] == ["markdown", "code", "code"]
    for cell in notebook.cells:
        cell.source = cell.source.replace(OLD, NEW)
        if cell.cell_type == "code":
            ast.parse(cell.source)
    notebook.cells[0].source += (
        "\n\n**V9-SPARSEFAST** changes only the internal Warp iteration"
        " over unused observed-cell report slots; science settings, 140-slot"
        " capacity, replay, checkpoints, W&B and HF diagnostics remain V9."
        " GPU T4x2 throughput and complete end-to-end run have not been"
        " independently validated yet. This notebook is a separate A/B run."
    )
    notebook.metadata["performance_variant"] = NEW
    notebook.metadata["change_scope"] = "internal_unused_report_slot_iteration_only"
    notebook.metadata["requires_gpu_throughput_verification"] = True
    python_bytes = sum(len(c.source.encode("utf-8")) for c in notebook.cells if c.cell_type == "code")
    output_bytes = len(nbformat.writes(notebook).encode("utf-8"))
    if max(python_bytes, output_bytes) >= 1_000_000:
        raise ValueError(f"Notebook over Kaggle size guard: {python_bytes}, {output_bytes}")
    opts.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, opts.output)
    print("NOTEBOOK", opts.output, "bytes", opts.output.stat().st_size)
    print("SOURCE_COMMIT", notebook.metadata["reference_commit"])
    print("SOURCE_SHA256", notebook.metadata["source_sha256"])


if __name__ == "__main__":
    main()
