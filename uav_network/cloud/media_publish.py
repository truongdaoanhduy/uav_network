"""Publish already-rendered Kaggle CPU visualization outputs to public HF and W&B."""
import argparse
import json
import os
from pathlib import Path

from ..configuration import load_config
from .client import kernel_status, run_kaggle
from .continuation import kernel_ref_for_segment
from .credentials import load_credentials
from .visualize import publish_visualization_media

ROOT = Path(__file__).resolve().parents[2]


def _resolved_config(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _local_result(output, algorithm, result):
    folder = Path(output) / "visualize" / algorithm
    image = folder / Path(result.get("render_path", "final_episode.png")).name
    video_name = Path(result.get("animation_path", "")).name
    if not video_name:
        candidates = list(folder.glob("final_episode.mp4")) + list(folder.glob("final_episode.gif"))
        if len(candidates) != 1:
            raise ValueError("Downloaded visualization has no unambiguous animation")
        video = candidates[0]
    else:
        video = folder / video_name
    metrics = folder / "metrics.json"
    if not image.is_file() or not video.is_file() or not metrics.is_file():
        raise FileNotFoundError("Downloaded Kaggle visualization output is incomplete")
    resolved = dict(result)
    resolved["render_path"] = str(image)
    resolved["animation_path"] = str(video)
    return resolved


def publish_downloaded_visualizations(output, key_file="key.txt"):
    """Resume the existing CPU W&B run and publish downloaded media without retraining."""
    output, key_file = Path(output), Path(key_file)
    spec = json.loads((output / "job.json").read_text())
    results = json.loads((output / "visualization-result.json").read_text())
    credentials = load_credentials(key_file)
    if (spec.get("entity"), spec.get("project")) != (credentials.entity, credentials.project):
        raise ValueError("Downloaded visualization W&B destination differs from credentials")
    config = load_config(_resolved_config(spec["config"]), spec.get("overrides", []))
    config.tracking.credentials_file = str(key_file.resolve())
    config.tracking.entity, config.tracking.project = spec["entity"], spec["project"]
    if spec.get("checkpoint_repo"):
        config.tracking.checkpoint_repo = spec["checkpoint_repo"]
    os.environ["WANDB_API_KEY"] = credentials.wandb_key
    import wandb
    publications = {}
    for algorithm, raw in results.items():
        if algorithm not in {"masac", "matd3"}:
            continue
        result = _local_result(output, algorithm, raw)
        run_id = result.get("wandb_run_id") or spec.get("runs", {}).get(algorithm, {}).get("id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("Downloaded visualization is missing its W&B run ID")
        with wandb.init(entity=spec["entity"], project=spec["project"], id=run_id,
                        resume="must", reinit="finish_previous", mode="online") as run:
            publication = publish_visualization_media(config, spec, algorithm, run_id, result, run,
                                                      fps=int(spec.get("cpu_eval", {}).get("fps", 10)))
            try:
                run.summary["visualization/posthoc_published"] = True
            except Exception:
                pass
        publications[algorithm] = publication
    (output / "visualization-publication.json").write_text(json.dumps(publications, indent=2))
    return publications


def download_and_publish_case(manifest_path, destination, key_file="key.txt"):
    """Download a completed CPU child with Kaggle API, then publish its media copies."""
    manifest_path, destination, key_file = Path(manifest_path), Path(destination), Path(key_file)
    manifest = json.loads(manifest_path.read_text())
    credentials = load_credentials(key_file)
    account = next((item for item in credentials.accounts if item.name == manifest.get("account")), None)
    if account is None:
        raise RuntimeError("Launch manifest account is missing from credentials")
    root_ref = manifest.get("root_kernel_ref", manifest.get("kernel_ref"))
    cpu_ref = kernel_ref_for_segment(root_ref, 0, cpu=True)
    marker = destination / "visualization-publication.json"
    if marker.is_file():
        return {"status": "published", "cpu_kernel_ref": cpu_ref,
                "publication": json.loads(marker.read_text()), "cached": True}
    status = kernel_status(cpu_ref, account)
    if status != "complete":
        return {"status": "pending", "cpu_kernel_ref": cpu_ref, "kernel_status": status}
    destination.mkdir(parents=True, exist_ok=True)
    run_kaggle(["kernels", "output", cpu_ref, "-p", str(destination), "-q"],
               account, timeout=300)
    publication = publish_downloaded_visualizations(destination, key_file)
    return {"status": "published", "cpu_kernel_ref": cpu_ref, "publication": publication}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Publish UAV visualization media to public HF and W&B")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", help="Already-downloaded Kaggle CPU output folder")
    group.add_argument("--manifest", help="Training launch manifest; download CPU output via Kaggle API first")
    parser.add_argument("--destination", help="Destination folder used with --manifest")
    parser.add_argument("--key-file", default=str(ROOT / "key.txt"))
    args = parser.parse_args(argv)
    if args.manifest:
        if not args.destination:
            parser.error("--destination is required with --manifest")
        result = download_and_publish_case(args.manifest, args.destination, args.key_file)
    else:
        result = publish_downloaded_visualizations(args.output, args.key_file)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
