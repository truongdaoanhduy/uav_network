"""Readable scientific identities shared by cloud launch and W&B tracking."""
import re


def compact_count(value):
    for scale, suffix in ((1_000_000, "M"), (1000, "K")):
        if value >= scale and value % scale == 0:
            return str(value // scale) + suffix
    return str(value)


def replay_layout(config):
    runtime = config.runtime
    if runtime.replay_cpu_fraction:
        cpu = round(runtime.replay_cpu_fraction * 100)
        return f"HYBRID-C{cpu}-G{100-cpu}"
    if runtime.replay_device == "cpu":
        return "CPU"
    return "FULLGPU" if runtime.replay_device == "cuda" else "AUTO"


def run_name(config, algorithm, mode, stage, suffix):
    settings = getattr(config, algorithm)
    stage = {"benchmark": "BENCH", "train": "TRAIN", "validate": "VALIDATE"}.get(stage, stage.upper())
    return (f"{stage}-{algorithm.upper()}-{mode.upper()}-{replay_layout(config)}"
            f"-E{config.runtime.num_envs}-B{settings.batch_size}"
            f"-R{compact_count(settings.replay_capacity)}-G{config.world.generation_version}"
            f"-S{config.runtime.seed}-{suffix}")


def kernel_slug(owner, config, algorithm, mode, stage, stamp, token):
    layout = replay_layout(config)
    layout = "h" + layout.split("-")[1][1:] if layout.startswith("HYBRID") else layout.lower()
    prefix = f"{stage[:3]}-g{config.world.generation_version}-{algorithm}-{mode}-{layout}-{stamp[4:8]}{stamp[9:13]}"
    return prefix[:50 - len(owner) - 1 - len(token) - 1].rstrip("-") + "-" + token


def authoritative_tags(config):
    tags = [tag for tag in config.tracking.tags
            if not re.fullmatch(
                r"cpu\d+[-_]gpu\d+|replay-(?:fullgpu|cpu|auto|hybrid-c\d+-g\d+)|"
                r"amp-(?:fp16|off)|torch-compile|compile-off|fused-adam|fused-off",
                tag, re.I)]
    runtime = config.runtime
    acceleration = [
        "amp-fp16" if runtime.amp_mode == "fp16" else "amp-off",
        "torch-compile" if runtime.compile_models else "compile-off",
        "fused-adam" if runtime.fused_optimizers else "fused-off",
    ]
    return list(dict.fromkeys([*tags, *acceleration,
                               "replay-" + replay_layout(config).lower(),
                               f"generation{config.world.generation_version}"]))


def related_run_name(root_name, stage, suffix):
    for prefix in ("TRAIN-", "VALIDATE-", "BENCH-"):
        if root_name.startswith(prefix):
            root_name = root_name[len(prefix):]
            break
    return f"{stage}-{root_name}-{suffix}"
