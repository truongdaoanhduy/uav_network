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


def sharing_marker(config, algorithm):
    settings = getattr(config, algorithm)
    if settings.share_actor_parameters and settings.share_critic_parameters:
        return "SHARED-ACTOR-CRITIC"
    if settings.share_actor_parameters:
        return "SHARED-ACTOR"
    if settings.share_critic_parameters:
        return "SHARED-CRITIC"
    return None


def run_name(config, algorithm, mode, stage, suffix):
    settings = getattr(config, algorithm)
    stage = {"benchmark": "BENCH", "train": "TRAIN", "validate": "VALIDATE"}.get(stage, stage.upper())
    sharing = sharing_marker(config, algorithm)
    sharing = f"-{sharing}" if sharing else ""
    return (f"{stage}-{algorithm.upper()}-{mode.upper()}{sharing}-{replay_layout(config)}"
            f"-E{config.runtime.num_envs}-B{settings.batch_size}"
            f"-R{compact_count(settings.replay_capacity)}-G{config.world.generation_version}"
            f"-S{config.runtime.seed}-{suffix}")


def kernel_slug(owner, config, algorithm, mode, stage, stamp, token, external_supervisor=False):
    layout = replay_layout(config)
    layout = "h" + layout.split("-")[1][1:] if layout.startswith("HYBRID") else layout.lower()
    marker = "gh-" if external_supervisor else ""
    prefix = f"{marker}{stage[:3]}-g{config.world.generation_version}-{algorithm}-{mode}-{layout}-{stamp[4:8]}{stamp[9:13]}"
    return prefix[:50 - len(owner) - 1 - len(token) - 1].rstrip("-") + "-" + token


def authoritative_tags(config, algorithm=None):
    tags = [tag for tag in config.tracking.tags
            if not re.fullmatch(
                r"cpu\d+[-_]gpu\d+|replay-(?:fullgpu|cpu|auto|hybrid-c\d+-g\d+)|"
                r"amp-(?:fp16|off)|torch-compile|compile-off|fused-adam|fused-off|"
                r"(?:masac-|matd3-)?actor-(?:shared|independent)|"
                r"(?:masac-|matd3-)?critic-(?:shared|independent)",
                tag, re.I)]
    runtime = config.runtime
    acceleration = [
        "amp-fp16" if runtime.amp_mode == "fp16" else "amp-off",
        "torch-compile" if runtime.compile_models else "compile-off",
        "fused-adam" if runtime.fused_optimizers else "fused-off",
    ]
    selected = algorithm or runtime.algorithm
    if selected == "both":
        sharing = []
        for name in ("masac", "matd3"):
            settings = getattr(config, name)
            sharing.extend([f"{name}-actor-" + ("shared" if settings.share_actor_parameters else "independent"),
                            f"{name}-critic-" + ("shared" if settings.share_critic_parameters else "independent")])
    else:
        settings = getattr(config, selected)
        sharing = ["actor-shared" if settings.share_actor_parameters else "actor-independent",
                   "critic-shared" if settings.share_critic_parameters else "critic-independent"]
    return list(dict.fromkeys([*tags, *acceleration, *sharing,
                               "replay-" + replay_layout(config).lower(),
                               f"generation{config.world.generation_version}"]))


def related_run_name(root_name, stage, suffix):
    for prefix in ("TRAIN-", "VALIDATE-", "BENCH-"):
        if root_name.startswith(prefix):
            root_name = root_name[len(prefix):]
            break
    return f"{stage}-{root_name}-{suffix}"
