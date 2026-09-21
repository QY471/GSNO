"""AFNO-compatible Harvard entrypoint for the runnable GSFusion models.

The data behavior and optimization defaults follow the senior AFNO project's
``Train_Harvard.py``. Model construction, checkpointing and metrics remain in
the shared ``Train_Cave.py`` trainer so E3, E6 and ADCI-NoGS use one training
implementation.

No dataset path is hard-coded. Pass ``--data_root`` or set ``HARVARD_ROOT``.
"""

import os
import runpy
import sys
from pathlib import Path


MODEL_ALIASES = {
    "gsno": "e3_constrained_elliptical_gaussian",
    "e3": "e3_constrained_elliptical_gaussian",
}


def translate_model_aliases(argv):
    translated = list(argv)
    for index, value in enumerate(translated):
        if value == "--model" and index + 1 < len(translated):
            raw_model = translated[index + 1]
            translated[index + 1] = MODEL_ALIASES.get(
                raw_model.lower(), raw_model
            )
        elif value.startswith("--model="):
            raw_model = value.split("=", 1)[1]
            translated[index] = "--model=" + MODEL_ALIASES.get(
                raw_model.lower(), raw_model
            )
    return translated


def get_arg_value(argv, name, default=None):
    prefix = name + "="
    for index, value in enumerate(argv):
        if value == name and index + 1 < len(argv):
            return argv[index + 1]
        if value.startswith(prefix):
            return value.split("=", 1)[1]
    return default


def has_arg(argv, name):
    prefix = name + "="
    return any(value == name or value.startswith(prefix) for value in argv)


def pop_arg_value(argv, name, default=None):
    prefix = name + "="
    cleaned = []
    found = default
    skip_next = False
    for index, value in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if value == name and index + 1 < len(argv):
            found = argv[index + 1]
            skip_next = True
        elif value.startswith(prefix):
            found = value.split("=", 1)[1]
        else:
            cleaned.append(value)
    return found, cleaned


def main():
    user_args = translate_model_aliases(sys.argv[1:])
    harvard_root, user_args = pop_arg_value(
        user_args, "--data_root", os.environ.get("HARVARD_ROOT")
    )

    if not harvard_root and not any(
        value in {"-h", "--help"} for value in user_args
    ):
        raise SystemExit(
            "Harvard data root is not configured. Pass --data_root PATH "
            "or set HARVARD_ROOT."
        )

    model_name = get_arg_value(
        user_args,
        "--model",
        "e3_constrained_elliptical_gaussian",
    )
    scale = get_arg_value(user_args, "--sf", "4")

    defaults = [
        "--dataset", "harvard",
        "--model", "e3_constrained_elliptical_gaussian",
        "--checkpoint_root", "Checkpoint_Harvard",
        "--sizeI", "64",
        "--batch_size", "32",
        "--ngpus", "1",
        "--trainset_num", "20000",
        "--sf", "4",
        "--seed", "1",
        "--kernel_type", "gaussian_blur",
        "--ep_total", "500",
        "--e_every", "5",
        "--lr", "0.0004",
        "--sam_weight", "0.01",
        "--sam_warmup_epochs", "5",
        "--scheduler", "cosine",
        "--lr_step_size", "5",
        "--lr_gamma", "0.95",
        "--afno_scheduler_stop", "200",
        "--eval_tile_size", "0",
        "--eval_tile_halo", "0",
        "--dim", "64",
        "--initialization_mode", "from_scratch",
        "--save_initial_state", "1",
    ]

    if harvard_root:
        root = os.path.abspath(os.path.expanduser(harvard_root))
        defaults.extend(
            [
                "--data_path", os.path.join(root, "Train"),
                "--test_data_path", os.path.join(root, "Test"),
            ]
        )

    if not has_arg(user_args, "--run_name"):
        defaults.extend(
            [
                "--run_name",
                f"Harvard_{model_name}_SF{scale}_500EP_S1_FROM_SCRATCH",
            ]
        )

    train_script = Path(__file__).resolve().with_name("Train_Cave.py")
    sys.argv = [str(train_script)] + defaults + user_args
    runpy.run_path(str(train_script), run_name="__main__")


if __name__ == "__main__":
    main()
