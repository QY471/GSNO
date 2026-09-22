import os
import argparse
import logging
import time
import sys
import importlib
import json
import re
import copy
import random
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as tud

from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets.CAVE_Dataset import cave_dataset
from datasets.Harvard_Dataset import (
    harvard_dataset,
    prepare_data_harvard as load_harvard_arrays,
)

from tools.Utils import *


def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'


original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

if os.name == "nt":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOCAL_CAVE_ROOT = os.path.join(PROJECT_ROOT, "Cave")
LOCAL_HARVARD_ROOT = os.path.join(PROJECT_ROOT, "Harvard")
DEFAULT_NUM_WORKERS = 0 if os.name == "nt" else 8
DATASET_CLASSES = {
    "cave": cave_dataset,
    "harvard": harvard_dataset,
}


def seed_data_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def load_matching_initialization(model, checkpoint_path):
    payload = torch.load(checkpoint_path, map_location="cpu")
    data_generator_state = (
        payload.get("data_generator_state")
        if isinstance(payload, dict)
        else None
    )
    state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }
    target_state = model.state_dict()
    unexpected = [key for key in state_dict if key not in target_state]
    mismatched = [
        key for key, value in state_dict.items()
        if key in target_state and target_state[key].shape != value.shape
    ]
    if unexpected or mismatched:
        raise RuntimeError(
            "Common initialization mismatch: "
            f"unexpected={unexpected}, shape_mismatch={mismatched}"
        )
    matched = {
        key: value
        for key, value in state_dict.items()
        if key in target_state
    }
    target_state.update(matched)
    model.load_state_dict(target_state, strict=True)
    model_only = [key for key in target_state if key not in state_dict]
    return len(matched), model_only, data_generator_state

def dataset_root_candidates(dataset_name):
    dataset_root = os.environ.get("DATASET_ROOT")
    if dataset_name == "cave":
        return [
            os.environ.get("CAVE_ROOT"),
            os.path.join(dataset_root, "Cave") if dataset_root else None,
            LOCAL_CAVE_ROOT,
        ]
    if dataset_name == "harvard":
        return [
            os.environ.get("HARVARD_ROOT"),
            os.path.join(dataset_root, "Harvard") if dataset_root else None,
            LOCAL_HARVARD_ROOT,
        ]
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def choose_dataset_root(dataset_name):
    candidates = [path for path in dataset_root_candidates(dataset_name) if path]
    for root in candidates:
        if os.path.isdir(os.path.join(root, "Train")) and os.path.isdir(os.path.join(root, "Test")):
            return root
    return candidates[-1]


def infer_test_path_from_train_path(train_path):
    norm = os.path.normpath(train_path)
    if os.path.basename(norm).lower() == "train":
        return os.path.join(os.path.dirname(norm), "Test")
    return os.path.join(norm, "Test")


def resolve_data_paths(opt):
    if opt.data_path is None and opt.test_data_path is None:
        dataset_root = choose_dataset_root(opt.dataset)
        opt.data_path = os.path.join(dataset_root, "Train")
        opt.test_data_path = os.path.join(dataset_root, "Test")
    elif opt.data_path is not None and opt.test_data_path is None:
        opt.test_data_path = infer_test_path_from_train_path(opt.data_path)
    elif opt.data_path is None and opt.test_data_path is not None:
        norm = os.path.normpath(opt.test_data_path)
        if os.path.basename(norm).lower() == "test":
            opt.data_path = os.path.join(os.path.dirname(norm), "Train")
        else:
            opt.data_path = os.path.join(norm, "Train")

    opt.data_path = os.path.abspath(opt.data_path)
    opt.test_data_path = os.path.abspath(opt.test_data_path)
    return opt


def prepare_dataset_inputs(opt, split, use_cache=False):
    """Load the train or test arrays for the selected dataset protocol."""
    del use_cache
    data_path = opt.data_path if split == "train" else opt.test_data_path
    if opt.dataset == "cave":
        list_path = os.path.join(data_path, f"{split.capitalize()}.txt")
        file_list = loadpath(list_path, shuffle=(split == "train"))
        hr_hsi, hr_msi = prepare_data(data_path, file_list, len(file_list))
        return hr_hsi, hr_msi, len(file_list)
    if opt.dataset == "harvard":
        scene_count = 67 if split == "train" else 10
        hr_hsi, hr_msi = load_harvard_arrays(data_path, scene_count)
        return hr_hsi, hr_msi, scene_count
    raise ValueError(f"Unsupported dataset: {opt.dataset}")


def unpack_dataset_batch(batch):
    """Unpack one CAVE or Harvard sample."""
    if len(batch) == 3:
        return batch
    raise ValueError(f"Expected a 3-item dataset batch, got {len(batch)}")

logger = logging.getLogger("LOG")
logger.setLevel(logging.INFO)
logger.handlers.clear()

MODEL_SPECS = {
    "gsno": {
        "module": "model.gsno",
        "class": ("GSNO",),
        "default_run": "GSNO_CAVE_x4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "adci_layers": 3,
            "max_axis_ratio": opt.elliptical_max_axis_ratio,
        },
        "init_config": {
            "custom_reset": True,
            "allowed_model_only_prefixes": (
                "gaussian_refine.anisotropy_head.",
            ),
        },
    },
}

MODEL_CHOICES = ["gsno"]


def default_run_env_name(model_key):
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", model_key).upper()
    return f"GSFUSION_{sanitized}_RUN_NAME"


def build_model_bundle(opt):
    spec = MODEL_SPECS[opt.model]
    module = importlib.import_module(spec["module"])
    class_names = spec["class"]
    if isinstance(class_names, str):
        class_names = (class_names,)

    model_cls = None
    resolved_class_name = None
    for class_name in class_names:
        model_cls = getattr(module, class_name, None)
        if model_cls is not None:
            resolved_class_name = class_name
            break

    if model_cls is None:
        available_names = [name for name in dir(module) if not name.startswith("_")]
        raise AttributeError(
            f"Module '{spec['module']}' does not define any of the expected classes "
            f"{class_names}. Available public names: {available_names}"
        )

    model = model_cls(**spec["kwargs"](opt))
    model_label = f'{spec["module"]}.{resolved_class_name}'
    loss_fn = getattr(module, "compute_loss", None)
    if loss_fn is None:
        fallback_module = importlib.import_module("model.gsno")
        loss_fn = getattr(fallback_module, "compute_loss")
    return model, model_label, loss_fn


def setup_run_context(opt):
    spec = MODEL_SPECS[opt.model]
    env_keys = [
        spec.get("run_env"),
        default_run_env_name(opt.model),
        "GSFUSION_RUN_NAME",
    ]
    env_run_name = None
    for env_key in env_keys:
        if env_key:
            env_run_name = os.environ.get(env_key)
            if env_run_name:
                break
    default_run_name = f"GSNO_{opt.dataset.upper()}_x{opt.sf}"
    run_name = opt.run_name or env_run_name or default_run_name

    log_dir = os.path.abspath(opt.checkpoint_root)
    log_dir_tb = "./run/" + run_name
    ckpt_dir = os.path.join(log_dir, run_name)

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir_tb, exist_ok=True)

    logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(
        os.path.join(ckpt_dir, "training.log"), encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('=======Option used=======')

    writer = SummaryWriter(log_dir=log_dir_tb)
    return run_name, ckpt_dir, writer


def forward_tiled(test_model, lr_hsi, hr_msi, sf, tile_size, halo):
    """Run aligned HR tiles while keeping LR/HR coordinates synchronized."""
    if lr_hsi.shape[0] != 1 or hr_msi.shape[0] != 1:
        raise ValueError("Tiled evaluation currently requires batch size 1")
    height, width = hr_msi.shape[-2:]
    tile_size = int(tile_size)
    halo = int(halo)
    if tile_size <= 0:
        return test_model(lr_hsi, hr_msi, sf)
    if tile_size % sf or halo % sf:
        raise ValueError(f"eval tile size/halo must be divisible by sf={sf}")
    if height % sf or width % sf:
        raise ValueError(f"HR evaluation size {(height, width)} must be divisible by sf={sf}")

    output = None
    for y0 in range(0, height, tile_size):
        y1 = min(height, y0 + tile_size)
        for x0 in range(0, width, tile_size):
            x1 = min(width, x0 + tile_size)
            ey0 = max(0, y0 - halo)
            ex0 = max(0, x0 - halo)
            ey1 = min(height, y1 + halo)
            ex1 = min(width, x1 + halo)

            lr_tile = lr_hsi[..., ey0 // sf:ey1 // sf, ex0 // sf:ex1 // sf]
            msi_tile = hr_msi[..., ey0:ey1, ex0:ex1]
            tile_output = test_model(lr_tile, msi_tile, sf)

            if output is None:
                output = tile_output.new_empty(
                    tile_output.shape[0], tile_output.shape[1], height, width
                )
            cy0, cy1 = y0 - ey0, y1 - ey0
            cx0, cx1 = x0 - ex0, x1 - ex0
            output[..., y0:y1, x0:x1] = tile_output[..., cy0:cy1, cx0:cx1]

    return output


def evaluate(
    test_model,
    data_path=None,
    sf=4,
    dataset_name="cave",
    dataset_options=None,
):
    test_model.eval()
    if data_path is None:
        dataset_root = choose_dataset_root(dataset_name)
        data_path = os.path.join(dataset_root, "Test")
    dataset_class = DATASET_CLASSES[dataset_name]
    opt_evaluate = copy.copy(dataset_options) if dataset_options is not None else argparse.Namespace()
    opt_evaluate.dataset = dataset_name
    opt_evaluate.data_path = data_path
    opt_evaluate.test_data_path = data_path
    opt_evaluate.sizeI = None
    opt_evaluate.batch_size = 1
    opt_evaluate.sf = sf
    opt_evaluate.seed = getattr(opt_evaluate, "seed", 1)
    opt_evaluate.kernel_type = "gaussian_blur"
    opt_evaluate.eval_tile_size = getattr(opt_evaluate, "eval_tile_size", 0)
    opt_evaluate.eval_tile_halo = getattr(opt_evaluate, "eval_tile_halo", 0)
    test_HR_HSI, test_HR_MSI, test_count = prepare_dataset_inputs(
        opt_evaluate, "test", use_cache=True
    )
    opt_evaluate.testset_num = test_count
    test_dataset = dataset_class(opt_evaluate, test_HR_HSI, test_HR_MSI, istrain=False)
    loader_test = tud.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=DEFAULT_NUM_WORKERS,
        shuffle=False
    )

    psnr_total = 0.0
    sam_total = 0.0
    ergas_total = 0.0
    k = 0
    for batch in loader_test:
        LR, RGB, HR = unpack_dataset_batch(batch)
        with torch.no_grad():
            LR, RGB, HR = LR.cuda(), RGB.cuda(), HR.cuda()
            out = forward_tiled(
                test_model, LR, RGB, opt_evaluate.sf,
                opt_evaluate.eval_tile_size, opt_evaluate.eval_tile_halo,
            )

            result = out.cpu().data.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)
            HR_np = HR.cpu().data.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)

        psnr = cal_psnr(result, HR_np)
        psnr_total += psnr
        sam_total += compute_sam(result, HR_np)
        # Keep the fixed-4 convention used by the project's formal CAVE tables.
        ergas_total += compute_ergas(result, HR_np, 4)
        k += 1

    average_psnr = psnr_total / k
    evaluate.last_metrics = {
        "psnr": average_psnr,
        "sam": sam_total / k,
        "ergas_fixed4": ergas_total / k,
        "num_images": k,
    }
    return average_psnr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Code for HSI Fusion")
    parser.add_argument('--data_path', default=None, type=str,
                        help='Path of the training data')
    parser.add_argument("--sizeI", default=64, type=int, help='The image size of the training patches')
    parser.add_argument("--batch_size", default=32, type=int, help='Batch size')
    parser.add_argument("--ngpus", default=1, type=int,
                        help="Number of visible GPUs used by DataParallel")
    parser.add_argument("--trainset_num", default=20000, type=int, help='The number of training samples of each epoch')
    parser.add_argument("--sf", default=4, type=int, help='Scaling factor')
    parser.add_argument("--seed", default=1, type=int, help='Random seed')
    parser.add_argument("--kernel_type", default='gaussian_blur', type=str, help='Kernel type')
    parser.add_argument('--test_data_path', default=None, type=str, help='Path of the testing data')
    parser.add_argument("--dataset", default="cave", choices=sorted(DATASET_CLASSES.keys()),
                        help='Dataset loader to use')
    parser.add_argument("--eval_tile_size", default=0, type=int,
                        help="Aligned HR tile size for evaluation; 0 keeps full-image inference")
    parser.add_argument("--eval_tile_halo", default=0, type=int,
                        help="Context halo around each evaluation tile")

    parser.add_argument("--ep_total", default=1000, type=int, help='Total epochs')
    parser.add_argument("--e_every", default=5, type=int, help='Evaluation interval')
    parser.add_argument("--lr", default=4e-4, type=float, help='Initial learning rate')
    parser.add_argument("--sam_weight", default=0.1, type=float, help='SAM loss weight')
    parser.add_argument("--sam_warmup_epochs", default=5, type=int, help='SAM warmup epochs')
    parser.add_argument("--grad_clip", default=0.0, type=float, help='Max grad norm; 0 disables clipping')
    parser.add_argument("--grad_accum_steps", default=1, type=int,
                        help="Number of micro-batches per optimizer update")
    parser.add_argument("--optimizer", default="adam", choices=["adam", "adamw"],
                        help="Optimizer used for training")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Optimizer weight decay")
    parser.add_argument("--scheduler", default="cosine", choices=["cosine", "cosine_iter", "constant", "sgdr", "step", "multistep", "fixed_multistep"],
                        help="Learning-rate scheduler")
    parser.add_argument("--cosine_tmax_epochs", default=0, type=int,
                        help="Cosine T_max in epochs; 0 uses --ep_total")
    parser.add_argument("--cosine_eta_min", default=1e-6, type=float,
                        help="Minimum learning rate for cosine schedulers")
    parser.add_argument("--lr_milestones", default="", type=str,
                        help="Comma-separated epoch milestones for --scheduler multistep")
    parser.add_argument("--sgdr_t0_epochs", default=400, type=int,
                        help="Epochs in the first SGDR cosine cycle")
    parser.add_argument("--lr_step_size", default=5, type=int,
                        help="Epoch interval for --scheduler step")
    parser.add_argument("--lr_gamma", default=0.95, type=float,
                        help="Multiplicative decay for --scheduler step")
    parser.add_argument("--scheduler_stop", default=200, type=int,
                        help="Exclusive final milestone for the fixed multistep scheduler")

    parser.add_argument(
                        "--model",
                        default="gsno",
                        choices=MODEL_CHOICES,
                        help='GSNO paper model')
    parser.add_argument("--run_name", default=None, type=str,
                        help='Optional experiment name for logs/checkpoints')
    parser.add_argument(
        "--checkpoint_root",
        default="Checkpoint",
        type=str,
        help="Root directory containing per-experiment checkpoint folders",
    )
    parser.add_argument("--dim", default=80, type=int)
    parser.add_argument(
        "--elliptical_max_axis_ratio", default=2.0, type=float,
        help="Maximum principal-axis std ratio for constrained elliptical Gaussian",
    )
    parser.add_argument("--common_init_checkpoint", default="", type=str,
                        help="Load every matching parameter from a shared initialization checkpoint")
    parser.add_argument(
        "--initialization_mode",
        default="auto",
        choices=["auto", "from_scratch", "checkpoint"],
        help=(
            "Explicit initialization provenance. 'from_scratch' rejects any "
            "common checkpoint; 'checkpoint' requires one; 'auto' preserves "
            "the legacy path-based behavior."
        ),
    )
    parser.add_argument(
        "--save_initial_state",
        default=0,
        type=int,
        help="Save the exact post-initialization model state in the run directory",
    )
    parser.add_argument("--export_init_checkpoint", default="", type=str,
                        help="Save the initialized model state and exit before loading data")
    parser.add_argument("--max_run_epochs", default=0, type=int,
                        help="Run at most this many epochs while keeping scheduler T_max=ep_total; 0 runs all")
    parser.add_argument("--checkpoint_every", default=0, type=int,
                        help="Save checkpoint_epoch_NNNN.pth every N completed epochs; 0 disables")
    parser.add_argument("--num_bands", default=31, type=int)
    parser.add_argument("--num_msi", default=3, type=int)
    opt = parser.parse_args()
    opt = resolve_data_paths(opt)

    if opt.initialization_mode == "auto":
        opt.effective_initialization_mode = (
            "checkpoint" if opt.common_init_checkpoint else "from_scratch"
        )
    elif opt.initialization_mode == "from_scratch":
        if opt.common_init_checkpoint:
            parser.error(
                "--initialization_mode from_scratch cannot be combined with "
                "--common_init_checkpoint"
            )
        opt.effective_initialization_mode = "from_scratch"
    else:
        if not opt.common_init_checkpoint:
            parser.error(
                "--initialization_mode checkpoint requires "
                "--common_init_checkpoint"
            )
        opt.effective_initialization_mode = "checkpoint"

    run_name, ckpt_dir, writer = setup_run_context(opt)
    with open(os.path.join(ckpt_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(vars(opt), handle, indent=2, default=str)
    dataset_class = DATASET_CLASSES[opt.dataset]

    print("Random Seed: ", opt.seed)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    print(opt)
    print(f"Run name: {run_name}")
    logger.info(str(opt))

    print("===> New Model")
    model, model_name, compute_loss_fn = build_model_bundle(opt)
    print(f"Model: {model_name}")

    print("===> Setting GPU")
    model = dataparallel(model, opt.ngpus)

    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f'[INFO] {model_name} #parameters: {num_params / 1e6:.3f} M')
    logger.info(f'{model_name} #parameters: {num_params / 1e6:.3f} M')

    pre_init_config = MODEL_SPECS[opt.model].get("init_config", {})
    if not pre_init_config.get("preserve_default_init", False):
        for layer in model.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None and pre_init_config.get("zero_conv_bias", True):
                    nn.init.zeros_(layer.bias)
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None and pre_init_config.get("zero_conv_bias", True):
                    nn.init.zeros_(layer.bias)
    else:
        print("[INIT] preserved model-native parameter initialization")

    model_ref = model.module if hasattr(model, "module") else model
    spec = MODEL_SPECS[opt.model]
    init_config = spec.get("init_config", {})

    model_ref.reset_custom_init()


    common_data_generator_state = None
    print(
        "[INIT] provenance "
        f"mode={opt.effective_initialization_mode} "
        f"checkpoint={opt.common_init_checkpoint or '<none>'}"
    )
    if opt.common_init_checkpoint:
        matched_count, model_only_keys, common_data_generator_state = (
            load_matching_initialization(model_ref, opt.common_init_checkpoint)
        )
        allowed_model_only_prefixes = tuple(
            init_config.get(
                "allowed_model_only_prefixes",
                (),
            )
        )
        invalid_model_only = [
            key for key in model_only_keys
            if not key.startswith(allowed_model_only_prefixes)
        ]
        if invalid_model_only:
            raise RuntimeError(
                "Shared initialization omitted non-Gaussian parameters: "
                f"{invalid_model_only}"
            )
        print(
            "[INIT] loaded shared initialization "
            f"matched={matched_count} gaussian_only={len(model_only_keys)} "
            f"path={opt.common_init_checkpoint}"
        )

    if opt.save_initial_state:
        initial_state_path = os.path.join(ckpt_dir, "initial_state.pth")
        if common_data_generator_state is None:
            initial_data_generator = torch.Generator()
            initial_data_generator.manual_seed(opt.seed)
            saved_data_generator_state = initial_data_generator.get_state()
        else:
            saved_data_generator_state = common_data_generator_state
        torch.save(
            {
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in model_ref.state_dict().items()
                },
                "data_generator_state": saved_data_generator_state,
                "meta": {
                    "model": opt.model,
                    "seed": opt.seed,
                    "initialization_mode": opt.effective_initialization_mode,
                    "common_init_checkpoint": opt.common_init_checkpoint,
                },
            },
            initial_state_path,
        )
        print(f"[INIT] saved exact initial state to {initial_state_path}")

    if opt.export_init_checkpoint:
        export_path = os.path.abspath(opt.export_init_checkpoint)
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        cpu_state = {
            key: value.detach().cpu()
            for key, value in model_ref.state_dict().items()
        }
        torch.save(
            {
                "state_dict": cpu_state,
                "data_generator_state": torch.get_rng_state(),
                "meta": {
                    "model": opt.model,
                    "seed": opt.seed,
                    "dim": opt.dim,
                },
            },
            export_path,
        )
        print(f"[INIT] exported initialized state to {export_path}")
        writer.close()
        sys.exit(0)

    HR_HSI, HR_MSI, train_scene_count = prepare_dataset_inputs(opt, "train")
    print(
        f"[DATA] dataset={opt.dataset} train_scenes={train_scene_count} "
        f"train_path={opt.data_path} test_path={opt.test_data_path}"
    )

    initial_epoch = findLastCheckpoint(save_dir=ckpt_dir)
    if initial_epoch > 0:
        print('resuming by loading epoch %04d' % initial_epoch)
        resume_path = os.path.join(ckpt_dir, 'model_%04d.pth' % initial_epoch)
        ckpt = torch.load(resume_path, map_location='cpu')
        if isinstance(ckpt, dict):
            state_dict = ckpt.get('state_dict', ckpt)
            model.load_state_dict(state_dict, strict=False)
        else:
            model = ckpt

    bestpsnr = 0.0
    best_epoch = 0
    ep_total = opt.ep_total
    e_every = opt.e_every
    lr = opt.lr

    if opt.optimizer == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=opt.weight_decay,
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=opt.weight_decay,
        )
    if opt.grad_accum_steps < 1:
        parser.error("--grad_accum_steps must be >= 1")
    nominal_micro_steps = (opt.trainset_num + opt.batch_size - 1) // opt.batch_size
    nominal_optimizer_steps = (
        nominal_micro_steps + opt.grad_accum_steps - 1
    ) // opt.grad_accum_steps
    if opt.scheduler == "cosine":
        cosine_tmax = opt.cosine_tmax_epochs or ep_total
        scheduler = CosineAnnealingLR(
            optimizer, T_max=cosine_tmax, eta_min=opt.cosine_eta_min
        )
    elif opt.scheduler == "cosine_iter":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=ep_total * nominal_optimizer_steps,
            eta_min=opt.cosine_eta_min,
        )
    elif opt.scheduler == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _epoch: 1.0)
    elif opt.scheduler == "sgdr":
        nominal_steps_per_epoch = (opt.trainset_num + opt.batch_size - 1) // opt.batch_size
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=opt.sgdr_t0_epochs * nominal_steps_per_epoch,
            T_mult=1,
            eta_min=1e-6,
        )
        print(
            f"[SCHEDULER] SGDR T_0={opt.sgdr_t0_epochs} epochs "
            f"= {opt.sgdr_t0_epochs * nominal_steps_per_epoch} iterations, "
            "T_mult=1, eta_min=1e-6"
        )
    elif opt.scheduler == "fixed_multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(range(1, opt.scheduler_stop, opt.lr_step_size)),
            gamma=opt.lr_gamma,
        )
    elif opt.scheduler == "multistep":
        milestones = [
            int(value) for value in opt.lr_milestones.split(",") if value.strip()
        ]
        if not milestones:
            parser.error("--scheduler multistep requires --lr_milestones")
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=opt.lr_gamma
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=opt.lr_step_size, gamma=opt.lr_gamma
        )

    train_loader_generator = torch.Generator()
    if common_data_generator_state is not None:
        train_loader_generator.set_state(common_data_generator_state)
        print("[DATA] restored training generator state from shared initialization")
    else:
        train_loader_generator.manual_seed(opt.seed)
        print(f"[DATA] initialized independent training generator with seed={opt.seed}")

    final_epoch = ep_total
    if opt.max_run_epochs > 0:
        final_epoch = min(ep_total, initial_epoch + opt.max_run_epochs)
        print(
            f"[RUN] limiting this invocation to epochs "
            f"[{initial_epoch}, {final_epoch}) while scheduler T_max remains {ep_total}"
        )

    for epoch in range(initial_epoch, final_epoch):
        model.train()

        train_sf = opt.sf
        epoch_opt = copy.copy(opt)
        epoch_opt.sf = train_sf
        dataset = dataset_class(epoch_opt, HR_HSI, HR_MSI)
        loader_train = tud.DataLoader(
            dataset,
            num_workers=DEFAULT_NUM_WORKERS,
            batch_size=opt.batch_size,
            shuffle=True,
            generator=train_loader_generator,
            worker_init_fn=seed_data_worker,
        )
        steps_per_epoch = len(loader_train)
        par = tqdm(loader_train, desc=f'Training Ep{epoch}', unit='batch', ascii=True)

        epoch_loss = 0.0
        start_time = time.time()
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch_item in enumerate(par):
            LR, RGB, HR = unpack_dataset_batch(batch_item)

            LR, RGB, HR = Variable(LR), Variable(RGB), Variable(HR)
            LR, RGB, HR = LR.cuda(), RGB.cuda(), HR.cuda()

            out = model(LR, RGB, train_sf)
            loss = compute_loss_fn(
                out, HR, epoch,
                sam_warmup_epochs=opt.sam_warmup_epochs,
                sam_weight=opt.sam_weight,
            )

            epoch_loss += loss.item()

            (loss / opt.grad_accum_steps).backward()
            should_step = (
                (batch_index + 1) % opt.grad_accum_steps == 0
                or batch_index + 1 == steps_per_epoch
            )
            if should_step:
                if opt.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if opt.scheduler in {"sgdr", "cosine_iter"}:
                    scheduler.step()

        if opt.scheduler in {"sgdr", "cosine_iter"}:
            # SGDR advances once per optimizer update to make T_0 an exact
            # 400-epoch iteration budget.
            pass
        elif opt.scheduler == "fixed_multistep":
            # Step the fixed milestone schedule at the requested epoch.
            scheduler.step(epoch)
        else:
            scheduler.step()

        epoch_loss_avg = epoch_loss / max(1, steps_per_epoch)
        writer.add_scalar('Loss/train/epoch', epoch_loss_avg, epoch)

        if epoch % e_every == 0:
            ave = evaluate(
                model, data_path=opt.test_data_path, sf=opt.sf,
                dataset_name=opt.dataset, dataset_options=opt,
            )
            eval_metrics = getattr(evaluate, "last_metrics", None)
            if ave > bestpsnr:
                bestpsnr = ave
                best_epoch = epoch

                best_model_path = os.path.join(ckpt_dir, 'best_model.pth')
                if os.path.exists(best_model_path):
                    os.remove(best_model_path)
                torch.save(model.state_dict(), best_model_path)

            logger.info(
                'Epoch: {}/{} average psnr: {:.7f} bestpsnr: {:.7f}, bestepoch: {}'.format(
                    epoch, ep_total - 1, ave, bestpsnr, best_epoch
                )
            )
            writer.add_scalar('PSNR/test', ave, epoch)
            if eval_metrics:
                logger.info(
                    'Eval metrics epoch {}: PSNR {:.7f}, SAM {:.7f}, ERGAS4 {:.7f}, images {}'.format(
                        epoch,
                        eval_metrics["psnr"],
                        eval_metrics["sam"],
                        eval_metrics["ergas_fixed4"],
                        eval_metrics["num_images"],
                    )
                )
                writer.add_scalar('SAM/test', eval_metrics["sam"], epoch)
                writer.add_scalar('ERGAS4/test', eval_metrics["ergas_fixed4"], epoch)
            model_for_stats = model.module if hasattr(model, "module") else model
            if hasattr(model_for_stats, "collect_gs_stats"):
                try:
                    gs_stats = model_for_stats.collect_gs_stats()
                except Exception as exc:
                    gs_stats = None
                    logger.warning("GS stats epoch {} failed: {}".format(epoch, repr(exc)))
                    print("GS stats epoch {} failed: {}".format(epoch, repr(exc)))
                if gs_stats:
                    stat_items = []
                    if isinstance(gs_stats, dict):
                        stat_sources = [("", gs_stats)]
                    elif isinstance(gs_stats, (list, tuple)):
                        stat_sources = []
                        for stat_idx, layer_stats in enumerate(gs_stats):
                            if not isinstance(layer_stats, dict):
                                stat_items.append(f"layer{stat_idx}: {layer_stats}")
                                continue
                            layer_name = layer_stats.get("layer", stat_idx)
                            stat_sources.append((f"layer{layer_name}/", layer_stats))
                    else:
                        stat_sources = []
                        stat_items.append(str(gs_stats))
                    for stat_prefix, stat_dict in stat_sources:
                        for stat_key, stat_value in stat_dict.items():
                            if stat_key == "layer":
                                continue
                            full_stat_key = f"{stat_prefix}{stat_key}"
                            try:
                                stat_float = float(stat_value)
                            except (TypeError, ValueError):
                                stat_items.append(f"{full_stat_key}: {stat_value}")
                                continue
                            stat_items.append(f"{full_stat_key}: {stat_float:.6f}")
                            writer.add_scalar(f"GSStats/{full_stat_key}", stat_float, epoch)
                    stat_msg = ", ".join(stat_items)
                    logger.info("GS stats epoch {}: {}".format(epoch, stat_msg))
                    print("GS stats epoch {}: {}".format(epoch, stat_msg))
            print(
                'Epoch: {}/{} average psnr: {:.7f} bestpsnr: {:.7f}, bestepoch: {}'.format(
                    epoch, ep_total - 1, ave, bestpsnr, best_epoch
                )
            )

        completed_epochs = epoch + 1
        if opt.checkpoint_every > 0 and (
            completed_epochs % opt.checkpoint_every == 0 or completed_epochs == ep_total
        ):
            periodic_path = os.path.join(
                ckpt_dir, "checkpoint_epoch_{:04d}.pth".format(completed_epochs)
            )
            torch.save(model.state_dict(), periodic_path)
            logger.info(
                "Saved periodic checkpoint after {} epochs: {}".format(
                    completed_epochs, periodic_path
                )
            )

        elapsed_time = time.time() - start_time
        logger.info(
            'Train epoch {}: loss {:.7f}, lr {:.8e}, time {:.2f}s'.format(
                epoch,
                epoch_loss_avg,
                optimizer.param_groups[0]["lr"],
                elapsed_time,
            )
        )
        writer.add_scalar('LR/train', optimizer.param_groups[0]["lr"], epoch)
        print(
            f'Epoch: {epoch}/{ep_total - 1} '
            f'train_sf: {train_sf} '
            f'loss: {epoch_loss_avg:.6f} '
            f'lr: {optimizer.param_groups[0]["lr"]:.2e} '
            f'time: {elapsed_time:.2f}s'
        )

    print('Epoch: {} best PSNR: {:.7f}'.format(best_epoch, bestpsnr))
    logger.info('Epoch: {} best PSNR: {:.7f}'.format(best_epoch, bestpsnr))
    writer.close()
