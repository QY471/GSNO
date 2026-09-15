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
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets.CAVE_Dataset import cave_dataset
from datasets.Harvard_Dataset import (
    harvard_dataset,
    prepare_data_harvard as load_harvard_arrays,
)
from datasets.Chikusei_AFNO_Dataset import ChikuseiAFNODataset
from datasets.RemoteHSIMSI_Dataset import RemoteHSIMSIDataset

from tools.Utils import *
from tools.SSIM import *


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
LOCAL_CHIKUSEI_ROOT = os.path.join(PROJECT_ROOT, "Chikusei_AFNO")
DEFAULT_NUM_WORKERS = 0 if os.name == "nt" else 8
DATASET_CLASSES = {
    "cave": cave_dataset,
    "harvard": harvard_dataset,
    "chikusei": ChikuseiAFNODataset,
    "remote_hsi_msi": RemoteHSIMSIDataset,
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

MODEL_ALIASES = {
    "gsno_nogs": "gsno_nogs_identity",
    # Kept so frozen E6 configs and older commands continue to resolve.
    "e6_msi_guided_hsi_routing": "e6_msi_routed_gaussian",
}


def normalize_model_name(model_name):
    return MODEL_ALIASES.get(model_name.lower(), model_name)


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
    if dataset_name == "chikusei":
        return [
            os.environ.get("CHIKUSEI_ROOT"),
            os.path.join(dataset_root, "Chikusei_AFNO") if dataset_root else None,
            LOCAL_CHIKUSEI_ROOT,
        ]
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def choose_dataset_root(dataset_name):
    candidates = [path for path in dataset_root_candidates(dataset_name) if path]
    for root in candidates:
        if dataset_name == "chikusei":
            if (
                os.path.isfile(os.path.join(root, "train_chikusei_gt_rgb.h5"))
                and os.path.isfile(os.path.join(root, "test_chikusei_gt_rgb.h5"))
            ):
                return root
            continue
        if os.path.isdir(os.path.join(root, "Train")) and os.path.isdir(os.path.join(root, "Test")):
            return root
    return candidates[-1]


def infer_test_path_from_train_path(train_path):
    norm = os.path.normpath(train_path)
    if os.path.basename(norm).lower() == "train":
        return os.path.join(os.path.dirname(norm), "Test")
    return os.path.join(norm, "Test")


def resolve_data_paths(opt):
    if opt.dataset in {"chikusei", "remote_hsi_msi"}:
        if opt.data_path is None or opt.test_data_path is None:
            if opt.dataset == "remote_hsi_msi":
                raise ValueError(
                    "remote_hsi_msi requires explicit --data_path and --test_data_path"
                )
            dataset_root = choose_dataset_root(opt.dataset)
            opt.data_path = opt.data_path or os.path.join(
                dataset_root, "train_chikusei_gt_rgb.h5"
            )
            opt.test_data_path = opt.test_data_path or os.path.join(
                dataset_root, "test_chikusei_gt_rgb.h5"
            )
        opt.data_path = os.path.abspath(opt.data_path)
        opt.test_data_path = os.path.abspath(opt.test_data_path)
        return opt
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
    if opt.dataset in {"chikusei", "remote_hsi_msi"}:
        import h5py

        with h5py.File(data_path, "r") as handle:
            sample_count = int(handle["GT"].shape[0])
        return None, None, sample_count
    raise ValueError(f"Unsupported dataset: {opt.dataset}")


def unpack_dataset_batch(batch):
    """Accept the project's 3-item batches and AFNO's 4-item batches."""
    if len(batch) == 3:
        return batch
    if len(batch) == 4:
        lr_hsi, hr_msi, hr_hsi, _coord = batch
        return lr_hsi, hr_msi, hr_hsi
    raise ValueError(f"Expected a 3- or 4-item dataset batch, got {len(batch)}")

logger = logging.getLogger("LOG")
logger.setLevel(logging.INFO)
logger.handlers.clear()

MODEL_SPECS = {
    "afno_zhujunwei_chikusei_unified": {
        "module": "model.baselines.GSFusion_AFNO_ZhuJunweiChikuseiUnified",
        "class": ("GSFusion",),
        "default_run": "AFNO_ZhuJunwei_Chikusei_Unified_SF4",
        "kwargs": lambda opt: {},
        "init_config": {
            "custom_reset": False,
            "zero_conv_bias": False,
        },
    },
    "dpformer_official_unified": {
        "module": "model.baselines.DPFormer_OfficialUnified",
        "class": ("GSFusion",),
        "default_run": "DPFormer_PR2026_OfficialUnified_CAVE4",
        "kwargs": lambda opt: {
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "training_scale": opt.sf,
        },
        # Released trainer applies Xavier to Conv/ConvTranspose layers.
        "init_config": {
            "custom_reset": False,
            "zero_conv_bias": False,
        },
    },
    "bhsrnet_official_unified": {
        "module": "model.baselines.BHSRNet_OfficialUnified",
        "class": ("GSFusion",),
        "default_run": "BHSRNet_CVPR2026_OfficialUnified_CAVE4",
        "kwargs": lambda opt: {
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "training_scale": opt.sf,
        },
        "init_config": {
            "custom_reset": False,
            "zero_conv_bias": False,
            "preserve_default_init": True,
        },
    },
    "clsnet_official_unified": {
        "module": "model.baselines.CLSNet_OfficialUnified",
        "class": ("GSFusion",),
        "default_run": "CLSNet_IF2026_OfficialUnified_CAVE4",
        "kwargs": lambda opt: {
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "training_scale": opt.sf,
        },
        "init_config": {
            "custom_reset": True,
            "zero_conv_bias": False,
            "preserve_default_init": True,
        },
    },
    "emrdiff_official_exact_unified": {
        "module": "model.baselines.EMRDiff_OfficialExactUnified",
        "class": ("GSFusion",),
        "default_run": "EMRDiff_CVPR2026_OfficialExactUnified_CAVE4",
        "kwargs": lambda opt: {
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
        },
        "init_config": {
            "custom_reset": False,
            "zero_conv_bias": False,
            "preserve_default_init": True,
        },
    },
    "emrdiff_official_unified": {
        "module": "model.baselines.EMRDiff_OfficialUnified",
        "class": ("GSFusion",),
        "default_run": "EMRDiff_CVPR2026_OfficialUnified_CAVE4",
        "kwargs": lambda opt: {
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
        },
        "init_config": {
            "custom_reset": False,
            "zero_conv_bias": False,
            "preserve_default_init": True,
        },
    },
    "dcinn_official_unified": {
        "module": "model.baselines.DCINN_OfficialUnified",
        "class": ("GSFusion",),
        "default_run": "DCINN_IJCV2024_OfficialUnified_CAVE4",
        "kwargs": lambda opt: {
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "dataset": opt.dataset,
            "calibration_data_path": opt.data_path,
        },
        "init_config": {
            "custom_reset": False,
            "zero_conv_bias": False,
            "preserve_default_init": True,
        },
    },
    "srlfnet_official_unified": {
        "module": "model.baselines.SRLFNet_OfficialUnified",
        "class": ("GSFusion",),
        "default_run": "SRLFNet_CVPR2025_OfficialUnified_CAVE4",
        "kwargs": lambda opt: {
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "dataset": opt.dataset,
            "calibration_data_path": opt.data_path,
            "training_scale": opt.sf,
        },
        "init_config": {
            "custom_reset": True,
            "zero_conv_bias": False,
            "preserve_default_init": True,
        },
    },
    "ramoe_official_unified": {
        "module": "model.baselines.RAMoE_OfficialUnified",
        "class": ("GSFusion",),
        "default_run": "RAMoE_TGRS2026_OfficialUnified_CAVE4",
        "kwargs": lambda opt: {
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
        },
        "init_config": {
            "custom_reset": False,
            "zero_conv_bias": False,
            "preserve_default_init": True,
        },
    },
    "feinfn_official_unified": {
        "module": "model.baselines.FeINFN_OfficialUnified",
        "class": ("GSFusion",),
        "default_run": "FeINFN_NeurIPS2024_OfficialUnified_CAVE4",
        "kwargs": lambda opt: {
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
        },
        "init_config": {
            "custom_reset": False,
            "zero_conv_bias": False,
            "preserve_default_init": True,
        },
    },
    "lrtn_official_unified": {
        "module": "model.baselines.LRTN_OfficialUnified",
        "class": ("GSFusion",),
        "default_run": "LRTN_IJCV2025_OfficialUnified_CAVE4",
        "kwargs": lambda opt: {
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
        },
        "init_config": {
            "custom_reset": True,
            "zero_conv_bias": False,
            "preserve_default_init": True,
        },
    },
    "pstun_official_unified": {
        "module": "model.baselines.PSTUN_OfficialUnified",
        "class": ("GSFusion",),
        "default_run": "PSTUN_IF2025_OfficialUnified_CAVE4",
        "kwargs": lambda opt: {
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
        },
        "init_config": {
            "custom_reset": False,
            "zero_conv_bias": False,
            "preserve_default_init": True,
        },
    },
    "otias_official_unified": {
        "module": "model.baselines.OTIAS_OfficialUnified",
        "class": ("GSFusion",),
        "default_run": "OTIAS_OfficialUnified_CAVE4",
        "kwargs": lambda opt: {
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "dataset": opt.dataset,
        },
        # Preserve the released PyTorch constructor initialization.  The
        # formal run is from scratch and never loads the public checkpoint.
        "init_config": {
            "custom_reset": False,
            "zero_conv_bias": False,
            "preserve_default_init": True,
        },
    },
    "dspnet_official_unified": {
        "module": "model.baselines.DSPNet_OfficialUnified",
        "class": ("GSFusion",),
        "default_run": "DSPNet_OfficialUnified_CAVE4",
        "kwargs": lambda opt: {
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
        },
        # Match the released trainer: Xavier Conv/ConvTranspose weights while
        # retaining constructor-created biases.
        "init_config": {
            "custom_reset": False,
            "zero_conv_bias": False,
        },
    },
    "dspnet_official_scale_specific_unified": {
        "module": "model.baselines.DSPNet_OfficialScaleSpecificUnified",
        "class": ("GSFusion",),
        "default_run": "DSPNet_OfficialScaleSpecificUnified",
        "kwargs": lambda opt: {
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "training_scale": opt.sf,
        },
        # Same Xavier initialization as the released DSPNet trainer.
        "init_config": {
            "custom_reset": False,
            "zero_conv_bias": False,
        },
    },
    "mimo_sst_official_unified": {
        "module": "model.baselines.MIMO_SST_OfficialUnified",
        "class": ("GSFusion",),
        "default_run": "MIMO_SST_OfficialUnified_CAVE4",
        "kwargs": lambda opt: {
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
        },
        # Preserve ordinary PyTorch initialization from the released model.
        "init_config": {
            "custom_reset": False,
            "zero_conv_bias": False,
            "preserve_default_init": True,
        },
    },
    "afno_zhujunwei_unified": {
        "module": "model.baselines.GSFusion_AFNO_ZhuJunweiUnified",
        "class": ("GSFusion",),
        "default_run": "AFNO_ZhuJunwei_Unified_CAVE4",
        "kwargs": lambda opt: {},
        "init_config": {
            "custom_reset": False,
            "zero_conv_bias": False,
        },
    },
    "gsno_aniso_structure": {
        "module": "model.archive.legacy.GSFusion_GSNO_AnisoStructure",
        "class": ("GSFusion",),
        "default_run": "GSFusion_GSNO_AnisoStructure_CAVE_4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "num_basis": opt.num_basis,
            "num_gs_layers": opt.num_gs_layers,
            "edsr_resblocks": opt.edsr_resblocks,
        },
        "init_config": {"zero_init_decoder_last": False},
    },
    "gsno_strong_b1_ffn_only": {
        "module": "model.archive.legacy.GSFusion_GSNO_StrongRefineAblation",
        "class": ("GSFusion",),
        "default_run": "GSNO_Strong_B1_FFNOnly_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "num_gs_layers": opt.num_gs_layers,
            "refine_mode": "ffn_only",
        },
        "init_config": {"zero_init_decoder_last": False},
    },
    "hr_fused_adaptive_gaussian_residual": {
        "module": "model.geometry.GSFusion_HRFused_AdaptiveGaussianResidual",
        "class": ("GSFusion",),
        "default_run": "GSFusion_HRFused_AdaptiveGaussianResidual_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "adci_layers": 3,
        },
        "init_config": {"custom_reset": True},
    },
    "edsr_circular_gaussian_residual": {
        "module": "model.baselines.GSFusion_EDSRBackbone_CircularGaussianResidual",
        "class": ("GSFusion",),
        "default_run": "GSFusion_EDSRBackbone_CircularNorm3Sigma_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "n_resblocks": 6,
        },
        "init_config": {"custom_reset": True},
    },
    "edsr_primitive_embedding_gaussian_residual": {
        "module": "model.baselines.GSFusion_EDSRBackbone_PrimitiveEmbedding_CircularGaussianResidual",
        "class": ("GSFusion",),
        "default_run": "GSFusion_EDSRBackbone_PrimitiveEmbedding_CircularNorm3Sigma_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "n_resblocks": 6,
        },
        "init_config": {"custom_reset": True},
    },
    "edsr_nogaussian": {
        "module": "model.baselines.GSFusion_EDSRBackbone_NoGaussian",
        "class": ("GSFusion",),
        "default_run": "GSFusion_EDSRBackbone_NoGaussian_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "n_resblocks": 6,
        },
        "init_config": {"custom_reset": True},
    },
    "gsno_cell_gaussian_difference": {
        "module": "model.archive.legacy.GSFusion_GSNO_CellGaussianDifference",
        "class": ("GSFusion",),
        "default_run": "GSNO_DSSGR_CellGaussianDifference_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "adci_layers": 3,
            "ffn_layers": 3,
            "canonical_scale": 4.0,
            "num_gs_layers": opt.num_gs_layers,
        },
        "init_config": {
            "zero_init_decoder_last": False,
            "legacy_gsno_init": True,
            "map_old_ffn": True,
        },
    },
    "baseline": {
        "module": "model.baselines.GSFusion_Baseline",
        "class": ("GsFusion",),
        "default_run": "GSFusion_Baseline_CAVE_4",
        "kwargs": lambda opt: {"dim": opt.dim, "num_gs_layers": opt.num_gs_layers},
        "init_config": {"zero_init_decoder_last": True},
    },
    "gsno": {
        "module": "model.GSFusion_GSNO",
        "class": ("GSFusion",),
        "default_run": "GSFusion_GSNO_CAVE_4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "num_basis": opt.num_basis,
            "num_gs_layers": opt.num_gs_layers,
            "edsr_resblocks": opt.edsr_resblocks,
        },
        "init_config": {"zero_init_decoder_last": False},
    },
    "gsno_nogs_identity": {
        "module": "model.baselines.GSFusion_GSNO_NoGS_Identity",
        "class": ("GSFusion",),
        "default_run": "GSFusion_GSNO_NoGS_Identity_CAVE_4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "num_basis": opt.num_basis,
            "num_gs_layers": opt.num_gs_layers,
            "edsr_resblocks": opt.edsr_resblocks,
        },
        "init_config": {"zero_init_decoder_last": False, "custom_reset": True},
    },
    "msi_guided_hsi_gs_scale_consistent": {
        "module": "model.transport.GSFusion_MSI_Guided_ScaleConsistent",
        "class": ("GSFusion",),
        "default_run": "GSFusion_MSI_Guided_HSIGS_ScaleConsistent_CAVE_4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "num_basis": opt.num_basis,
            "num_gs_layers": opt.num_gs_layers,
            "edsr_resblocks": opt.edsr_resblocks,
        },
        "init_config": {"custom_reset": True},
    },
    "e5_spectral_anchored_value": {
        "module": "model.archive.legacy.GSFusion_HRFused_Circular_SpectralAnchoredValue",
        "class": ("GSFusion",),
        "default_run": "E5_HRFused_Circular_SpectralAnchoredValue_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "adci_layers": 3,
        },
        "init_config": {"custom_reset": True},
    },
    "e6_msi_routed_gaussian": {
        "module": "model.GSFusion_E6_MSIRoutedGaussian",
        "class": ("GSFusion",),
        "default_run": "E6_MSIRoutedGaussian_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "adci_layers": 3,
        },
        "init_config": {"custom_reset": True},
    },
    "e6_rag_reference_aware_gaussian_weighting": {
        "module": "model.transport.GSFusion_E6_ReferenceAwareGaussianWeighting",
        "class": ("GSFusion",),
        "default_run": "E6_RAG_ReferenceAwareGaussianWeighting_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "adci_layers": 3,
            "key_dim": 8,
        },
        "init_config": {"custom_reset": True},
    },
    "e6_lr_primitive_gaussian_transport": {
        "module": "model.transport.GSFusion_E6_LRPrimitiveGaussianTransport",
        "class": ("GSFusion",),
        "default_run": "E6_LRPrimitiveGaussianTransport_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "adci_layers": 3,
        },
        "init_config": {"custom_reset": True},
    },
    "e6_lr_primitive_gaussian_transport_adci_cuda_exact": {
        "module": "model.backbones.GSFusion_E6_LRPrimitiveGaussianTransportADCICUDAExact",
        "class": ("GSFusion",),
        "default_run": "E6_LRPrimitiveGaussianTransport_ADCICUDAExact_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "adci_layers": 3,
        },
        "init_config": {"custom_reset": True},
    },
    "e6_lr_primitive_multiscale_gaussian_transport_adci_cuda_exact": {
        "module": "model.backbones.GSFusion_E6_LRPrimitiveMultiScaleGaussianTransportADCICUDAExact",
        "class": ("GSFusion",),
        "default_run": "E6_LRPrimitiveMultiScaleGaussianTransport_ADCICUDAExact_HARVARD8",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "adci_layers": 3,
        },
        "init_config": {
            "custom_reset": True,
            "allowed_model_only_prefixes": (
                "gaussian_transport.expert_scales",
                "gaussian_transport.expert_logit_head.",
            ),
        },
    },
    "e6_lr_primitive_expert_value_multiscale_gaussian_transport_adci_cuda_exact": {
        "module": "model.backbones.GSFusion_E6_LRPrimitiveExpertValueMultiScaleGaussianTransportADCICUDAExact",
        "class": ("GSFusion",),
        "default_run": "E6_LRPrimitiveExpertValueMultiScaleGaussianTransport_ADCICUDAExact_HARVARD8",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "adci_layers": 3,
        },
        "init_config": {
            "custom_reset": True,
            "allowed_model_only_prefixes": (
                "gaussian_transport.expert_scales",
                "gaussian_transport.expert_logit_head.",
                "gaussian_transport.expert_value_residual_head.",
            ),
        },
    },
    "e6_lr_primitive_bounded_covariance_gaussian_transport_adci_cuda_exact": {
        "module": "model.backbones.GSFusion_E6_LRPrimitiveBoundedCovarianceGaussianTransportADCICUDAExact",
        "class": ("GSFusion",),
        "default_run": "E6_LRPrimitiveBoundedCovarianceGaussianTransport_ADCICUDAExact_HARVARD8",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "adci_layers": 3,
        },
        "init_config": {
            "custom_reset": True,
            "allowed_model_only_prefixes": (
                "gaussian_transport.covariance_head.",
            ),
        },
    },
    "e6_lr_primitive_value_context_gaussian_transport_adci_cuda_exact": {
        "module": "model.backbones.GSFusion_E6_LRPrimitiveValueContextGaussianTransportADCICUDAExact",
        "class": ("GSFusion",),
        "default_run": "E6_LRPrimitiveValueContextGaussianTransport_ADCICUDAExact_HARVARD8",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "adci_layers": 3,
        },
        "init_config": {
            "custom_reset": True,
            "allowed_model_only_prefixes": (
                "gaussian_transport.value_context.",
            ),
        },
    },
    "e6_lr_primitive_detail_only": {
        "module": "model.transport.GSFusion_E6_LRPrimitiveDetailOnly",
        "class": ("GSFusion",),
        "default_run": "E6_LRPrimitiveDetailOnly_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "adci_layers": 3,
        },
        "init_config": {"custom_reset": True},
    },
    "e6_lr_primitive_scale_normalized_detail": {
        "module": "model.GSFusion_E6_LRPrimitiveScaleNormalizedDetail",
        "class": ("GSFusion",),
        "default_run": "E6_LRPrimitiveScaleNormalizedDetail_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "adci_layers": 3,
            "anchor_stride_hr": 2,
        },
        "init_config": {
            "custom_reset": True,
            "allowed_model_only_prefixes": (
                "detail_direction_proj.",
                "msi_contrast_gate.",
                "gaussian_transport.",
            ),
        },
    },
    "e7_spectral_anchor_routing": {
        "module": "model.archive.legacy.GSFusion_HRFused_Circular_SpectralAnchorRouting",
        "class": ("GSFusion",),
        "default_run": "E7_HRFused_Circular_SpectralAnchorRouting_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "adci_layers": 3,
        },
        "init_config": {"custom_reset": True},
    },
    "e8_e6_gaussian_aware_hr_local_reconstruction": {
        "module": (
            "model.transport."
            "GSFusion_HRFused_Circular_E6GaussianAwareHRLocalReconstruction"
        ),
        "class": ("GSFusion",),
        "default_run": "E8_E6_GaussianAwareHRLocalReconstruction_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "adci_layers": 3,
        },
        "init_config": {"custom_reset": True},
    },

    "hr_fused_isotropic_gaussian_residual": {
        "module": "model.geometry.GSFusion_HRFused_AdaptiveGaussianResidual_Isotropic",
        "class": ("GSFusion",),
        "default_run": "GSFusion_HRFused_IsotropicGaussianResidual_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim, "num_bands": opt.num_bands,
            "num_msi": opt.num_msi, "adci_layers": 3,
        },
        "init_config": {"custom_reset": True},
    },
    "hr_fused_circular_spectral_value": {
        "module": "model.archive.legacy.GSFusion_HRFused_Circular_SpectralValue",
        "class": ("GSFusion",),
        "default_run": "E2_HRFused_Circular_SpectralValue_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim, "num_bands": opt.num_bands,
            "num_msi": opt.num_msi, "adci_layers": 3,
        },
        "init_config": {"custom_reset": True},
    },
    "hr_fused_circular_primitive_embedding": {
        "module": "model.GSFusion_HRFused_Circular_PrimitiveEmbedding",
        "class": ("GSFusion",),
        "default_run": "E3_HRFused_Circular_PrimitiveEmbedding_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim, "num_bands": opt.num_bands,
            "num_msi": opt.num_msi, "adci_layers": 3,
        },
        "init_config": {"custom_reset": True},
    },
    "e3_hr_factorized_reference_continuous": {
        "module": "model.GSFusion_E3_HRFactorizedReferenceContinuous",
        "class": ("GSFusion",),
        "default_run": "E3_HRFactorizedReferenceContinuous_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "adci_layers": 3,
            "factorized_cap_ratio": 0.25,
        },
        "init_config": {
            "custom_reset": True,
            "allowed_model_only_prefixes": (
                "hsi_spectral_direction.",
                "msi_local_gate.",
                "factorized_out.",
            ),
        },
    },
    "e3_hr_factorized_reference_continuous_adci_cuda_exact": {
        "module": "model.backbones.GSFusion_E3_HRFactorizedReferenceContinuousADCICUDAExact",
        "class": ("GSFusion",),
        "default_run": "E3_HRFactorizedReferenceContinuous_ADCICUDAExact_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "adci_layers": 3,
            "factorized_cap_ratio": 0.25,
        },
        "init_config": {
            "custom_reset": True,
            "allowed_model_only_prefixes": (
                "hsi_spectral_direction.",
                "msi_local_gate.",
                "factorized_out.",
            ),
        },
    },
    "adci_cuda_concat_gi": {
        "module": "model.backbones.GSFusion_ADCICUDAConcatGI",
        "class": ("GSFusion",),
        "default_run": "ADCICUDA_ConcatGI_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "adci_layers": 3,
            "gi_layers": 1,
            "gi_heads": 8,
        },
        "init_config": {},
    },
    "e3_constrained_elliptical_gaussian": {
        "module": "model.GSFusion_E3_ConstrainedEllipticalGaussian",
        "class": ("GSFusion",),
        "default_run": "E3_DIM80_ConstrainedEllipticalGaussian_CAVE4",
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
    "e3_constrained_elliptical_gaussian_adci_cuda_exact": {
        "module": "model.backbones.GSFusion_E3_ConstrainedEllipticalGaussianADCICUDAExact",
        "class": ("GSFusion",),
        "default_run": "E3_DIM80_ConstrainedEllipticalGaussian_ADCICUDAExact_CAVE4",
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
    "e3_constrained_elliptical_gaussian_adci_cuda_exact_continuous": {
        "module": "model.backbones.GSFusion_E3_ConstrainedEllipticalGaussianADCICUDAExactContinuous",
        "class": ("GSFusion",),
        "default_run": "E3_DIM80_ConstrainedEllipticalGaussian_ADCICUDAExact_Continuous_CAVE4",
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
    "e3_gaussian_geometry_shared_isotropic": {
        "module": "model.backbones.GSFusion_E3_GaussianGeometryAblation",
        "class": ("GSFusion",),
        "default_run": "E3_DIM80_GaussianGeometry_SharedIsotropic_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "adci_layers": 3,
            "max_axis_ratio": opt.elliptical_max_axis_ratio,
            "geometry_mode": "shared_isotropic",
        },
        "init_config": {
            "custom_reset": True,
            "allowed_model_only_prefixes": (
                "gaussian_refine.global_scale_logit",
            ),
        },
    },
    "e3_gaussian_geometry_adaptive_isotropic": {
        "module": "model.backbones.GSFusion_E3_GaussianGeometryAblation",
        "class": ("GSFusion",),
        "default_run": "E3_DIM80_GaussianGeometry_AdaptiveIsotropic_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "adci_layers": 3,
            "max_axis_ratio": opt.elliptical_max_axis_ratio,
            "geometry_mode": "adaptive_isotropic",
        },
        "init_config": {"custom_reset": True},
    },
    "e3_gaussian_geometry_anisotropic_no_rotation": {
        "module": "model.backbones.GSFusion_E3_GaussianGeometryAblation",
        "class": ("GSFusion",),
        "default_run": "E3_DIM80_GaussianGeometry_AnisotropicNoRotation_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "adci_layers": 3,
            "max_axis_ratio": opt.elliptical_max_axis_ratio,
            "geometry_mode": "anisotropic_no_rotation",
        },
        "init_config": {"custom_reset": True},
    },
    "e3_gaussian_geometry_full": {
        "module": "model.backbones.GSFusion_E3_GaussianGeometryAblation",
        "class": ("GSFusion",),
        "default_run": "E3_DIM80_GaussianGeometry_Full_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "adci_layers": 3,
            "max_axis_ratio": opt.elliptical_max_axis_ratio,
            "geometry_mode": "full",
        },
        "init_config": {"custom_reset": True},
    },
    "gspan_hsi_continuous_gaussian": {
        "module": "model.backbones.GSFusion_GSPanInspiredHSIContinuous",
        "class": ("GSFusion",),
        "default_run": "GSPanInspired_HSIContinuousGaussian_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "adci_layers": 3,
            "max_axis_ratio": opt.elliptical_max_axis_ratio,
        },
        "init_config": {
            "custom_reset": True,
        },
    },
    "e3_chikusei_spectral_anchored_gaussian_adci_cuda_exact_continuous": {
        "module": "model.backbones.GSFusion_E3_ChikuseiSpectralAnchoredGaussianADCICUDAExactContinuous",
        "class": ("GSFusion",),
        "default_run": "Chikusei_E3_DIM80_SpectralAnchoredHR_Gaussian_ADCICUDAExact_Continuous",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "adci_layers": 3,
            "max_axis_ratio": opt.elliptical_max_axis_ratio,
        },
        "init_config": {"custom_reset": True},
    },
    "e3_chikusei_gated_spectral_anchor_gaussian_adci_cuda_exact_continuous": {
        "module": "model.backbones.GSFusion_E3_ChikuseiGatedSpectralAnchorGaussianADCICUDAExactContinuous",
        "class": ("GSFusion",),
        "default_run": "Chikusei_E3_DIM96_GatedSpectralAnchor_Gaussian_ADCICUDAExact_Continuous",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "adci_layers": 3,
            "max_axis_ratio": opt.elliptical_max_axis_ratio,
            "spectral_anchor_initial_gate": 0.1,
            "gaussian_rms_cap": 0.0,
        },
        "init_config": {"custom_reset": True},
    },
    "e3_chikusei_gated_spectral_anchor_capped_gaussian_adci_cuda_exact_continuous": {
        "module": "model.backbones.GSFusion_E3_ChikuseiGatedSpectralAnchorCappedGaussianADCICUDAExactContinuous",
        "class": ("GSFusion",),
        "default_run": "Chikusei_E3_DIM96_GatedSpectralAnchor_CappedGaussian_ADCICUDAExact_Continuous",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "adci_layers": 3,
            "max_axis_ratio": opt.elliptical_max_axis_ratio,
            "spectral_anchor_initial_gate": 0.1,
        },
        "init_config": {"custom_reset": True},
    },
    "e3_constrained_elliptical_gaussian_adci_cuda_exact_continuous_nogaussian": {
        "module": "model.controls.GSFusion_E3_ConstrainedEllipticalGaussianADCICUDAExactContinuousNoGaussian",
        "class": ("GSFusion",),
        "default_run": "E3_DIM80_ConstrainedEllipticalGaussian_ADCICUDAExact_Continuous_NoGaussian_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "adci_layers": 3,
            "max_axis_ratio": opt.elliptical_max_axis_ratio,
        },
        "init_config": {"custom_reset": True},
    },
    "e3_cai_gki_decoupled_geometry": {
        "module": "model.backbones.Q3_GSFusion_E3_DecoupledGaussianGeometry",
        "class": ("GSFusion",),
        "default_run": "Q3_Decoupled_CAI_GKI_CAVE4",
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
    "e3_parallel_adci_1_cuda_exact_continuous": {
        "module": "model.backbones.GSFusion_E3_ConstrainedEllipticalGaussianADCICUDAExactContinuous",
        "class": ("GSFusion",),
        "default_run": "E3_DIM80_ParallelADCI_L1PerStream_ADCICUDAExact_Continuous_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "adci_layers": 1,
            "max_axis_ratio": opt.elliptical_max_axis_ratio,
        },
        "init_config": {
            "custom_reset": True,
            "allowed_model_only_prefixes": (
                "gaussian_refine.anisotropy_head.",
            ),
        },
    },
    "e3_parallel_adci_2_cuda_exact_continuous": {
        "module": "model.backbones.GSFusion_E3_ConstrainedEllipticalGaussianADCICUDAExactContinuous",
        "class": ("GSFusion",),
        "default_run": "E3_DIM80_ParallelADCI_L2PerStream_ADCICUDAExact_Continuous_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "adci_layers": 2,
            "max_axis_ratio": opt.elliptical_max_axis_ratio,
        },
        "init_config": {
            "custom_reset": True,
            "allowed_model_only_prefixes": (
                "gaussian_refine.anisotropy_head.",
            ),
        },
    },
    "e3_parallel_adci_4_cuda_exact_continuous": {
        "module": "model.backbones.GSFusion_E3_ConstrainedEllipticalGaussianADCICUDAExactContinuous",
        "class": ("GSFusion",),
        "default_run": "E3_DIM80_ParallelADCI_L4PerStream_ADCICUDAExact_Continuous_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "adci_layers": 4,
            "max_axis_ratio": opt.elliptical_max_axis_ratio,
        },
        "init_config": {
            "custom_reset": True,
            "allowed_model_only_prefixes": (
                "gaussian_refine.anisotropy_head.",
            ),
        },
    },
    "e3_hsi_upsample_first_adci_cuda_exact_continuous": {
        "module": "model.backbones.GSFusion_E3_HSIUpsampleFirstADCICUDAExactContinuous",
        "class": ("GSFusion",),
        "default_run": "E3_DIM80_HSIUpsampleFirst_ADCICUDAExact_Continuous_CAVE4",
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
    "e3_concat_before_adci_cuda_exact_continuous": {
        "module": "model.backbones.GSFusion_E3_ConcatBeforeADCICUDAExactContinuous",
        "class": ("GSFusion",),
        "default_run": "E3_DIM80_ConcatBeforeADCI_ADCICUDAExact_Continuous_CAVE4",
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
    "e3_concat_before_adci_1_cuda_exact_continuous": {
        "module": "model.backbones.GSFusion_E3_ConcatBeforeADCI_LayerSweep",
        "class": ("GSFusion",),
        "default_run": "E3_DIM80_ConcatBeforeADCI_L1_ADCICUDAExact_Continuous_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "concat_adci_layers": 1,
            "max_axis_ratio": opt.elliptical_max_axis_ratio,
        },
        "init_config": {"custom_reset": True},
    },
    "e3_concat_before_adci_2_cuda_exact_continuous": {
        "module": "model.backbones.GSFusion_E3_ConcatBeforeADCI_LayerSweep",
        "class": ("GSFusion",),
        "default_run": "E3_DIM80_ConcatBeforeADCI_L2_ADCICUDAExact_Continuous_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "concat_adci_layers": 2,
            "max_axis_ratio": opt.elliptical_max_axis_ratio,
        },
        "init_config": {"custom_reset": True},
    },
    "e3_concat_before_adci_3_cuda_exact_continuous": {
        "module": "model.backbones.GSFusion_E3_ConcatBeforeADCI_LayerSweep",
        "class": ("GSFusion",),
        "default_run": "E3_DIM80_ConcatBeforeADCI_L3_ADCICUDAExact_Continuous_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "concat_adci_layers": 3,
            "max_axis_ratio": opt.elliptical_max_axis_ratio,
        },
        "init_config": {"custom_reset": True},
    },
    "e3_concat_before_adci_6_cuda_exact_continuous": {
        "module": "model.backbones.GSFusion_E3_ConcatBeforeADCI_LayerSweep",
        "class": ("GSFusion",),
        "default_run": "E3_DIM80_ConcatBeforeADCI_L6_ADCICUDAExact_Continuous_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "concat_adci_layers": 6,
            "max_axis_ratio": opt.elliptical_max_axis_ratio,
        },
        "init_config": {"custom_reset": True},
    },
    "e3_constrained_elliptical_gaussian_adci_pytorch_continuous": {
        "module": "model.backbones.GSFusion_E3_ConstrainedEllipticalGaussianADCIPyTorchContinuous",
        "class": ("GSFusion",),
        "default_run": "E3_DIM80_ConstrainedEllipticalGaussian_ADCIPyTorch_Continuous_CAVE4",
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
    "e3_bounded_covariance_gaussian": {
        "module": "model.geometry.GSFusion_E3_BoundedCovarianceGaussian",
        "class": ("GSFusion",),
        "default_run": "E3_DIM80_BoundedCovarianceGaussian_CAVE4",
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
                "gaussian_refine.covariance_head.",
            ),
        },
    },
    "e3_primitive80_constrained_ellipse_cuda": {
        "module": "model.geometry.GSFusion_E3_Primitive80ConstrainedEllipse",
        "class": ("GSFusion",),
        "default_run": "E3_Primitive80_ConstrainedEllipse_CUDA_CAVE4",
        "kwargs": lambda opt: {
            "dim": 64,
            "primitive_dim": 80,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "adci_layers": 3,
            "max_axis_ratio": opt.elliptical_max_axis_ratio,
        },
        "init_config": {
            "custom_reset": True,
            "allowed_model_only_prefixes": (
                "primitive_input.",
                "primitive_residual.",
                "gaussian_refine.",
            ),
        },
    },
    "e3_three_local_conv_parammatched": {
        "module": "model.controls.GSFusion_E3_ThreeLocalConvParamMatched",
        "class": ("GSFusion",),
        "default_run": "E3_ThreeLocalConvParamMatched_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "adci_layers": 3,
            "local_hidden_dim": 90,
        },
        "init_config": {
            "custom_reset": True,
            "allowed_model_only_prefixes": ("local_conv_refine.",),
        },
    },
    "e3_single_conv_replacement": {
        "module": "model.controls.GSFusion_E3_SingleConvReplacement",
        "class": ("GSFusion",),
        "default_run": "E3_SingleConvReplacement_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "adci_layers": 3,
        },
        "init_config": {
            "custom_reset": True,
            "allowed_model_only_prefixes": ("gaussian_refine.conv.",),
        },
    },
    "e3_hpm_a": {
        "module": "model.controls.GSFusion_E3_HPM_A",
        "class": ("GSFusion",),
        "default_run": "E3_HPM_A_BranchLocalization_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim, "num_bands": opt.num_bands,
            "num_msi": opt.num_msi, "adci_layers": 3,
            "pointwise_expansion": 2,
        },
        "init_config": {"custom_reset": True},
    },
    "e3_hsi_neighbor_norm_adci": {
        "module": "model.backbones.GSFusion_E3_HSINeighborNormADCI",
        "class": ("GSFusion",),
        "default_run": "E3_HSINeighborNormADCI_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim, "num_bands": opt.num_bands,
            "num_msi": opt.num_msi, "adci_layers": 3,
        },
        "init_config": {
            "custom_reset": True,
            "allowed_model_only_prefixes": (
                "adci_hsi_layers.0.neighbor_log_scale",
                "adci_hsi_layers.1.neighbor_log_scale",
                "adci_hsi_layers.2.neighbor_log_scale",
            ),
        },
    },
    "e3_hsi_anchored_rmscap_adci": {
        "module": "model.backbones.GSFusion_E3_HSIAnchoredRMSCapADCI",
        "class": ("GSFusion",),
        "default_run": "E3_HSIAnchoredRMSCapADCI_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim, "num_bands": opt.num_bands,
            "num_msi": opt.num_msi, "adci_layers": 3,
            "adci_rms_cap_multiplier": opt.adci_rms_cap_multiplier,
        },
        "init_config": {
            "custom_reset": True,
            "allowed_model_only_prefixes": (
                "adci_hsi_layers.0.running_score_rms",
                "adci_hsi_layers.0.num_batches_tracked",
                "adci_hsi_layers.1.running_score_rms",
                "adci_hsi_layers.1.num_batches_tracked",
                "adci_hsi_layers.2.running_score_rms",
                "adci_hsi_layers.2.num_batches_tracked",
            ),
        },
    },
    "e3_pointwise": {
        "module": "model.controls.GSFusion_E3_Pointwise",
        "class": ("GSFusion",),
        "default_run": "E3_Pointwise_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim, "num_bands": opt.num_bands,
            "num_msi": opt.num_msi, "adci_layers": 3,
            "pointwise_expansion": 2,
        },
        "init_config": {"custom_reset": True},
    },
    "e3_gdi_v1": {
        "module": "model.backbones.GSFusion_GDI",
        "class": ("GSFusion",),
        "default_run": "E3_GDI_v1_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim, "num_bands": opt.num_bands,
            "num_msi": opt.num_msi, "pointwise_blocks": 2,
            "pointwise_expansion": 2.0,
        },
        "init_config": {"custom_reset": True},
    },
    "e3_gdi_derivative_v2": {
        "module": "model.backbones.GSFusion_GDI_Derivative",
        "class": ("GSFusion",),
        "default_run": "E3_GDI_Derivative_v2_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim, "num_bands": opt.num_bands,
            "num_msi": opt.num_msi, "pointwise_blocks": 2,
            "pointwise_expansion": 2.0,
        },
        "init_config": {"custom_reset": True},
    },
    "hr_fused_circular_primitive_embedding_stdmin020": {
        "module": "model.geometry.GSFusion_E3_StdMin020",
        "class": ("GSFusion",),
        "default_run": "E3_HRFused_Circular_PrimitiveEmbedding_StdMin020_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim, "num_bands": opt.num_bands,
            "num_msi": opt.num_msi, "adci_layers": 3,
        },
        "init_config": {"custom_reset": True},
    },
    "e3_hr_s2_circular": {
        "module": "model.transport.GSFusion_HRSemiDenseGaussian",
        "class": ("GSFusion",),
        "default_run": "E3_HRSemiDense_S2_Circular_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim, "num_bands": opt.num_bands,
            "num_msi": opt.num_msi, "adci_layers": 3,
            "anchor_stride_hr": 2,
        },
        "init_config": {"custom_reset": True},
    },
    "e3_hr_s4_circular": {
        "module": "model.transport.GSFusion_HRSemiDenseGaussian",
        "class": ("GSFusion",),
        "default_run": "E3_HRSemiDense_S4_Circular_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim, "num_bands": opt.num_bands,
            "num_msi": opt.num_msi, "adci_layers": 3,
            "anchor_stride_hr": 4,
        },
        "init_config": {"custom_reset": True},
    },
    "e3_ablation_canvas_scaled_std": {
        "module": "model.geometry.GSFusion_E3_Ablation_CanvasScaledStd",
        "class": ("GSFusion",),
        "default_run": "E3_Ablation_CanvasScaledStd_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim, "num_bands": opt.num_bands,
            "num_msi": opt.num_msi, "adci_layers": 3,
        },
        "init_config": {"custom_reset": True},
    },
    "e3_ablation_raw_scatter_sum": {
        "module": "model.mechanism.GSFusion_E3_Ablation_RawScatterSum",
        "class": ("GSFusion",),
        "default_run": "E3_Ablation_RawScatterSum_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim, "num_bands": opt.num_bands,
            "num_msi": opt.num_msi, "adci_layers": 3,
        },
        "init_config": {"custom_reset": True},
    },
    "e3_ablation_fixed_window": {
        "module": "model.mechanism.GSFusion_E3_Ablation_FixedWindow",
        "class": ("GSFusion",),
        "default_run": "E3_Ablation_FixedWindow_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim, "num_bands": opt.num_bands,
            "num_msi": opt.num_msi, "adci_layers": 3,
        },
        "init_config": {"custom_reset": True},
    },
    "e3_ablation_three_gaussian_layers": {
        "module": "model.mechanism.GSFusion_E3_Ablation_ThreeGaussianLayers",
        "class": ("GSFusion",),
        "default_run": "E3_Ablation_ThreeGaussianLayers_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim, "num_bands": opt.num_bands,
            "num_msi": opt.num_msi, "adci_layers": 3,
        },
        "init_config": {
            "custom_reset": True,
            "allowed_model_only_prefixes": ("gaussian_refine_extra.",),
        },
    },
    "e3_three_constrained_elliptical_gaussian": {
        "module": "model.GSFusion_E3_ThreeConstrainedEllipticalGaussian",
        "class": ("GSFusion",),
        "default_run": "E3_ThreeConstrainedEllipticalGaussian_CAVE4",
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
                "gaussian_refine_extra.",
            ),
        },
    },
    "e3_parallel_constrained_elliptical_gaussian": {
        "module": "model.GSFusion_E3_ParallelConstrainedEllipticalGaussian",
        "class": ("GSFusion",),
        "default_run": "E3_ParallelConstrainedEllipticalGaussian_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "adci_layers": 3,
            "max_axis_ratio": opt.elliptical_max_axis_ratio,
        },
        "init_config": {"custom_reset": True},
    },
    "e3_parallel_two_constrained_elliptical_gaussian": {
        "module": "model.GSFusion_E3_ParallelTwoConstrainedEllipticalGaussian",
        "class": ("GSFusion",),
        "default_run": "E3_ParallelTwoConstrainedEllipticalGaussian_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "adci_layers": 3,
            "max_axis_ratio": opt.elliptical_max_axis_ratio,
        },
        "init_config": {"custom_reset": True},
    },
    "e3_parallel_two_constrained_elliptical_gaussian_adci_cuda_exact": {
        "module": "model.backbones.GSFusion_E3_ParallelTwoConstrainedEllipticalGaussianADCICUDAExact",
        "class": ("GSFusion",),
        "default_run": "E3_ParallelTwoConstrainedEllipticalGaussian_ADCICUDAExact_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "adci_layers": 3,
            "max_axis_ratio": opt.elliptical_max_axis_ratio,
        },
        "init_config": {"custom_reset": True},
    },
    "e3_parallel_two_shared_value_gaussian_adci_cuda_exact": {
        "module": "model.backbones.GSFusion_E3_ParallelTwoSharedValueADCICUDAExact",
        "class": ("GSFusion",),
        "default_run": "E3_ParallelTwoSharedValueGaussian_ADCICUDAExact_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "adci_layers": 3,
            "max_axis_ratio": opt.elliptical_max_axis_ratio,
        },
        "init_config": {"custom_reset": True},
    },
    "e3_anchor_bounded_shared_value_aux_gaussian_adci_cuda_exact": {
        "module": "model.backbones.GSFusion_E3_AnchorBoundedSharedValueAuxADCICUDAExact",
        "class": ("GSFusion",),
        "default_run": "E3_DIM80_AnchorBoundedSharedValueAuxGaussian_ADCICUDAExact_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "adci_layers": 3,
            "max_axis_ratio": opt.elliptical_max_axis_ratio,
            "auxiliary_beta_max": 0.25,
            "correction_rms_cap_ratio": 0.25,
            "auxiliary_gate_init_logit": -6.0,
        },
        "init_config": {"custom_reset": True},
    },
    "e3_gaussian_operator1": {
        "module": "model.operators.GSFusion_E3_GaussianOperator1",
        "class": ("GSFusion",),
        "default_run": "E3_GaussianOperator1_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim, "num_bands": opt.num_bands,
            "num_msi": opt.num_msi, "adci_layers": 3,
        },
        "init_config": {
            "custom_reset": True,
            "allowed_model_only_prefixes": ("operator_blocks.",),
        },
    },
    "e3_gaussian_operator1_cell_integrated_3sigma": {
        "module": "model.operators.GSFusion_E3_GaussianOperator1CellIntegrated3Sigma",
        "class": ("GSFusion",),
        "default_run": "E3_GaussianOperator1_CellIntegrated3Sigma_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim, "num_bands": opt.num_bands,
            "num_msi": opt.num_msi, "adci_layers": 3,
        },
        "init_config": {
            "custom_reset": True,
            "allowed_model_only_prefixes": ("operator_blocks.",),
        },
    },
    "e3_gaussian_operator1_point_query_conservative_window": {
        "module": "model.operators.GSFusion_E3_GaussianOperator1PointQueryConservativeWindow",
        "class": ("GSFusion",),
        "default_run": "E3_GaussianOperator1_PointQueryConservativeWindow_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim, "num_bands": opt.num_bands,
            "num_msi": opt.num_msi, "adci_layers": 3,
        },
        "init_config": {
            "custom_reset": True,
            "allowed_model_only_prefixes": ("operator_blocks.",),
        },
    },
    "e3_gaussian_operator2_effective_mixing": {
        "module": "model.operators.GSFusion_E3_GaussianOperator2EffectiveMixing",
        "class": ("GSFusion",),
        "default_run": "E3_GaussianOperator2EffectiveMixing_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim, "num_bands": opt.num_bands,
            "num_msi": opt.num_msi, "adci_layers": 3,
            "operator_std_min_px": 0.45,
            "operator_initial_std_px": 0.55,
        },
        "init_config": {
            "custom_reset": True,
            "allowed_model_only_prefixes": ("operator_blocks.",),
        },
    },
    "e3_gaussian_detail_integral": {
        "module": "model.operators.GSFusion_E3_GaussianDetailIntegral",
        "class": ("GaussianDetailIntegralGSFusion",),
        "default_run": "E3_GaussianDetailIntegral_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim, "num_bands": opt.num_bands,
            "num_msi": opt.num_msi, "adci_layers": 3,
        },
        "init_config": {
            "custom_reset": True,
            "allowed_model_only_prefixes": ("detail_operator.", "detail_out."),
        },
    },
    "e3_pointwise_detail_injection": {
        "module": "model.operators.GSFusion_E3_GaussianDetailIntegral",
        "class": ("PointwiseDetailInjectionGSFusion",),
        "default_run": "E3_PointwiseDetailInjection_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim, "num_bands": opt.num_bands,
            "num_msi": opt.num_msi, "adci_layers": 3,
        },
        "init_config": {
            "custom_reset": True,
            "allowed_model_only_prefixes": ("detail_operator.", "detail_out."),
        },
    },
    "e3_conv_zero_sum_detail_integral": {
        "module": "model.operators.GSFusion_E3_GaussianDetailIntegral",
        "class": ("ConvZeroSumDetailIntegralGSFusion",),
        "default_run": "E3_ConvZeroSumDetailIntegral_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim, "num_bands": opt.num_bands,
            "num_msi": opt.num_msi, "adci_layers": 3,
        },
        "init_config": {
            "custom_reset": True,
            "allowed_model_only_prefixes": ("detail_operator.", "detail_out."),
        },
    },
    "e3_forced_local_operator": {
        "module": "model.operators.GSFusion_E3_ForcedLocalOperator",
        "class": ("GSFusion",),
        "default_run": "E3_ForcedLocalOperator_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim, "num_bands": opt.num_bands,
            "num_msi": opt.num_msi, "adci_layers": 3,
        },
        "init_config": {
            "custom_reset": True,
            "allowed_model_only_prefixes": ("operator_blocks.",),
        },
    },
    "e3_forced_cnn_operator": {
        "module": "model.operators.GSFusion_E3_ForcedCNNOperator",
        "class": ("GSFusion",),
        "default_run": "E3_ForcedCNNOperator_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim, "num_bands": opt.num_bands,
            "num_msi": opt.num_msi, "adci_layers": 3,
        },
        "init_config": {
            "custom_reset": True,
            "allowed_model_only_prefixes": ("operator_blocks.",),
        },
    },
    "e3_no_gaussian": {
        "module": "model.controls.GSFusion_E3_NoGaussian",
        "class": ("GSFusion",),
        "default_run": "E3_NoGaussian_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim, "num_bands": opt.num_bands,
            "num_msi": opt.num_msi, "adci_layers": 3,
        },
        "init_config": {"custom_reset": True},
    },
    "e3_param_matched_pointwise": {
        "module": "model.controls.GSFusion_E3_ParamMatchedPointwise",
        "class": ("GSFusion",),
        "default_run": "E3_ParamMatchedPointwise_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "adci_layers": 3,
            "pointwise_hidden_dim": opt.pointwise_hidden_dim,
        },
        "init_config": {
            "custom_reset": True,
            "allowed_model_only_prefixes": ("pointwise_latent.",),
        },
    },
    "e3_constrained_elliptical_gaussian_adci_cuda_exact_continuous_param_matched_pointwise": {
        "module": "model.controls.GSFusion_E3_ConstrainedEllipticalGaussianADCICUDAExactContinuousParamMatchedPointwise",
        "class": ("GSFusion",),
        "default_run": "E3_DIM80_ADCICUDAExact_Continuous_ParamMatchedPointwise_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "adci_layers": 3,
            "max_axis_ratio": opt.elliptical_max_axis_ratio,
            "pointwise_hidden_dim": opt.pointwise_hidden_dim,
        },
        "init_config": {
            "custom_reset": True,
            "allowed_model_only_prefixes": ("pointwise_latent.",),
        },
    },
    "e3_fixed_sigma": {
        "module": "model.mechanism.GSFusion_E3_FixedSigma",
        "class": ("GSFusion",),
        "default_run": "E3_FixedSigma_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "adci_layers": 3,
            "fixed_sigma_hr": opt.fixed_sigma_hr,
        },
        "init_config": {
            "custom_reset": True,
            "allowed_model_only_prefixes": (
                "gaussian_refine.fixed_sigma_hr",
            ),
        },
    },
    "e3_adci_cuda_exact": {
        "module": "model.backbones.GSFusion_E3_ADCICUDAExact",
        "class": ("GSFusion",),
        "default_run": "E3_ADCICUDAExact_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "adci_layers": 3,
        },
        "init_config": {"custom_reset": True},
    },
    "e3_lsci_v1": {
        "module": "model.backbones.GSFusion_E3_LSCI_v1",
        "class": ("GSFusion",),
        "default_run": "E3_LSCI_v1_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "adci_layers": 3,
            "num_heads": opt.lsci_num_heads,
            "ffn_hidden_dim": opt.lsci_ffn_hidden_dim or opt.dim,
        },
        "init_config": {
            "custom_reset": True,
            "allowed_model_only_prefixes": (
                "lsci_hsi_layers.",
                "lsci_msi_layers.",
            ),
        },
    },
    "e3_cari_v1": {
        "module": "model.backbones.GSFusion_E3_CARI_v1",
        "class": ("GSFusion",),
        "default_run": "E3_CARI_v1_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim,
            "num_bands": opt.num_bands,
            "num_msi": opt.num_msi,
            "adci_layers": 3,
            "num_groups": 8,
            "score_hidden_dim": 110,
        },
        "init_config": {
            "custom_reset": True,
            "allowed_model_only_prefixes": (
                "cari_hsi_layers.",
                "cari_msi_layers.",
            ),
        },
    },
    "hr_fused_full_geometry_primitive_embedding": {
        "module": "model.geometry.GSFusion_HRFused_FullGeometry_PrimitiveEmbedding",
        "class": ("GSFusion",),
        "default_run": "E3FullGeometry_HRFused_PrimitiveEmbedding_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim, "num_bands": opt.num_bands,
            "num_msi": opt.num_msi, "adci_layers": 3,
        },
        "init_config": {"custom_reset": True},
    },
    "hr_fused_circular_spectral_enhanced_value": {
        "module": "model.archive.legacy.GSFusion_HRFused_Circular_SpectralEnhancedValue",
        "class": ("GSFusion",),
        "default_run": "E4_HRFused_Circular_SpectralEnhancedValue_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim, "num_bands": opt.num_bands,
            "num_msi": opt.num_msi, "adci_layers": 3,
        },
        "init_config": {"custom_reset": True},
    },
    "exact_a_residual_latent_circular_gaussian": {
        "module": "model.archive.legacy.GSFusion_ResidualLatent_CircularGaussianResidual",
        "class": ("GSFusion",),
        "default_run": "ExactA_ResidualLatent_CircularGaussian_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim, "num_bands": opt.num_bands,
            "num_msi": opt.num_msi, "adci_layers": 3,
        },
        "init_config": {"custom_reset": True},
    },
    "exact_c_hsi_base_msi_geometry_transport": {
        "module": "model.archive.legacy.GSFusion_HSIBase_MSIGeometryGaussianTransport",
        "class": ("GSFusion",),
        "default_run": "ExactC_HSIBase_MSIGeometryTransport_CAVE4",
        "kwargs": lambda opt: {
            "dim": opt.dim, "num_bands": opt.num_bands,
            "num_msi": opt.num_msi, "adci_layers": 3,
        },
        "init_config": {"custom_reset": True},
    },
}

MODEL_SPECS["e3_cnn_mlp"] = {
    "module": "model.controls.GSFusion_E3_CNNMLP",
    "class": ("GSFusion",),
    "default_run": "E3_CNNMLP_CAVE4",
    "kwargs": lambda opt: {"dim": opt.dim, "num_bands": opt.num_bands,
                           "num_msi": opt.num_msi, "adci_layers": 3},
    "init_config": {"custom_reset": True, "allowed_model_only_prefixes":
                    ("cnn_hsi_layers.", "cnn_msi_layers.", "pointwise_latent.")},
}

MODEL_CHOICES = sorted(set(MODEL_SPECS.keys()) | set(MODEL_ALIASES.keys()))


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
        fallback_module = importlib.import_module("model.GSFusion_GSNO")
        loss_fn = getattr(fallback_module, "compute_loss")
    return model, model_label, loss_fn


def setup_run_context(opt):
    spec = MODEL_SPECS[opt.model]
    env_keys = [
        spec.get("run_env"),
        default_run_env_name(opt.model),
        "GSFUSION_RUN_NAME",
        "GSFUSION_V2_RUN_NAME",
        "GSFUSION_V1_RUN_NAME",
    ]
    env_run_name = None
    for env_key in env_keys:
        if env_key:
            env_run_name = os.environ.get(env_key)
            if env_run_name:
                break
    default_run_name = spec["default_run"]
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


def forward_tiled(test_model, lr_hsi, hr_msi, sf, tile_size, halo, return_aux=False):
    """Run aligned HR tiles while keeping LR/HR coordinates synchronized."""
    if lr_hsi.shape[0] != 1 or hr_msi.shape[0] != 1:
        raise ValueError("Tiled evaluation currently requires batch size 1")
    height, width = hr_msi.shape[-2:]
    tile_size = int(tile_size)
    halo = int(halo)
    if tile_size <= 0:
        if return_aux:
            return test_model(lr_hsi, hr_msi, sf, return_aux=True)
        return test_model(lr_hsi, hr_msi, sf)
    if tile_size % sf or halo % sf:
        raise ValueError(f"eval tile size/halo must be divisible by sf={sf}")
    if height % sf or width % sf:
        raise ValueError(f"HR evaluation size {(height, width)} must be divisible by sf={sf}")

    output = None
    aux_output = {}
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
            if return_aux:
                tile_output, tile_aux = test_model(
                    lr_tile, msi_tile, sf, return_aux=True
                )
            else:
                tile_output = test_model(lr_tile, msi_tile, sf)
                tile_aux = {}

            if output is None:
                output = tile_output.new_empty(
                    tile_output.shape[0], tile_output.shape[1], height, width
                )
            cy0, cy1 = y0 - ey0, y1 - ey0
            cx0, cx1 = x0 - ex0, x1 - ex0
            output[..., y0:y1, x0:x1] = tile_output[..., cy0:cy1, cx0:cx1]

            for key, value in tile_aux.items():
                if not torch.is_tensor(value) or value.ndim != 4:
                    continue
                if value.shape[-2:] != tile_output.shape[-2:]:
                    continue
                if key not in aux_output:
                    aux_output[key] = value.new_empty(
                        value.shape[0], value.shape[1], height, width
                    )
                aux_output[key][..., y0:y1, x0:x1] = value[..., cy0:cy1, cx0:cx1]

    if return_aux:
        return output, aux_output
    return output


def evaluate(
    test_model,
    data_path=None,
    sf=4,
    dataset_name="cave",
    return_x0=False,
    return_base=False,
    dataset_options=None,
):
    test_model.eval()
    if data_path is None:
        dataset_root = choose_dataset_root(dataset_name)
        data_path = (
            os.path.join(dataset_root, "test_chikusei_gt_rgb.h5")
            if dataset_name == "chikusei"
            else os.path.join(dataset_root, "Test")
        )
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
    x0_psnr_total = 0.0
    base_psnr_total = 0.0
    k = 0
    for batch in loader_test:
        LR, RGB, HR = unpack_dataset_batch(batch)
        with torch.no_grad():
            LR, RGB, HR = Variable(LR), Variable(RGB), Variable(HR)
            LR, RGB, HR = LR.cuda(), RGB.cuda(), HR.cuda()
            if return_x0 or return_base:
                out, aux = forward_tiled(
                    test_model, LR, RGB, opt_evaluate.sf,
                    opt_evaluate.eval_tile_size, opt_evaluate.eval_tile_halo,
                    return_aux=True,
                )
                if return_x0:
                    x0_result = aux["x0_no_dc"].cpu().data.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)
                if return_base:
                    base_result = aux["base"].cpu().data.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)
            else:
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
        if return_x0:
            x0_psnr_total += cal_psnr(x0_result, HR_np)
        if return_base:
            base_psnr_total += cal_psnr(base_result, HR_np)
        k += 1

    average_psnr = psnr_total / k
    evaluate.last_metrics = {
        "psnr": average_psnr,
        "sam": sam_total / k,
        "ergas_fixed4": ergas_total / k,
        "num_images": k,
    }
    if return_x0 and return_base:
        return average_psnr, x0_psnr_total / k, base_psnr_total / k
    if return_x0:
        return average_psnr, x0_psnr_total / k
    if return_base:
        return average_psnr, base_psnr_total / k
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
    parser.add_argument("--chikusei_augment", default=1, type=int, choices=(0, 1),
                        help="Enable Chikusei random rotations and flips")
    parser.add_argument("--remote_hsi_augment", default=1, type=int, choices=(0, 1),
                        help="Enable unified remote-HSI random rotations and flips")
    parser.add_argument("--eval_tile_size", default=0, type=int,
                        help="Aligned HR tile size for evaluation; 0 keeps full-image inference")
    parser.add_argument("--eval_tile_halo", default=0, type=int,
                        help="Context halo around each evaluation tile")

    parser.add_argument("--ep_total", default=500, type=int, help='Total epochs')
    parser.add_argument("--e_every", default=5, type=int, help='Evaluation interval')
    parser.add_argument("--lr", default=4e-4, type=float, help='Initial learning rate')
    parser.add_argument("--sam_weight", default=0.1, type=float, help='SAM loss weight')
    parser.add_argument("--sam_warmup_epochs", default=5, type=int, help='SAM warmup epochs')
    parser.add_argument("--grad_clip", default=0.0, type=float, help='Max grad norm; 0 disables clipping')
    parser.add_argument("--grad_accum_steps", default=1, type=int,
                        help="Number of micro-batches per optimizer update")
    parser.add_argument("--optimizer", default="adam", choices=["adam", "adamw"],
                        help="Optimizer; AdamW is used by the released OTIAS trainer")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Optimizer weight decay")
    parser.add_argument("--scheduler", default="cosine", choices=["cosine", "cosine_iter", "constant", "sgdr", "step", "multistep", "afno_multistep"],
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
    parser.add_argument("--afno_scheduler_stop", default=200, type=int,
                        help="Exclusive final milestone for the supplied AFNO scheduler")

    parser.add_argument("--model", default="gsno_nogs_identity", choices=MODEL_CHOICES,
                        help='Which model implementation to train')
    parser.add_argument("--run_name", default=None, type=str,
                        help='Optional experiment name for logs/checkpoints')
    parser.add_argument(
        "--checkpoint_root",
        default="Checkpoint",
        type=str,
        help="Root directory containing per-experiment checkpoint folders",
    )
    parser.add_argument("--dim", default=32, type=int)
    parser.add_argument(
        "--pointwise_hidden_dim", default=0, type=int,
        help="0 automatically matches the formal E3 Gaussian-branch parameter delta",
    )
    parser.add_argument(
        "--fixed_sigma_hr", default=0.0, type=float,
        help="Fixed circular Gaussian sigma in physical HR-pixel units",
    )
    parser.add_argument(
        "--elliptical_max_axis_ratio", default=2.0, type=float,
        help="Maximum principal-axis std ratio for constrained elliptical Gaussian",
    )
    parser.add_argument(
        "--adci_rms_cap_multiplier", default=2.0, type=float,
        help="Per-channel HSI ADCI score cap as a multiple of running 4x RMS",
    )
    parser.add_argument("--lsci_num_heads", default=8, type=int)
    parser.add_argument(
        "--lsci_ffn_hidden_dim", default=0, type=int,
        help="0 uses --dim for the LSCI pointwise FFN hidden width",
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
    parser.add_argument("--num_basis", default=16, type=int)
    parser.add_argument("--num_gs_layers", default=3, type=int)
    parser.add_argument("--edsr_resblocks", default=6, type=int)
    parser.add_argument(
        "--random_sf_list",
        default="",
        type=str,
        help="Comma-separated train scales for random-scale training, e.g. '2,3,4'. Empty keeps --sf fixed.",
    )
    parser.add_argument(
        "--random_sf_mode",
        default="batch",
        choices=["batch", "epoch"],
        help="When --random_sf_list is set, sample a scale per batch or per epoch.",
    )

    opt = parser.parse_args()
    opt.model = normalize_model_name(opt.model)
    opt.random_sf_values = [
        int(sf.strip()) for sf in opt.random_sf_list.split(",") if sf.strip()
    ]
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

    if hasattr(model_ref, "stage2_parameter_counts"):
        counts = model_ref.stage2_parameter_counts()
        message = "Stage2 parameters: " + ", ".join(
            f"{key}={value}" for key, value in counts.items()
        )
        print(f"[INFO] {message}")
        logger.info(message)

    # Initialize the final decoder layer only for models that opt into it.
    if init_config.get("zero_init_decoder_last", False):
        if hasattr(model_ref, "decoder") and len(model_ref.decoder) > 0:
            last_layer = model_ref.decoder[-1]
            if hasattr(last_layer, "weight"):
                nn.init.zeros_(last_layer.weight)
            if hasattr(last_layer, "bias") and last_layer.bias is not None:
                nn.init.zeros_(last_layer.bias)
            print("[INIT] zero-initialized decoder last layer")
        else:
            print("[WARN] init_config requested decoder zero-init, but model has no decoder")
    else:
        print("[INIT] skip decoder zero-init (by init_config)")

    if init_config.get("custom_reset", False):
        if hasattr(model_ref, "reset_custom_init"):
            model_ref.reset_custom_init()
            print("[INIT] applied model reset_custom_init")
        else:
            print("[WARN] init_config requested custom_reset, but model has no reset_custom_init")


    common_data_generator_state = None
    print(
        "[INIT] provenance "
        f"mode={opt.effective_initialization_mode} "
        f"checkpoint={opt.common_init_checkpoint or '<none>'}"
    )
    if opt.common_init_checkpoint:
        if init_config.get("legacy_gsno_init", False):
            payload = torch.load(opt.common_init_checkpoint, map_location="cpu")
            report = model_ref.load_legacy_gsno_state_dict(
                payload,
                map_old_ffn=bool(init_config.get("map_old_ffn", True)),
            )
            allowed_missing_prefixes = tuple(
                init_config.get("allowed_missing_prefixes", ("gaussian_refine.",))
            )
            invalid_missing = [
                key for key in report["missing_keys"]
                if not key.startswith(allowed_missing_prefixes)
            ]
            if invalid_missing:
                raise RuntimeError(
                    "Legacy GSNO initialization omitted shared parameters: "
                    f"{invalid_missing}"
                )
            if isinstance(payload, dict):
                common_data_generator_state = payload.get("data_generator_state")
            print(
                "[INIT] loaded legacy GSNO initialization "
                f"matched={report['loaded_count']} "
                f"new_gaussian={len(report['missing_keys'])} "
                f"path={opt.common_init_checkpoint}"
            )
        else:
            matched_count, model_only_keys, common_data_generator_state = (
                load_matching_initialization(model_ref, opt.common_init_checkpoint)
            )
            allowed_model_only_prefixes = tuple(
                init_config.get(
                    "allowed_model_only_prefixes",
                    ("continuous_upsampler.",),
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
                    "num_gs_layers": opt.num_gs_layers,
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
    elif opt.scheduler == "afno_multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(range(1, opt.afno_scheduler_stop, opt.lr_step_size)),
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
        model_for_epoch = model.module if hasattr(model, "module") else model
        if hasattr(model_for_epoch, "set_training_epoch"):
            model_for_epoch.set_training_epoch(epoch)
        model.train()

        if opt.random_sf_values and opt.random_sf_mode == "batch":
            loaders_by_sf = {}
            iterators_by_sf = {}
            for sf_value in opt.random_sf_values:
                sf_opt = copy.copy(opt)
                sf_opt.sf = sf_value
                sf_dataset = dataset_class(sf_opt, HR_HSI, HR_MSI)
                sf_loader = tud.DataLoader(
                    sf_dataset,
                    num_workers=DEFAULT_NUM_WORKERS,
                    batch_size=opt.batch_size,
                    shuffle=True,
                    generator=train_loader_generator,
                    worker_init_fn=seed_data_worker,
                )
                loaders_by_sf[sf_value] = sf_loader
                iterators_by_sf[sf_value] = iter(sf_loader)
            steps_per_epoch = max(len(loader) for loader in loaders_by_sf.values())
            par = tqdm(range(steps_per_epoch), desc=f'Training Ep{epoch}', unit='batch', ascii=True)
            epoch_train_sfs = []
        else:
            train_sf = random.choice(opt.random_sf_values) if opt.random_sf_values else opt.sf
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
            epoch_train_sfs = [train_sf]

        epoch_loss = 0.0
        start_time = time.time()
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch_item in enumerate(par):
            if opt.random_sf_values and opt.random_sf_mode == "batch":
                train_sf = random.choice(opt.random_sf_values)
                epoch_train_sfs.append(train_sf)
                try:
                    LR, RGB, HR = unpack_dataset_batch(
                        next(iterators_by_sf[train_sf])
                    )
                except StopIteration:
                    iterators_by_sf[train_sf] = iter(loaders_by_sf[train_sf])
                    LR, RGB, HR = unpack_dataset_batch(
                        next(iterators_by_sf[train_sf])
                    )
            else:
                LR, RGB, HR = unpack_dataset_batch(batch_item)

            LR, RGB, HR = Variable(LR), Variable(RGB), Variable(HR)
            LR, RGB, HR = LR.cuda(), RGB.cuda(), HR.cuda()

            model_for_loss = model.module if hasattr(model, "module") else model
            if hasattr(model_for_loss, "training_step"):
                loss = model_for_loss.training_step(LR, RGB, HR, train_sf)
            else:
                out = model(LR, RGB, train_sf)
                loss_kwargs = {
                    "sam_warmup_epochs": opt.sam_warmup_epochs,
                    "sam_weight": opt.sam_weight,
                }
                loss = compute_loss_fn(out, HR, epoch, **loss_kwargs)

            epoch_loss += loss.item()

            (loss / opt.grad_accum_steps).backward()
            if hasattr(model_for_loss, "collect_gradient_stats"):
                model_for_loss.collect_gradient_stats()
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
        elif opt.scheduler == "afno_multistep":
            # Preserve the supplied AFNO script's explicit epoch stepping.
            scheduler.step(epoch)
        else:
            scheduler.step()

        epoch_loss_avg = epoch_loss / max(1, steps_per_epoch)
        writer.add_scalar('Loss/train/epoch', epoch_loss_avg, epoch)
        if opt.random_sf_values:
            sf_counts = {sf_value: epoch_train_sfs.count(sf_value) for sf_value in opt.random_sf_values}
            logger.info('Train random sf epoch {} mode {} counts {}'.format(epoch, opt.random_sf_mode, sf_counts))
            print('Train random sf epoch {} mode {} counts {}'.format(epoch, opt.random_sf_mode, sf_counts))

        if epoch % e_every == 0:
            ave = evaluate(
                model, data_path=opt.test_data_path, sf=opt.sf,
                dataset_name=opt.dataset, dataset_options=opt,
            )
            ave_x0 = None
            eval_metrics = getattr(evaluate, "last_metrics", None)
            if ave > bestpsnr:
                bestpsnr = ave
                best_epoch = epoch

                best_model_path = os.path.join(ckpt_dir, 'best_model.pth')
                if os.path.exists(best_model_path):
                    os.remove(best_model_path)
                torch.save(model.state_dict(), best_model_path)

            if ave_x0 is None:
                logger.info(
                    'Epoch: {}/{} average psnr: {:.7f} bestpsnr: {:.7f}, bestepoch: {}'.format(
                        epoch, ep_total - 1, ave, bestpsnr, best_epoch
                    )
                )
            else:
                logger.info(
                    'Epoch: {}/{} x0 psnr: {:.7f} x1 psnr: {:.7f} bestx1: {:.7f}, bestepoch: {}'.format(
                        epoch, ep_total - 1, ave_x0, ave, bestpsnr, best_epoch
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
            if ave_x0 is not None:
                writer.add_scalar('PSNR/test_x0', ave_x0, epoch)
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
            f'train_sf: {(epoch_train_sfs[-1] if epoch_train_sfs else train_sf)} '
            f'loss: {epoch_loss_avg:.6f} '
            f'lr: {optimizer.param_groups[0]["lr"]:.2e} '
            f'time: {elapsed_time:.2f}s'
        )

    print('Epoch: {} best PSNR: {:.7f}'.format(best_epoch, bestpsnr))
    logger.info('Epoch: {} best PSNR: {:.7f}'.format(best_epoch, bestpsnr))
    writer.close()
