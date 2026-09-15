#!/usr/bin/env python3
"""Validate and summarize the completed GSNO Gaussian-operator diagnostics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


CHECKPOINT_SHA = "7e821ef1095e2f61bbbea66afcb356fe4095d428ab8fcf376157380750e093b0"
EXTENSION_SHA = "1f713bd1aa4b032838ccf25ac1fdca737eb11f6415ec7d3b81a40a90b777e25f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--consistency", required=True)
    parser.add_argument("--density", required=True)
    parser.add_argument("--stability", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def load(path: str) -> dict:
    return json.loads(Path(path).resolve().read_text(encoding="utf-8"))


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def f(value: float, digits: int = 7) -> str:
    return f"{float(value):.{digits}g}"


def derive_claim_decisions(
    binary: dict,
    consistency: dict,
    floor_inactive: bool,
    stability: dict,
) -> dict:
    adaptive_rows = [
        row for row in consistency["summary"]
        if row["support"] == "adaptive_3sigma"
    ]
    by_method = {}
    for row in adaptive_rows:
        by_method.setdefault(row["method"], []).append(row)

    def converged(row: dict) -> bool:
        return (
            row["empirical_log_error_vs_log_h_slope"] is not None
            and row["empirical_log_error_vs_log_h_slope"] > 0.0
            and row["finest_relative_l2"] < row["coarsest_relative_l2"]
        )

    normalized_rows = by_method.get("normalized", [])
    raw_rows = by_method.get("raw_scatter_vs_integral", [])
    normalized_convergence = len(normalized_rows) == 3 and all(
        converged(row) for row in normalized_rows
    )
    raw_density_divergence = len(raw_rows) == 3 and all(
        row["finest_relative_l2"] > row["coarsest_relative_l2"]
        for row in raw_rows
    )
    density_normalization_robust = normalized_convergence and raw_density_divergence

    stability_rows = [
        stability["summary"][str(scale)] for scale in stability["scales"][1:]
    ]
    value_bound_term_dominates = all(
        row["value_term_sample_mean_mean"] > row["kernel_term_sample_mean_mean"]
        for row in stability_rows
    )
    stability_bound_verified = stability["bound_violation_max"] <= 5e-6
    binary_verified = binary["all_checks_pass"] and binary["numerical_checks_pass"]
    core_evidence_pass = all(
        (
            binary_verified,
            normalized_convergence,
            density_normalization_robust,
            floor_inactive,
            stability_bound_verified,
        )
    )

    return {
        "binary_provenance_and_equivalence_verified": binary_verified,
        "formal_cuda_isolated_normalized_convergence_supported": normalized_convergence,
        "density_normalization_more_sampling_robust_than_raw_scatter": density_normalization_robust,
        "real_checkpoint_floor_inactive_on_audited_pixels": floor_inactive,
        "gaussian_stage_stability_bound_numerically_verified": stability_bound_verified,
        "value_bound_term_dominates_kernel_term_at_all_shifted_scales": value_bound_term_dominates,
        "relationship_to_causal_swap": (
            "consistent: value term dominates at every audited shifted scale"
            if value_bound_term_dominates
            else "mixed: do not claim that this feature-space bound independently confirms the causal swap"
        ),
        "formal_model_decision": (
            "retain_formal_model; no theorem-driven architecture change is recommended"
            if core_evidence_pass
            else "manual_review_required; do not automatically modify the formal model from one failed diagnostic"
        ),
        "gaussian_stage_cde_claim_strength": (
            "conditional numerical consistency for the isolated normalized Gaussian stage"
            if normalized_convergence
            else "not established by the current isolated convergence diagnostic"
        ),
        "strict_algebraic_properties": [
            "constant reproduction when nonnegative local mass remains above the denominator floor",
            "common positive kernel-mass scaling invariance when the floor is inactive",
            "positive convex averaging and boundedness",
            "non-expansiveness with respect to values for a fixed normalized kernel",
        ],
        "assumption_dependent_analytical_properties": [
            "cross-observation perturbation stability under bounded values and positive local mass",
            "uniform-grid normalized quadrature consistency for sufficiently regular continuous fields and kernels",
            "formal rectangular 3sigma tail lower bound and covariance well-posedness under the audited parameterization",
        ],
        "not_established": [
            "whole-network strict discretization invariance or CDE",
            "ReNO",
            "area-analytic Gaussian integration in the formal ExactContinuous CUDA",
            "true arbitrary-resolution super-resolution from direct point output queries",
        ],
    }


def main() -> None:
    args = parse_args()
    if os.environ.get("CONDA_DEFAULT_ENV") != "sqy":
        raise RuntimeError("formal summary requires `conda activate sqy`")

    binary = load(args.binary)
    cell = load(args.cell)
    consistency = load(args.consistency)
    density = load(args.density)
    stability = load(args.stability)

    assert binary["all_checks_pass"] is True
    assert binary["numerical_checks_pass"] is True
    assert binary["formal_extension_sha256"] == EXTENSION_SHA
    assert all(binary["provenance_checks"].values())

    assert cell["protocol_version"] == "conservative-overlap-q2-v1"
    assert cell["images"] == 12
    assert cell["checkpoint_sha256"] == CHECKPOINT_SHA

    assert consistency["protocol_version"] == "fixed-query-uniform-source-quadrature-v2"
    assert consistency["environment"]["conda_default_env"] == "sqy"
    assert consistency["environment"]["extension_sha256"] == EXTENSION_SHA
    assert len(consistency["per_grid"]) == 48
    assert len(consistency["summary"]) == 18
    assert consistency["property_tests"]["value_non_expansiveness"]["satisfied_with_2e_6_tolerance"]
    opacity = consistency["property_tests"]["opacity_common_scaling"]
    assert opacity["multipliers"] == [0.25, 0.5, 1.0, 1.25]
    assert max(item["maximum_scaled_opacity"] for item in opacity["results"].values()) <= 1.0

    assert density["protocol_version"] == "frozen-cave-density-floor-audit-v1"
    assert density["checkpoint_sha256"] == CHECKPOINT_SHA
    assert density["images_per_scale"] == 12
    assert density["runtime"]["conda_default_env"] == "sqy"
    assert density["runtime"]["extension_sha256"] == EXTENSION_SHA

    assert stability["protocol_version"] == "paired-scale-sampled-normalized-kernel-decomposition-v1"
    assert stability["checkpoint_sha256"] == CHECKPOINT_SHA
    assert stability["images"] == 12
    assert stability["sample_queries_per_scene"] == 256
    assert stability["runtime"]["conda_default_env"] == "sqy"
    assert stability["runtime"]["extension_sha256"] == EXTENSION_SHA
    assert stability["mirror_max_abs_vs_formal_cuda"] <= 5e-5
    assert stability["bound_violation_max"] <= 5e-6

    synthetic_rows = [
        row
        for row in consistency["summary"]
        if row["support"] == "adaptive_3sigma"
    ]
    synthetic_rows.sort(key=lambda row: (row["case"], row["method"]))
    opacity_max_error = max(
        item["max_abs"] for item in opacity["results"].values()
    )

    density_rows = []
    for scale in density["scales"]:
        values = density["summary"][str(scale)]
        density_rows.append({"scale": scale, **values})
    floor_inactive = all(
        row["min"] >= density["implementation_floor"]
        and row["fraction_lt_1em06"] == 0.0
        for row in density_rows
    )

    stability_rows = []
    for scale in stability["scales"][1:]:
        values = stability["summary"][str(scale)]
        stability_rows.append({"scale": scale, **values})

    cell_rows = []
    for target_size in sorted(cell["summary"], key=int):
        values = cell["summary"][target_size]
        cell_rows.append({"target_size": int(target_size), **values})

    claim_decisions = derive_claim_decisions(
        binary,
        consistency,
        floor_inactive,
        stability,
    )

    summary = {
        "state": "validated post-hoc factual summary; no training or model selection",
        "checkpoint_sha256": CHECKPOINT_SHA,
        "formal_extension_sha256": EXTENSION_SHA,
        "binary_equivalence": {
            "all_checks_pass": binary["all_checks_pass"],
            "tensor_errors": binary["tensor_errors"],
        },
        "synthetic_operator": {
            "protocol_version": consistency["protocol_version"],
            "adaptive_3sigma_convergence": synthetic_rows,
            "constant_channel_max_abs_error": consistency["property_tests"]["constant_channel_max_abs_error"],
            "opacity_common_scaling_max_abs_error": opacity_max_error,
            "value_non_expansiveness": consistency["property_tests"]["value_non_expansiveness"],
        },
        "real_checkpoint_density": {
            "protocol_version": density["protocol_version"],
            "floor": density["implementation_floor"],
            "floor_inactive_on_audited_pixels": floor_inactive,
            "per_scale": density_rows,
        },
        "gaussian_stage_stability": {
            "protocol_version": stability["protocol_version"],
            "mirror_max_abs_vs_formal_cuda": stability["mirror_max_abs_vs_formal_cuda"],
            "bound_violation_max": stability["bound_violation_max"],
            "per_scale": stability_rows,
        },
        "output_cell_boundary": {
            "protocol_version": cell["protocol_version"],
            "per_target_size": cell_rows,
            "interpretation_boundary": cell["interpretation_boundary"],
        },
        "claim_decisions": claim_decisions,
        "claim_boundary": (
            "These diagnostics support the Gaussian-stage numerical and stability "
            "claims only. They do not prove whole-network discretization invariance, "
            "continuous-discrete equivalence, or arbitrary-resolution super-resolution."
        ),
    }

    lines = [
        "# GSNO Gaussian算子证据自动汇总（待人工复核）",
        "",
        "本报告只在五组artifact全部通过协议、SHA、场景数和数值门禁后生成。它是事实摘录，不自动升级论文主张。",
        "",
        "## 1. 运行与来源门禁",
        "",
        f"- 正式checkpoint SHA256：`{CHECKPOINT_SHA}`",
        f"- 正式adaptive3 binary SHA256：`{EXTENSION_SHA}`",
        f"- 原构建源码重编译与正式binary数值等价：`{binary['all_checks_pass']}`",
        f"- 正式CUDA镜像最大绝对误差：`{f(stability['mirror_max_abs_vs_formal_cuda'])}`",
        "",
        "## 2. 合成Gaussian算子收敛",
        "",
        "| 连续场 | 方法 | 最粗网格误差 | 最细网格误差 | log(error)-log(h)斜率 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in synthetic_rows:
        lines.append(
            f"| {row['case']} | {row['method']} | {f(row['coarsest_relative_l2'])} | "
            f"{f(row['finest_relative_l2'])} | {f(row['empirical_log_error_vs_log_h_slope'])} |"
        )
    value_test = consistency["property_tests"]["value_non_expansiveness"]
    lines.extend(
        [
            "",
            f"常数通道最大误差为`{f(consistency['property_tests']['constant_channel_max_abs_error'])}`；合法opacity公共缩放最大输出误差为`{f(opacity_max_error)}`；value的L∞输入扰动`{f(value_test['input_perturbation_linf'])}`对应输出差`{f(value_test['output_difference_linf'])}`。",
            "",
            "## 3. 正式checkpoint原始density",
            "",
            "| 倍率 | min | p0.1 | p1 | median | `<1e-6`比例 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in density_rows:
        lines.append(
            f"| {row['scale']}x | {f(row['min'])} | {f(row['p0_1'])} | "
            f"{f(row['p1'])} | {f(row['p50'])} | {f(row['fraction_lt_1em06'])} |"
        )
    lines.extend(
        [
            "",
            f"在已审计像素上`1e-6` floor是否完全未触发：`{floor_inactive}`。这里统计的是clamp前原始density。",
            "",
            "## 4. Gaussian-stage跨观测稳定性分解",
            "",
            "| 倍率 | value rel-L2 | normalized kernel L1 | value界项 | kernel界项 | Gaussian输出差 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in stability_rows:
        lines.append(
            f"| {row['scale']}x | {f(row['value_relative_l2_mean'])} | "
            f"{f(row['kernel_l1_sample_mean_mean'])} | {f(row['value_term_sample_mean_mean'])} | "
            f"{f(row['kernel_term_sample_mean_mean'])} | {f(row['output_difference_sample_mean_mean'])} |"
        )
    lines.extend(
        [
            "",
            f"逐查询点稳定性界最大超差为`{f(stability['bound_violation_max'])}`。该表是Gaussian stage的同场景特征空间诊断，不是whole-network误差界，也不替代已有geometry/value causal swap。",
            "",
            "## 5. 输出cell边界实验",
            "",
            "| 输出网格 | direct PSNR | q=2 PSNR | canonical restriction PSNR | q=2相对direct | 锚点类型 |",
            "|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in cell_rows:
        lines.append(
            f"| {row['target_size']} | {f(row['direct_psnr_to_cell_gt'])} | "
            f"{f(row['quadrature_psnr_to_cell_gt'])} | {f(row['canonical_psnr_to_cell_gt'])} | "
            f"{f(row['quadrature_gain_over_direct_db'])} | {row['quadrature_anchor_kind']} |"
        )
    lines.extend(
        [
            "",
            "## 6. 结论边界",
            "",
            "这些结果只能支持Gaussian stage的数值性质、正式实现来源、真实density条件和局部稳定性分解。它们不能证明whole-network strict DI/CDE，也不能把粗输出点查询称作真实任意尺度超分。",
            "",
            "## 7. 对理论总纲八个问题的逐项回答",
            "",
            "1. **严格代数性质：** 在非负权重、局部质量高于floor等显式条件下，常数复现、公共正质量缩放不变、凸平均有界性和固定kernel下对value的非扩张性成立。这里的‘严格’只属于Gaussian stage，不属于whole network。",
            "2. **带条件的分析命题：** cross-observation扰动界、uniform-grid normalized quadrature consistency、矩形3sigma尾界和协方差良定性均依赖源码审计中写明的正质量、正则性与参数化条件。",
            f"3. **正式CUDA收敛：** `{claim_decisions['formal_cuda_isolated_normalized_convergence_supported']}`；结论强度为“{claim_decisions['gaussian_stage_cde_claim_strength']}”。",
            f"4. **density normalization受控优势：** `{claim_decisions['density_normalization_more_sampling_robust_than_raw_scatter']}`；只有normalized三种场都随网格细化收敛且RawScatter三种场都随采样密度发散时才记为通过。",
            f"5. **真实CAVE density floor：** 已审计像素上floor完全未触发=`{claim_decisions['real_checkpoint_floor_inactive_on_audited_pixels']}`。该结论不外推到未审计数据集。",
            f"6. **value/kernel与causal swap：** {claim_decisions['relationship_to_causal_swap']}。该分解是稳定性界项，不是第二次因果干预。",
            f"7. **Gaussian-stage CDE可写强度：** {claim_decisions['gaussian_stage_cde_claim_strength']}；禁止升级成whole-network strict DI/CDE或ReNO。",
            f"8. **是否修改正式模型：** {claim_decisions['formal_model_decision']}。cell-aware输出属于单独读出方向，不回写或替换当前最强正式checkpoint。",
            "",
            "### 明确未成立",
            "",
            *[f"- {item}" for item in claim_decisions["not_established"]],
            "",
        ]
    )

    output_json = Path(args.output_json).resolve()
    output_md = Path(args.output_md).resolve()
    atomic_write(output_json, json.dumps(summary, indent=2))
    atomic_write(output_md, "\n".join(lines))
    print(json.dumps({"output_json": str(output_json), "output_md": str(output_md)}, indent=2))


if __name__ == "__main__":
    main()
