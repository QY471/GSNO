# -*- coding: utf-8 -*-
"""
E6 单文件学习版：MSI-guided selection of local HSI content for Gaussian value.

本文件将以下两个原文件合并到一起：
1. GSFusion_E6_MSIRoutedGaussian.py
   （原名：GSFusion_HRFused_Circular_MSIGuidedHSIRouting.py）
2. GSFusion_HRFused_Circular_PrimitiveValueCommon.py

合并只改变代码组织方式，不改变 E6 的计算逻辑。
这是便于阅读和交接的单文件注释版；正式训练默认仍使用
Train_Cave.py 中注册的 e6_msi_routed_gaussian。
仍然依赖：
- model.GSFusion_GSNO 中的 ADCI、compute_loss、sam_loss
- extensions/adaptive3_rasterizer 中的 CUDA GaussianRasterizer
"""

# 允许类型注解引用尚未定义的类，并推迟类型注解求值。
from __future__ import annotations

# math 用于 attention logits 的 sqrt(d) 缩放。
import math
# os、sys 用于定位并加入 CUDA rasterizer 扩展路径。
import os
import sys
# 这些类型用于补充函数参数、返回值和统计字典的类型说明。
from typing import Dict, List, Optional, Tuple

# PyTorch 主包。
import torch
# 神经网络模块与参数初始化工具。
import torch.nn as nn
# 常用无状态函数：插值、padding、unfold、GELU 等。
import torch.nn.functional as F

# E6 继续复用原 GSNO 中的 ADCI 编码器和损失函数。
from model.GSFusion_GSNO import ADCI, compute_loss, sam_loss


def _resolve_adaptive_gaussian_rasterizer():
    """定位并导入自定义 CUDA Gaussian rasterizer。"""

    # __file__ 是当前文件路径；连续 dirname 两次得到项目根目录。
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 优先读取环境变量指定的扩展路径；若未指定，则使用项目内默认路径。
    extension_root = os.environ.get(
        "GSFUSION_ADAPTIVE_RASTER_ROOT",
        os.path.join(repo_root, "extensions", "adaptive3_rasterizer"),
    )

    # 若扩展目录尚未加入 Python 搜索路径，则插入到最前面。
    if extension_root not in sys.path:
        sys.path.insert(0, extension_root)

    # 从编译完成的 CUDA/PyTorch 扩展中导入 GaussianRasterizer。
    from diff_srgaussian_rasterization import GaussianRasterizer

    # 把类返回给调用者，避免在文件加载时立即实例化。
    return GaussianRasterizer


class HRAdaptiveGaussianResidualDualSource(nn.Module):
    """
    HR 网格圆 Gaussian residual renderer。

    DualSource 表示：
    - transport_x 负责预测 Gaussian geometry；
    - value_x 负责预测 Gaussian 携带的 residual value。
    """

    def __init__(
        self,
        dim: int,
        std_min_px: float = 0.30,
        std_max_px: float = 1.50,
        max_offset_px: float = 1.00,
        sigma_radius: float = 3.0,
    ) -> None:
        # 初始化 nn.Module 的内部成员与参数注册机制。
        super().__init__()

        # 动态定位 CUDA rasterizer 类。
        GaussianRasterizer = _resolve_adaptive_gaussian_rasterizer()

        # 渲染通道数为 dim+1：前 dim 个是分子，最后 1 个是 density。
        self.rasterizer = GaussianRasterizer(dim + 1)

        # 保存超参数，并统一转成稳定的 Python 数值类型。
        self.dim = int(dim)
        self.std_min_px = float(std_min_px)
        self.std_max_px = float(std_max_px)
        self.max_offset_px = float(max_offset_px)
        self.sigma_radius = float(sigma_radius)

        # geometry_head 对每个 HR 位置输出 2 个数：opacity raw 和 std raw。
        self.geometry_head = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, 2, kernel_size=1),
        )

        # residual_value_head 把 value source 转成每个 Gaussian 携带的 dim 维修正值。
        self.residual_value_head = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
        )

        # 保存最近一次 forward 的诊断统计；初始化时尚无数据。
        self.last_stats: Optional[Dict[str, float]] = None

    def reset_residual_init(self) -> None:
        """让 Gaussian value 分支初始化为零输出。"""

        # 最后一层权重清零，使初始 delta_value 为 0。
        nn.init.zeros_(self.residual_value_head[-1].weight)
        # 最后一层偏置也清零。
        nn.init.zeros_(self.residual_value_head[-1].bias)

    @staticmethod
    def _pixel_centers(height, width, device, dtype):
        """生成 HR 网格全部像素中心坐标，排列为 [1, H*W, 2]。"""

        # yy、xx 分别保存每个网格位置的 y 坐标和 x 坐标。
        yy, xx = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )

        # 按 [x, y] 顺序拼接，再展平所有 H*W 个位置。
        return torch.stack([xx, yy], dim=-1).view(1, height * width, 2)

    def forward(
        self,
        transport_x: torch.Tensor,
        value_x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        输入：
        - transport_x: [B,C,H,W]，用于 geometry head；
        - value_x: [B,C,H,W]，用于 value head。

        输出：
        - gaussian_delta: [B,C,H,W]，密度归一化后的 Gaussian latent residual。
        """

        # 未单独传入 value_x 时，geometry 和 value 共用 transport_x。
        if value_x is None:
            value_x = transport_x

        # 双源输入必须形状一致，保证每个位置一一对应。
        if transport_x.shape != value_x.shape:
            raise ValueError(
                "transport_x and value_x must have identical BCHW shapes, got "
                f"{tuple(transport_x.shape)} and {tuple(value_x.shape)}"
            )

        # 拆出 batch、通道数和 HR 空间尺寸。
        batch, channels, height, width = transport_x.shape

        # 从 transport_x 预测每个位置的 2 个 geometry raw 参数。
        raw = self.geometry_head(transport_x)

        # [B,2,H,W] → [B,H,W,2] → [B,H*W,2]。
        raw = raw.permute(0, 2, 3, 1).contiguous().view(
            batch, height * width, 2
        )

        # 第 1 个 raw 经过 sigmoid 映射到 (0.05, 1.0) 作为 opacity。
        opacity = 0.05 + 0.95 * torch.sigmoid(raw[..., 0:1])

        # 第 2 个 raw 映射到 [std_min_px, std_max_px]。
        std_scalar_px = self.std_min_px + (
            self.std_max_px - self.std_min_px
        ) * torch.sigmoid(raw[..., 1:2])

        # 同一个标量复制到 x、y 两轴，因此是圆 Gaussian：sigma_x=sigma_y。
        std_px = std_scalar_px.expand(-1, -1, 2)

        # E6 固定 offset 为 0；此变量仅用于统计与明确结构语义。
        offset_px = raw.new_zeros(batch, height * width, 2)

        # E6 固定 rho 为 0，因此没有 xy 相关项。
        rho = raw.new_zeros(batch, height * width, 1)

        # 为每个 HR 位置生成固定像素中心坐标，并扩展到整个 batch。
        means_px = self._pixel_centers(
            height, width, transport_x.device, transport_x.dtype
        ).expand(batch, height * width, 2)

        # 将坐标限制在合法图像范围内；对当前整数中心而言通常不会改变数值。
        means_px = torch.stack(
            [
                means_px[..., 0].clamp(0.0, float(width - 1)),
                means_px[..., 1].clamp(0.0, float(height - 1)),
            ],
            dim=-1,
        )

        # 从 value source 预测每个 primitive 携带的 dim 维 latent residual value。
        delta_value = self.residual_value_head(value_x)

        # [B,C,H,W] → [B,H*W,C]，满足 rasterizer 的 primitive 列表输入格式。
        delta_value = delta_value.permute(0, 2, 3, 1).contiguous().view(
            batch, height * width, channels
        )

        # 为每个 primitive 构造一个常数 1，用于同步渲染 Gaussian density。
        ones = delta_value.new_ones(batch, height * width, 1)

        # 前 channels 维渲染加权 value，最后 1 维渲染权重总和 density。
        values_with_density = torch.cat([delta_value, ones], dim=-1)

        # 根据最大 sigma 和 3-sigma 窗口估计 rasterizer 所需的归一化覆盖比例。
        raster_ratio = min(
            1.0,
            max(
                self.sigma_radius * self.std_max_px / max(width, 1),
                self.sigma_radius * self.std_max_px / max(height, 1),
            ),
        )

        # 调用 CUDA rasterizer，在同一 HR 网格上渲染全部 Gaussian primitive。
        rasterized = self.rasterizer(
            opacity.float(),
            means_px.float(),
            std_px.float(),
            rho.float(),
            values_with_density.float(),
            height,
            width,
            1,
            raster_ratio,
            debug=False,
            adaptive_window=True,
            sigma_radius=self.sigma_radius,
        )

        # rasterizer 输出通常为 [B,H,W,C+1]，转回 PyTorch 常用 BCHW。
        rasterized = rasterized.permute(0, 3, 1, 2).contiguous()

        # 前 channels 个通道是 sum_i w_i * value_i，即归一化前的分子。
        numerator = rasterized[:, :channels]

        # 最后一个通道是 sum_i w_i，即当前位置的 Gaussian density。
        density = rasterized[:, channels:channels + 1]

        # 用 density 归一化，得到局部 Gaussian 加权平均；防止除零。
        gaussian_delta = numerator / density.clamp_min(1e-6)

        # 下面只记录诊断数据，不参与梯度和模型预测。
        with torch.no_grad():
            # geometry source 的平均绝对幅值。
            transport_abs = transport_x.detach().abs().mean()
            # primitive residual value 的平均绝对幅值。
            value_abs = delta_value.detach().abs().mean()
            # 渲染后 Gaussian delta 的平均绝对幅值。
            delta_abs = gaussian_delta.detach().abs().mean()

            # 保存最近一次 Gaussian 状态，供日志或消融诊断读取。
            self.last_stats = {
                "hrgs_opacity_mean": float(opacity.detach().mean()),
                "hrgs_opacity_std": float(opacity.detach().std()),
                "hrgs_std_x_mean_px": float(std_px[..., 0].detach().mean()),
                "hrgs_std_y_mean_px": float(std_px[..., 1].detach().mean()),
                "hrgs_std_min_px": float(std_px.detach().min()),
                "hrgs_std_max_px": float(std_px.detach().max()),
                "hrgs_rho_abs_mean": float(rho.detach().abs().mean()),
                "hrgs_offset_abs_mean_px": float(offset_px.detach().abs().mean()),
                "hrgs_offset_max_abs_px": float(offset_px.detach().abs().max()),
                "hrgs_density_min": float(density.detach().min()),
                "hrgs_density_mean": float(density.detach().mean()),
                "hrgs_value_abs_mean": float(value_abs),
                "hrgs_delta_abs_mean": float(delta_abs),
                "hrgs_input_abs_mean": float(transport_abs),
                "hrgs_delta_input_ratio": float(
                    delta_abs / (transport_abs + 1e-8)
                ),
                "hrgs_raster_ratio": float(raster_ratio),
                "hrgs_adaptive_window": 1.0,
                "hrgs_sigma_radius": float(self.sigma_radius),
            }

        # 返回 [B,C,H,W] 的 Gaussian latent residual。
        return gaussian_delta


class MSIGuidedHSILocalRouting(nn.Module):
    """使用 MSI keys，从 3×3 HSI values 中选择局部内容。"""

    def __init__(self, dim: int, routing_dim: Optional[int] = None) -> None:
        # 初始化 nn.Module。
        super().__init__()

        # 保存原始 latent 通道数。
        self.dim = int(dim)

        # routing_dim 默认取 dim/4；dim=64 时为 16。
        self.routing_dim = int(routing_dim or max(1, dim // 4))

        # HSI 中心特征 → query，通道从 dim 压到 routing_dim。
        self.q_proj = nn.Conv2d(dim, self.routing_dim, 1, bias=False)

        # MSI 特征 → key，通道同样压到 routing_dim。
        self.k_proj = nn.Conv2d(dim, self.routing_dim, 1, bias=False)

        # HSI 特征 → value，保持 dim 通道。
        self.v_proj = nn.Conv2d(dim, dim, 1, bias=False)

        # 将聚合后的 HSI context 映射成可加到 E_g 上的 routing_delta。
        self.out_proj = nn.Conv2d(dim, dim, 1, bias=True)

        # 保存最近一次 routing 权重统计。
        self.last_stats: Optional[Dict[str, float]] = None

    def reset_output_init(self) -> None:
        """让 routing 分支初始化为零修正。"""

        # out_proj 清零后，初始 routing_delta 恒为 0。
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, f_h: torch.Tensor, f_m: torch.Tensor) -> torch.Tensor:
        """
        f_h: 上采样到 HR 网格的 HSI latent。
        f_m: HR-MSI latent。
        两者形状必须相同：[B,C,H,W]。
        """

        # routing 要逐位置对齐，因此先检查形状是否一致。
        if f_h.shape != f_m.shape:
            raise ValueError(
                f"F_H and F_M must match, got {tuple(f_h.shape)} and "
                f"{tuple(f_m.shape)}"
            )

        # 读取 batch 和 HR 空间尺寸；_channels 不再单独使用。
        batch, _channels, height, width = f_h.shape

        # 当前 HSI feature 产生 query。
        q = self.q_proj(f_h)

        # MSI feature 产生 key。
        k = self.k_proj(f_m)

        # HSI feature 产生被选择和聚合的 value。
        v = self.v_proj(f_h)

        # 对 key 做边缘复制 padding，再提取每个位置的 3×3 共 9 个邻域 key。
        k_neighbors = F.unfold(
            F.pad(k, (1, 1, 1, 1), mode="replicate"), kernel_size=3
        ).view(batch, self.routing_dim, 9, height, width)

        # 同样提取与 9 个 key 一一对应的 9 个 HSI value。
        v_neighbors = F.unfold(
            F.pad(v, (1, 1, 1, 1), mode="replicate"), kernel_size=3
        ).view(batch, self.dim, 9, height, width)

        # q 与 9 个 MSI key 做点积，再除以 sqrt(routing_dim) 稳定数值尺度。
        logits = (
            q.unsqueeze(2) * k_neighbors
        ).sum(dim=1) / math.sqrt(self.routing_dim)

        # 在 9 个邻居维度上 softmax，使每个位置的 9 个权重和为 1。
        weights = torch.softmax(logits, dim=1)

        # 用 MSI 决定的权重，对 9 个 HSI value 加权求和。
        context = (v_neighbors * weights.unsqueeze(1)).sum(dim=2)

        # 将 context 投影为最终 routing 修正量。
        routing_delta = self.out_proj(context)

        # 以下统计不参与训练梯度。
        with torch.no_grad():
            # 计算 9 邻域权重熵；越大越接近均匀选择，越小越集中。
            entropy = -(
                weights.detach() * weights.detach().clamp_min(1e-8).log()
            ).sum(dim=1).mean()

            # 保存 routing 权重的集中程度诊断。
            self.last_stats = {
                "primitive_routing_entropy": float(entropy),
                "primitive_routing_max_weight_mean": float(
                    weights.detach().max(dim=1).values.mean()
                ),
            }

        # 返回 [B,C,H,W] 的 HSI 局部内容修正量。
        return routing_delta


class HRFusedPrimitiveValueBase(nn.Module):
    """E6 的公共双分支编码、主融合、primitive 与 Gaussian 层。"""

    def __init__(
        self,
        dim: int = 64,
        num_bands: int = 31,
        num_msi: int = 3,
        adci_layers: int = 3,
    ) -> None:
        # 初始化 nn.Module。
        super().__init__()

        # 保存模型维度与输入通道配置。
        self.dim = int(dim)
        self.num_bands = int(num_bands)
        self.num_msi = int(num_msi)

        # LR-HSI：31 通道 → dim 通道；只做逐像素通道投影。
        self.shallow_encoder1 = nn.Conv2d(num_bands, dim, 1)

        # HR-MSI：3 通道 → dim 通道。
        self.shallow_encoder2 = nn.Conv2d(num_msi, dim, 1)

        # HSI 分支堆叠 adci_layers 个 ADCI；默认 3 层。
        self.adci_hsi_layers = nn.ModuleList(
            [ADCI(dim, dim) for _ in range(adci_layers)]
        )

        # MSI 分支同样堆叠 3 个 ADCI。
        self.adci_msi_layers = nn.ModuleList(
            [ADCI(dim, dim) for _ in range(adci_layers)]
        )

        # 主融合头：拼接后的 2*dim 通道压回 dim 通道。
        self.conv0 = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )

        # Gaussian latent residual renderer。
        self.gaussian_refine = HRAdaptiveGaussianResidualDualSource(dim)

        # 最终 residual decoder 的第一层，保持 dim 通道。
        self.fc1 = nn.Conv2d(dim, dim, 1)

        # 最终 residual decoder 的第二层，dim → 31 波段。
        self.fc2 = nn.Conv2d(dim, num_bands, 1)

        # 保存当前随机数生成器状态，使新增分支不改变旧 E3 层的初始化契约。
        extra_rng_state = torch.get_rng_state()

        # 从 joint=[F_H,F_M] 构造基础 primitive embedding E0。
        self.primitive_input = nn.Conv2d(2 * dim, dim, 1)

        # 对 E0 做逐位置的非线性通道残差细化。
        self.primitive_residual = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )

        # 恢复随机数状态，保持后续训练脚本的初始化可复现性。
        torch.set_rng_state(extra_rng_state)

        # 保存最近一次 primitive/routing 诊断数据。
        self._last_primitive_stats: Dict[str, float] = {}

    def reset_common_init(self) -> None:
        """执行 E6 各残差出口的零初始化。"""

        # 最终 31 波段 residual 初始为 0，模型初始输出等于 bicubic base。
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

        # Gaussian value head 初始输出为 0，因此 gaussian_delta 初始为 0。
        self.gaussian_refine.reset_residual_init()

        # primitive residual 初始为 0，因此 E_g 初始等于 E0。
        nn.init.zeros_(self.primitive_residual[-1].weight)
        nn.init.zeros_(self.primitive_residual[-1].bias)

    def _encode_common(
        self, lr_hsi: torch.Tensor, hr_msi: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """完成 E6 在 routing 之前的公共编码与两条 HR 特征路径。"""

        # 目标空间大小直接读取 HR-MSI 的 H、W。
        target_size = hr_msi.shape[-2:]

        # 将 LR-HSI 双三次上采样，作为最终图像级残差的 base。
        base = F.interpolate(
            lr_hsi, size=target_size, mode="bicubic", align_corners=False
        )

        # LR-HSI 先投影为 dim 通道 latent。
        f_hsi = self.shallow_encoder1(lr_hsi)

        # HR-MSI 投影为 dim 通道 latent。
        f_msi = self.shallow_encoder2(hr_msi)

        # 依次通过 HSI 分支 ADCI。
        for layer in self.adci_hsi_layers:
            f_hsi = layer(f_hsi)

        # 依次通过 MSI 分支 ADCI。
        for layer in self.adci_msi_layers:
            f_msi = layer(f_msi)

        # 将 LR-HSI latent 上采样到 HR 网格，得到 F_H。
        f_h = F.interpolate(
            f_hsi, size=target_size, mode="bicubic", align_corners=False
        )

        # MSI 本来就在 HR 网格，直接记为 F_M。
        f_m = f_msi

        # 在通道维拼接 F_H 与 F_M，得到 [B,2C,H,W] 的 joint。
        joint = torch.cat([f_h, f_m], dim=1)

        # 主融合路径：joint → F_fused。
        f_fused = self.conv0(joint)

        # primitive 路径：joint → 基础 primitive embedding E0。
        e0 = self.primitive_input(joint)

        # 对 E0 加一个可学习修正，得到最终 primitive embedding E_g。
        e_g = e0 + self.primitive_residual(e0)

        # 返回 E6 后续 forward 所需的全部中间结果。
        return base, f_h, f_m, joint, f_fused, e_g

    def _record_primitive_stats(
        self,
        transport_x: torch.Tensor,
        value_x: torch.Tensor,
        pointwise_delta: Optional[torch.Tensor] = None,
        routing_delta: Optional[torch.Tensor] = None,
        routing: Optional[MSIGuidedHSILocalRouting] = None,
    ) -> None:
        """记录 primitive source 与 routing 的诊断统计，不改变预测。"""

        # 统计过程不需要梯度。
        with torch.no_grad():
            # 记录 geometry source 与 value source 的平均绝对幅值。
            stats = {
                "primitive_transport_source_abs_mean": float(
                    transport_x.detach().abs().mean()
                ),
                "primitive_value_source_abs_mean": float(
                    value_x.detach().abs().mean()
                ),
            }

            # 兼容其他 E5/E7 实验中的 pointwise correction；E6 通常不传。
            if pointwise_delta is not None:
                stats["primitive_pointwise_delta_abs_mean"] = float(
                    pointwise_delta.detach().abs().mean()
                )

            # E6 会记录 routing_delta 的平均绝对幅值。
            if routing_delta is not None:
                stats["primitive_routing_delta_abs_mean"] = float(
                    routing_delta.detach().abs().mean()
                )

            # 若 routing 已记录注意力统计，则合并进总统计字典。
            if routing is not None and routing.last_stats:
                stats.update(routing.last_stats)

            # 保存最近一次统计结果。
            self._last_primitive_stats = stats

    def collect_gs_stats(self) -> List[Dict[str, float]]:
        """把 Gaussian 和 primitive/routing 统计统一返回给训练日志。"""

        # 复制 Gaussian renderer 最近一次统计；没有时使用空字典。
        stats = dict(self.gaussian_refine.last_stats or {})

        # 合并 primitive 与 routing 统计。
        stats.update(self._last_primitive_stats)

        # 没有统计则返回空列表；否则统一标记为 hr_gaussian 层。
        return [] if not stats else [{"layer": "hr_gaussian", **stats}]


class GSFusion(HRFusedPrimitiveValueBase):
    """E6：使用 MSI keys 选择局部 HSI content，修正 Gaussian value。"""

    def __init__(
        self,
        dim: int = 64,
        num_bands: int = 31,
        num_msi: int = 3,
        adci_layers: int = 3,
        **_: object,
    ) -> None:
        # 先构造双分支编码、主融合、primitive、Gaussian 和输出解码器。
        super().__init__(dim, num_bands, num_msi, adci_layers)

        # 暂存随机状态，避免新增 routing 改变旧层初始化后的全局 RNG 契约。
        extra_rng_state = torch.get_rng_state()

        # 构造 MSI-guided HSI local routing；dim=64 时 routing_dim=16。
        self.routing = MSIGuidedHSILocalRouting(dim, routing_dim=dim // 4)

        # 恢复随机状态。
        torch.set_rng_state(extra_rng_state)

        # 用于日志或实验记录的人类可读结构说明。
        self.arch_summary = (
            "E6: E3-common ADCI/fusion/primitive embedding; geometry reads E_g; "
            "MSI keys route a 3x3 neighborhood of HSI values; routed correction "
            "is added to E_g value source before circular normalized splatting"
        )

        # 执行 E6 专用零初始化。
        self.reset_custom_init()

    def reset_custom_init(self) -> None:
        """初始化所有公共残差出口，并让 routing 初始输出为零。"""

        # 初始化 fc2、Gaussian value head、primitive residual。
        self.reset_common_init()

        # 初始化 routing out_proj，使 routing_delta 初始为 0。
        self.routing.reset_output_init()

    def forward(self, lr_hsi: torch.Tensor, hr_msi: torch.Tensor, sf=None):
        """执行 E6 完整前向传播。"""

        # 当前 E6 不显式使用倍率数值；倍率由输入尺寸比例隐式确定。
        del sf

        # 公共编码：得到 bicubic base、F_H、F_M、主融合特征和 primitive embedding。
        base, f_h, f_m, _joint, f_fused, e_g = self._encode_common(
            lr_hsi, hr_msi
        )

        # 以 HSI 为 query/value、MSI 为 key，从 3×3 邻域构造 HSI routing 修正。
        routing_delta = self.routing(f_h, f_m)

        # Gaussian value source = 基础 primitive embedding + 路由得到的 HSI 修正。
        value_source = e_g + routing_delta

        # geometry 读取 e_g；value head 读取 value_source；输出 Gaussian latent delta。
        gaussian_delta = self.gaussian_refine(
            transport_x=e_g, value_x=value_source
        )

        # 记录 primitive、routing 和 Gaussian 诊断统计。
        self._record_primitive_stats(
            e_g,
            value_source,
            routing_delta=routing_delta,
            routing=self.routing,
        )

        # 将 Gaussian latent residual 加到普通 HR 融合主特征上。
        refined = f_fused + gaussian_delta

        # 把 64 维 refined latent 解码为 31 波段图像 residual。
        residual = self.fc2(F.gelu(self.fc1(refined)))

        # 最终 HR-HSI = bicubic LR-HSI + 学习到的 31 波段 residual。
        return base + residual


# 规定从该文件使用 import * 时公开的符号。
__all__ = [
    "GSFusion",
    "HRAdaptiveGaussianResidualDualSource",
    "HRFusedPrimitiveValueBase",
    "MSIGuidedHSILocalRouting",
    "compute_loss",
    "sam_loss",
]
