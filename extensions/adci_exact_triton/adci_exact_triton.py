"""Fused 3x3 value aggregation for ADCI using Triton CUDA kernels.

Softmax deliberately remains the native PyTorch operation.  The custom CUDA
path only replaces ``v_unfold -> multiply -> sum`` and reproduces unfold's
zero-padding semantics without materializing a nine-times-larger value tensor.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - non-CUDA development hosts
    triton = None
    tl = None


def _torch_weighted_reference(
    weights: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    batch, channels, height, width = values.shape
    neighbors = F.unfold(values, kernel_size=3, padding=1).view(
        batch, channels, 9, height, width
    )
    neighbors = neighbors.permute(0, 3, 4, 2, 1).contiguous()
    return torch.sum(neighbors * weights, dim=3).permute(0, 3, 1, 2).contiguous()


if triton is not None:

    @triton.jit
    def _weighted_aggregate_forward(
        weights_ptr,
        values_ptr,
        output_ptr,
        height: tl.constexpr,
        width: tl.constexpr,
        channels: tl.constexpr,
        block_channels: tl.constexpr,
    ):
        spatial_index = tl.program_id(0)
        channel_offsets = tl.arange(0, block_channels)
        channel_mask = channel_offsets < channels

        pixels_per_batch = height * width
        batch_index = spatial_index // pixels_per_batch
        pixel_index = spatial_index - batch_index * pixels_per_batch
        row = pixel_index // width
        col = pixel_index - row * width
        weight_base = spatial_index * 9 * channels + channel_offsets

        output = tl.zeros((block_channels,), tl.float32)
        for neighbor in tl.static_range(0, 9):
            delta_row = neighbor // 3 - 1
            delta_col = neighbor % 3 - 1
            neighbor_row = row + delta_row
            neighbor_col = col + delta_col
            spatial_mask = (
                (neighbor_row >= 0)
                & (neighbor_row < height)
                & (neighbor_col >= 0)
                & (neighbor_col < width)
            )
            weight = tl.load(
                weights_ptr + weight_base + neighbor * channels,
                mask=channel_mask,
                other=0.0,
            ).to(tl.float32)
            value_offset = (
                ((batch_index * channels + channel_offsets) * height + neighbor_row)
                * width
                + neighbor_col
            )
            value = tl.load(
                values_ptr + value_offset,
                mask=channel_mask & spatial_mask,
                other=0.0,
            ).to(tl.float32)
            output += weight * value

        output_offset = (
            ((batch_index * channels + channel_offsets) * height + row) * width
            + col
        )
        tl.store(output_ptr + output_offset, output, mask=channel_mask)


    @triton.jit
    def _weighted_aggregate_backward(
        weights_ptr,
        values_ptr,
        grad_output_ptr,
        grad_weights_ptr,
        grad_values_ptr,
        height: tl.constexpr,
        width: tl.constexpr,
        channels: tl.constexpr,
        block_channels: tl.constexpr,
    ):
        spatial_index = tl.program_id(0)
        channel_offsets = tl.arange(0, block_channels)
        channel_mask = channel_offsets < channels

        pixels_per_batch = height * width
        batch_index = spatial_index // pixels_per_batch
        pixel_index = spatial_index - batch_index * pixels_per_batch
        row = pixel_index // width
        col = pixel_index - row * width
        output_offset = (
            ((batch_index * channels + channel_offsets) * height + row) * width
            + col
        )
        grad_output = tl.load(
            grad_output_ptr + output_offset, mask=channel_mask, other=0.0
        ).to(tl.float32)
        weight_base = spatial_index * 9 * channels + channel_offsets

        # Gradient of every weight owned by this output location.
        for neighbor in tl.static_range(0, 9):
            delta_row = neighbor // 3 - 1
            delta_col = neighbor % 3 - 1
            neighbor_row = row + delta_row
            neighbor_col = col + delta_col
            spatial_mask = (
                (neighbor_row >= 0)
                & (neighbor_row < height)
                & (neighbor_col >= 0)
                & (neighbor_col < width)
            )
            value_offset = (
                ((batch_index * channels + channel_offsets) * height + neighbor_row)
                * width
                + neighbor_col
            )
            value = tl.load(
                values_ptr + value_offset,
                mask=channel_mask & spatial_mask,
                other=0.0,
            ).to(tl.float32)
            tl.store(
                grad_weights_ptr + weight_base + neighbor * channels,
                grad_output * value,
                mask=channel_mask,
            )

        # Gather the nine output contributions to this input value.  This is
        # deterministic and avoids atomic additions from neighboring outputs.
        grad_value = tl.zeros((block_channels,), tl.float32)
        for neighbor in tl.static_range(0, 9):
            delta_row = neighbor // 3 - 1
            delta_col = neighbor % 3 - 1
            center_row = row - delta_row
            center_col = col - delta_col
            center_mask = (
                (center_row >= 0)
                & (center_row < height)
                & (center_col >= 0)
                & (center_col < width)
            )
            center_spatial = (
                batch_index * pixels_per_batch + center_row * width + center_col
            )
            contributing_weight_offset = (
                center_spatial * 9 * channels
                + neighbor * channels
                + channel_offsets
            )
            contributing_output_offset = (
                ((batch_index * channels + channel_offsets) * height + center_row)
                * width
                + center_col
            )
            weight = tl.load(
                weights_ptr + contributing_weight_offset,
                mask=channel_mask & center_mask,
                other=0.0,
            ).to(tl.float32)
            contributing_grad = tl.load(
                grad_output_ptr + contributing_output_offset,
                mask=channel_mask & center_mask,
                other=0.0,
            ).to(tl.float32)
            grad_value += weight * contributing_grad

        tl.store(
            grad_values_ptr + output_offset,
            grad_value,
            mask=channel_mask,
        )


class _ADCIWeightedAggregate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        if triton is None:
            raise RuntimeError("Triton is not installed")
        if weights.dtype != torch.float32 or values.dtype != torch.float32:
            raise TypeError("ADCI Triton kernel currently requires float32")
        if not weights.is_cuda or not values.is_cuda:
            raise TypeError("ADCI Triton kernel requires CUDA tensors")
        if not weights.is_contiguous() or not values.is_contiguous():
            raise ValueError("weights and values must be contiguous")

        batch, channels, height, width = values.shape
        expected = (batch, height, width, 9, channels)
        if tuple(weights.shape) != expected:
            raise ValueError(f"weights shape {tuple(weights.shape)} != {expected}")

        output = torch.empty_like(values)
        block_channels = triton.next_power_of_2(channels)
        _weighted_aggregate_forward[(batch * height * width,)](
            weights,
            values,
            output,
            height=height,
            width=width,
            channels=channels,
            block_channels=block_channels,
            num_warps=4,
        )
        ctx.save_for_backward(weights, values)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        weights, values = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_weights = torch.empty_like(weights)
        grad_values = torch.empty_like(values)
        batch, channels, height, width = values.shape
        block_channels = triton.next_power_of_2(channels)
        _weighted_aggregate_backward[(batch * height * width,)](
            weights,
            values,
            grad_output,
            grad_weights,
            grad_values,
            height=height,
            width=width,
            channels=channels,
            block_channels=block_channels,
            num_warps=4,
        )
        return grad_weights, grad_values


def adci_exact_aggregate(
    scores: torch.Tensor,
    values: torch.Tensor,
    *,
    use_triton: bool = True,
) -> torch.Tensor:
    """Apply native softmax followed by exact 3x3 zero-padded aggregation."""
    weights = F.softmax(scores, dim=3)
    if (
        use_triton
        and triton is not None
        and weights.is_cuda
        and values.is_cuda
        and weights.dtype == torch.float32
        and values.dtype == torch.float32
    ):
        return _ADCIWeightedAggregate.apply(weights.contiguous(), values.contiguous())
    return _torch_weighted_reference(weights, values)


__all__ = ["adci_exact_aggregate"]
