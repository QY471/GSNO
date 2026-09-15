#include "forward.h"
#include "config.h"

#include <math.h>
#include <stdio.h>

__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
    renderCUDA(
        const int numGaussians,
        const float* __restrict__ opacity,
        const float2* __restrict__ means,
        const float2* __restrict__ stds,
        const float* __restrict__ rhos,
        const float* __restrict__ colors,
        const float* __restrict__ keys,
        const float* __restrict__ gamma,
        const int keyDim,
        const int sH,
        const int sW,
        const float scaleFactor,
        const float rasterRatio,
        const bool adaptiveWindow,
        const float sigmaRadius,
        const int numChannels,
        float* __restrict__ outImage)
{
    // Make this kernel to be per gaussian instead of per pixel

    // Get all Gaussian Parameters and necessary variables for Eq. 1 and 2
    int gaussianIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (gaussianIdx >= numGaussians) return;

    float alfa = opacity[gaussianIdx];
    float stdX = stds[gaussianIdx].x;
    float stdY = stds[gaussianIdx].y;
    float meanX = means[gaussianIdx].x;
    float meanY = means[gaussianIdx].y;

    const float halfCell = 0.5f / scaleFactor;
    float rH = adaptiveWindow ? sigmaRadius * stdY + halfCell : rasterRatio * sH / scaleFactor;
    float rW = adaptiveWindow ? sigmaRadius * stdX + halfCell : rasterRatio * sW / scaleFactor;

    // Restrict traversal to a conservative bounding box around the existing
    // global support window. Keep the original in-loop test below so the set
    // of contributing pixels remains unchanged at floating-point boundaries.
    const float centerRow = scaleFactor * meanY;
    const float centerCol = scaleFactor * meanX;
    const float radiusRows = scaleFactor * rH;
    const float radiusCols = scaleFactor * rW;
    const int rowBegin = max(0, (int)floorf(centerRow - radiusRows) - 1);
    const int rowEnd = min(sH, (int)ceilf(centerRow + radiusRows) + 2);
    const int colBegin = max(0, (int)floorf(centerCol - radiusCols) - 1);
    const int colEnd = min(sW, (int)ceilf(centerCol + radiusCols) + 2);

    for (int rows = rowBegin; rows < rowEnd; rows++) {
        for (int cols = colBegin; cols < colEnd; cols++) {
            // Get pixel coordinates and check if pixel is within the Gaussian influence
            float deltaX = (cols/scaleFactor - meanX);
            float deltaY = (rows/scaleFactor - meanY);
            if (fabs(deltaX) >= rW || fabs(deltaY) >= rH) continue;

            // Integrate the axis-aligned Gaussian over the output pixel cell.
            // Bounds are clipped to the requested sigma support, so a small
            // Gaussian can overlap a neighbouring cell without requiring the
            // neighbour's centre to lie inside the support.
            const float rawLowerX = (deltaX - halfCell) / stdX;
            const float rawUpperX = (deltaX + halfCell) / stdX;
            const float rawLowerY = (deltaY - halfCell) / stdY;
            const float rawUpperY = (deltaY + halfCell) / stdY;
            const float lowerX = fmaxf(rawLowerX, -sigmaRadius);
            const float upperX = fminf(rawUpperX, sigmaRadius);
            const float lowerY = fmaxf(rawLowerY, -sigmaRadius);
            const float upperY = fminf(rawUpperY, sigmaRadius);
            if (lowerX >= upperX || lowerY >= upperY) continue;

            constexpr float invSqrtTwo = 0.7071067811865475f;
            const float massX = 0.5f * (
                erff(upperX * invSqrtTwo) - erff(lowerX * invSqrtTwo));
            const float massY = 0.5f * (
                erff(upperY * invSqrtTwo) - erff(lowerY * invSqrtTwo));
            const float f = massX * massY;
            const int targetIdx = rows * sW + cols;
            float cosine = 0.0f;
            for (int k = 0; k < keyDim; k++) {
                cosine += keys[gaussianIdx * keyDim + k] *
                          keys[targetIdx * keyDim + k];
            }
            float fAlfa = f * alfa * expf(gamma[0] * cosine);

            // Eq. 2
            for (int c = 0; c < numChannels; c++) {
                int idx = rows * sW * numChannels + cols * numChannels + c;
                float color = colors[gaussianIdx * numChannels + c];
                atomicAdd(&outImage[idx], fAlfa * color);
            }
        }
    }
}

void FORWARD::render(
    const dim3 grid, dim3 block,
    const int numGaussians,
    const float* __restrict__ opacity,
    const float2* __restrict__ means,
    const float2* __restrict__ stds,
    const float* __restrict__ rhos,
    const float* __restrict__ colors,
    const float* __restrict__ keys,
    const float* __restrict__ gamma,
    const int keyDim,
    const int sH,
    const int sW,
    const float scaleFactor,
    const float rasterRatio,
    const bool adaptiveWindow,
    const float sigmaRadius,
    const int numChannels,
    float* __restrict__ outImage)
{
    renderCUDA<<<grid, block>>>(
        numGaussians,
        opacity,
        means,
        stds,
        rhos,
        colors,
        keys,
        gamma,
        keyDim,
        sH, sW,
        scaleFactor,
        rasterRatio,
        adaptiveWindow,
        sigmaRadius,
        numChannels,
        outImage);
}
