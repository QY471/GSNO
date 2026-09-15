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
    float stdXY = stdX * stdY;
    float stdX2 = stdX * stdX;
    float stdY2 = stdY * stdY;
    float meanX = means[gaussianIdx].x;
    float meanY = means[gaussianIdx].y;
    float rho = rhos[gaussianIdx];
    float beta = 1 - rho * rho;
    float betaRoot = sqrt(beta);
    float exp1 = -1 / (2 * beta);

    float rH = adaptiveWindow ? sigmaRadius * stdY : rasterRatio * sH / scaleFactor;
    float rW = adaptiveWindow ? sigmaRadius * stdX : rasterRatio * sW / scaleFactor;

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

            // Finish computing Eq. 1
            float deltaX2 = deltaX * deltaX;
            float deltaY2 = deltaY * deltaY;
            float deltaXY = deltaX * deltaY;
            float exp2 = deltaX2 / stdX2 + deltaY2 / stdY2 - 2 * rho * deltaXY / stdXY;
            float f = 1 / (2 * M_PI * stdXY * betaRoot) * exp(exp1 * exp2);
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
