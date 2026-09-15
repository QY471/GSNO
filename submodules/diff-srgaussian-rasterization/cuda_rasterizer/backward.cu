#include "backward.h"
#include "config.h"

#include <math.h>

__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
    renderCUDA(
        const int numGaussians,
        const float* __restrict__ opacity,
        const float2* __restrict__ means,
        const float2* __restrict__ stds,
        const float* __restrict__ rhos,
        const float* __restrict__ colors,
        const float* __restrict__ grad_output,
        const int sH, int sW,
        const float scaleFactor,
        const float rasterRatio,
        const int numChannels,
        float* __restrict__ dL_dopacity,
        float2* __restrict__ dL_dmeans,
        float2* __restrict__ dL_dstds,
        float* __restrict__ dL_drhos,
        float* __restrict__ dL_dcolors
    )
{
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

    // Initialize gaussian gradient values
    float grad_opacity = 0;
    float grad_meanX = 0;
    float grad_meanY = 0;
    float grad_stdX = 0;
    float grad_stdY = 0;
    float grad_rho = 0;

    float rH = rasterRatio * sH / scaleFactor;
    float rW = rasterRatio * sW / scaleFactor;

    // Iterate through all pixels of the image
    for (int rows = 0; rows < sH; rows++) {
        for (int cols = 0; cols < sW; cols++) {
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

            // Now compute gradients
            for (int c = 0; c < numChannels; c++) {
                int idx = rows * sW * numChannels + cols * numChannels + c;
                float grad = grad_output[idx];

                if (grad != 0) {
                    float color = colors[gaussianIdx * numChannels + c];

                    // Opacity grad
                    grad_opacity += grad * f * color;
                    // Mean X grad
                    float dL_df = grad * alfa * color;
                    float df_dmeanx = f * exp1 * ((-2 * deltaX / stdX2) + (2 * rho * deltaY / stdXY));
                    grad_meanX += dL_df * df_dmeanx;
                    // Mean Y grad
                    float df_dmeany = f * exp1 * ((-2 * deltaY / stdY2) + (2 * rho * deltaX / stdXY));
                    grad_meanY += dL_df * df_dmeany;
                    // Std X grad
                    float df_dstdx = -f / stdX + f * exp1 * (-2 * deltaX2 / stdX2 / stdX + 2 * rho * deltaXY / stdXY / stdX);
                    grad_stdX += dL_df * df_dstdx;
                    // Std Y grad
                    float df_dstdy = -f / stdY + f * exp1 * (-2 * deltaY2 / stdY2 / stdY + 2 * rho * deltaXY / stdXY / stdY);
                    grad_stdY += dL_df * df_dstdy;
                    // Rho grad
                    float df_drho = f * ((rho / beta) - (rho * exp2 / (beta * beta)) - (exp1 * 2 * deltaXY / stdXY));
                    grad_rho += dL_df * df_drho;
                    // Color grad
                    dL_dcolors[gaussianIdx * numChannels + c] += grad * alfa * f;
                }
            }
        }
    }
    dL_dopacity[gaussianIdx] = grad_opacity;
    dL_dmeans[gaussianIdx].x = grad_meanX;
    dL_dmeans[gaussianIdx].y = grad_meanY;
    dL_dstds[gaussianIdx].x = grad_stdX;
    dL_dstds[gaussianIdx].y = grad_stdY;
    dL_drhos[gaussianIdx] = grad_rho;
}

void BACKWARD::render(
    const dim3 grid, dim3 block,
    const int numGaussians,
    const float* __restrict__ opacity,
    const float2* __restrict__ means,
    const float2* __restrict__ stds,
    const float* __restrict__ rhos,
    const float* __restrict__ colors,
    const float* __restrict__ grad_output,
    const int sH, int sW,
    const float scaleFactor,
    const float rasterRatio,
    const int numChannels,
    float* __restrict__ dL_dopacity,
    float2* __restrict__ dL_dmeans,
    float2* __restrict__ dL_dstds,
    float* __restrict__ dL_drhos,
    float* __restrict__ dL_dcolors
)
{
    renderCUDA<<<grid, block>>>(
        numGaussians,
        opacity,
        means,
        stds,
        rhos,
        colors,
        grad_output,
        sH, sW,
        scaleFactor,
        rasterRatio,
        numChannels,
        dL_dopacity,
        dL_dmeans,
        dL_dstds,
        dL_drhos,
        dL_dcolors
    );
}
