
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <math.h>
#include "kernel.h";

// Kernel function to add the elements of two arrays
__global__
void Std_Dev(int n, float *x, float *y)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		y[i] = pow(x[i] - y[i], 2);

}

void cuda::FuncTest()
{
	int N = 1 << 25;
	float *x, *y;

	// Allocate Unified Memory – accessible from CPU or GPU
	cudaMallocManaged(&x, N * sizeof(float));
	cudaMallocManaged(&y, N * sizeof(float));

	// initialize x and y arrays on the host
	for (int i = 0; i < N; i++) {
		x[i] = i % 256;
		y[i] = 255 - i % 256;
	}

	// Run kernel on 1M elements on the GPU
	int blockSize = 256;
	int numBlocks = (N + blockSize - 1) / blockSize;
	Std_Dev << <numBlocks, blockSize >> > (N, x, y);

	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();

	//Standard deviation
	float percent = 0.0f;
	for (int i = 0; i < N; i++) {
		percent += y[i];
	}
	percent = sqrt(percent / (N - 1));
	std::cout << "Percent: " << percent;
	// Free memory
	cudaFree(x);
	cudaFree(y);
}
