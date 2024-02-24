#include <stdio.h>

// CUDA Kernel for Vector Addition
__global__ void vecAdd(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = 1024; // Size of vectors
    float *A, *B, *C; // Host vectors
    float *d_A, *d_B, *d_C; // Device (GPU) vectors

    // Allocate memory on the host
    A = (float *)malloc(N * sizeof(float));
    B = (float *)malloc(N * sizeof(float));
    C = (float *)malloc(N * sizeof(float));

    // Initialize vectors on the host
    for(int i = 0; i < N; i++) {
        A[i] = i * 5;
        B[i] = i * 2;
    }

    // Allocate memory on the device
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    // Copy host vectors to device
    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);

    // Execute the vector addition kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Display the result
    for(int i = 0; i < 10; i++) {
        printf("C[%d] = %f\n", i, C[i]);
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(A);
    free(B);
    free(C);

    return 0;
}
