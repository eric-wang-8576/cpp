#include <iostream>
#include <random>
#include <iomanip>

#define DEBUG 0

#define N 10000

#define BLOCK_SIZE 512

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> dis(0.0, 100.0);

__host__ void populateArray(float* A, int size) {
    for (int i = 0; i < size; ++i) {
        A[i] = dis(gen);
    }
}

__host__ float hostReduction(float* A, int size) {
    float res = 0;
    for (int i = 0; i < size; ++i) {
        res += A[i];
    }
    return res;
}

__host__ bool compare(float expected, float actual) {
    std::cout << std::setprecision(10) 
        << "expected = " << expected
        << ", actual = " << actual << std::endl;
    if (std::abs(expected - actual) > 5) {
        return false;
    }
    return true;
}

__host__ void print(float* A, int size) {
    for (int i = 0; i < size; ++i) {
        std::cout << A[i] << ", ";
    }
    std::cout << std::endl;
}

__global__ void kernelReduction(float* input, float* output, int size) {
    int blockSize = blockDim.x;
    int tx = threadIdx.x;
    int bx = blockIdx.x;

    // Load values into shared memory 
    __shared__ float partialSum[BLOCK_SIZE];
    if (bx * blockSize + tx < size) {
        partialSum[tx] = input[bx * blockSize + tx];
    } else {
        partialSum[tx] = 0;
    }
    __syncthreads();

    // Perform reduction
    for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
        if (tx < stride) {
            partialSum[tx] += partialSum[tx + stride];
            __syncthreads();
        }
    }

    // Write final output value
    output[bx] = partialSum[0];
}

int main() {
    // Generate input
    float* input;
    cudaError_t err = cudaHostAlloc((void**) &input, N * sizeof(float), cudaHostAllocDefault);
    if (err != cudaSuccess) {
        std::cout << "Failed to allocate host-pinned memory" << std::endl;
        exit(1);
    }
    populateArray(input, N);

    // Calculate correct sum
    float expectedSum = hostReduction(input, N);
    
    // Calculate using GPU
    int numBlocks = (N - 1) / BLOCK_SIZE + 1;
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, numBlocks * sizeof(float));
    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Execute kernel
    dim3 dimGrid(numBlocks, 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);

    kernelReduction<<<dimGrid, dimBlock>>>(d_input, d_output, N);
    
    // Copy results back to host, and perform final reduction
    float* output = (float*) malloc(numBlocks * sizeof(float));
    cudaMemcpy(output, d_output, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    float actualSum = hostReduction(output, numBlocks);

    if constexpr(DEBUG) {
        print(output, numBlocks);
    }

    // Compare results
    if (compare(expectedSum, actualSum)) {
        std::cout << "Success!" << std::endl;
    } else {
        std::cout << "Failure!" << std::endl;
    }
}
