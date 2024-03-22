#include <iostream>
#include <random>

#define DEBUG 0

#define N 524288

#define BLOCK_SIZE 128

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<int> dis(0, 100);

__host__ void populateArray(int* A, int size) {
    for (int i = 0; i < size; ++i) {
        A[i] = dis(gen);
    }
}

__host__ void hostScan(int* A, int size, int* scan) {
    int sum = 0;
    for (int i = 0; i < size; ++i) {
        sum += A[i];
        scan[i] = sum;
    }
}

__host__ void print(int* A, int size) {
    std::cout << "[ ";
    for (int i = 0; i < size; ++i) {
        std::cout << A[i] << ", ";
    }
    std::cout << "]" << std::endl;
}

__host__ bool compare(int* expected, int size, int* actual) {
    for (int i = 0; i < size; ++i) {
        if (expected[i] != actual[i]) {
            std::cout << "Error on index " << i
                << ", expected = " << expected[i]
                << ", actual = " << actual[i] << std::endl;
            return false;
        }
    }
    return true;
}

__global__ void kernelScan(int* input, int* output, int* sums, int size) {
    int blockSize = blockDim.x;
    int tx = threadIdx.x;
    int bx = blockIdx.x;

    // Load values into shared memory 
    __shared__ int partialScan[BLOCK_SIZE];
    if (bx * blockSize + tx < size) {
        partialScan[tx] = input[bx * blockSize + tx];
    } else {
        partialScan[tx] = 0;
    }
    __syncthreads();

    // Reduction phase
    for (int stride = 1; stride < BLOCK_SIZE; stride <<= 1) {
        int idx = (tx + 1) * stride * 2 - 1;
        if (idx < BLOCK_SIZE) {
            partialScan[idx] += partialScan[idx - stride];
        }
        __syncthreads();
    }

    // Reverse phase
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        int idx = (tx + 1) * stride * 2 - 1;
        if (idx + stride < BLOCK_SIZE) {
            partialScan[idx + stride] += partialScan[idx];
        }
        __syncthreads();
    }

    // Write values to output and sums
    if (bx * blockSize + tx < size) {
        output[bx * blockSize + tx] = partialScan[tx];
    }

    if (tx == 0) {
        sums[bx] = partialScan[BLOCK_SIZE - 1];
    }
}

__global__ void kernelBlockSum(int* output, int* sumScan, int size) {
    int blockSize = blockDim.x;
    int tx = threadIdx.x;
    int bx = blockIdx.x;

    if (bx * blockSize + tx < size) {
        output[bx * blockSize + tx] += sumScan[bx];
    }
}

// Performs inclusive prefix sum on input, writes the result to output
__host__ void gpuScan(int* input, int size, int* output) {
    // Prepare Data
    int numBlocks = (size - 1) / BLOCK_SIZE + 1;
    int *d_input, *d_output, *d_sums;
    cudaMalloc(&d_input, size * sizeof(int));
    cudaMalloc(&d_output, size * sizeof(int));
    cudaMalloc(&d_sums, numBlocks * sizeof(int));
    cudaMemcpy(d_input, input, size * sizeof(int), cudaMemcpyHostToDevice);

    // Execute kernel, each block computes one value
    dim3 dimGrid(numBlocks, 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    kernelScan<<<dimGrid, dimBlock>>>(d_input, d_output, d_sums, size);

    // If we have more than one block, we need to recurse on sums array
    if (numBlocks != 1) {
        int sums[numBlocks];
        int sumScan[numBlocks];
        cudaMemcpy(sums, d_sums, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);

        // Recurse
        gpuScan(sums, numBlocks, sumScan);

        for (int i = numBlocks - 1; i > 0; --i) {
            sumScan[i] = sumScan[i - 1];
        }
        sumScan[0] = 0;

        // Copy sumScan into sums, then launch adding kernel 
        cudaMemcpy(d_sums, sumScan, numBlocks * sizeof(int), cudaMemcpyHostToDevice);
        kernelBlockSum<<<dimGrid, dimBlock>>>(d_output, d_sums, size);
    }
    
    // Copy results back to host
    cudaMemcpy(output, d_output, size * sizeof(int), cudaMemcpyDeviceToHost);
}

int main() {
    // Generate input
    int* input;
    cudaError_t err = cudaHostAlloc((void**) &input, N * sizeof(int), cudaHostAllocDefault);
    if (err != cudaSuccess) {
        std::cout << "Failed to allocate host-pinned memory" << std::endl;
        exit(1);
    }
    populateArray(input, N);

    if constexpr(DEBUG) {
        print(input, N);
    }

    // Calculate correct scan
    int expectedScan[N];
    hostScan(input, N, expectedScan);

    if constexpr(DEBUG) {
        print(expectedScan, N);
    }

    // Calculate GPU Scan
    int actualScan[N];
    gpuScan(input, N, actualScan);

    if constexpr(DEBUG) {
        print(actualScan, N);
    }

    // Compare results
    if (compare(expectedScan, N, actualScan)) {
        std::cout << "Success!" << std::endl;
    } else {
        std::cout << "Failure!" << std::endl;
    }
}
