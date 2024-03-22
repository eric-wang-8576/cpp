#include <iostream>
#include <random>
#include <iomanip>

#define DEBUG 0

#define N 10000000

#define BLOCK_SIZE 512
#define NUM_BLOCKS 25
#define MAX_VAL 100
#define NUM_BINS 10

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<int> dis(0, MAX_VAL);

__host__ void populateArray(int* A, int size) {
    for (int i = 0; i < size; ++i) {
        A[i] = dis(gen);
    }
}

__host__ void hostHistogram(int* A, int size, int* hist) {
    for (int i = 0; i < NUM_BINS; ++i) {
        hist[i] = 0;
    }

    for (int i = 0; i < size; ++i) {
        hist[A[i] % NUM_BINS]++;
    }
}

__host__ bool compare(int* expected, int* actual) {
    for (int i = 0; i < NUM_BINS; ++i) {
        if (expected[i] != actual[i]) {
            std::cout << "For index i = " << i 
                << ", expected = " << expected[i]
                << ", actual = " << actual[i] << std::endl;
            return false;
        }
    }
    return true;
}

__host__ void printHistogram(int* A) {
    for (int i = 0; i < NUM_BINS; ++i) {
        std::cout << A[i] << ", ";
    }
    std::cout << std::endl;
}

__global__ void kernelHistogram(int* input, int size, int* hist) {
    int blockSize = blockDim.x;
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    
    // Create private histogram
    __shared__ int privateHist[NUM_BINS];
    if (tx < NUM_BINS) {
        privateHist[tx] = 0;
    }
    __syncthreads();

    // Populate privateHist
    int i = bx * blockSize + tx;
    int stride = gridDim.x * blockSize; // number of threads
    while (i < size) {
        atomicAdd(&(privateHist[input[i] % NUM_BINS]), 1);
        i += stride;
    }

    __syncthreads();
    if (tx < NUM_BINS) {
        atomicAdd(&(hist[tx]), privateHist[tx]);
    }
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

    // Calculate correct hist
    int expectedHist[10];
    hostHistogram(input, N, expectedHist);
    if constexpr(DEBUG) {
        printHistogram(expectedHist);
    }
    
    // Calculate using GPU
    int *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_output, NUM_BINS * sizeof(int));
    cudaMemcpy(d_input, input, N * sizeof(int), cudaMemcpyHostToDevice);

    // Execute kernel
    dim3 dimGrid(NUM_BLOCKS, 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);

    kernelHistogram<<<dimGrid, dimBlock>>>(d_input, N, d_output);
    
    // Copy results back to host
    int actualHist[10];
    cudaMemcpy(actualHist, d_output, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);

    if constexpr(DEBUG) {
        printHistogram(actualHist);
    }

    // Compare results
    if (compare(expectedHist, actualHist)) {
        std::cout << "Success!" << std::endl;
    } else {
        std::cout << "Failure!" << std::endl;
    }
}
