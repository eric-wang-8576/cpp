#include <iostream>
#include <random>
#include <chrono>

#define DEBUG 0

// Assume kernel dimension is odd
#define KERNEL_DIM 11
#define IMG_DIM 512

#define KERNEL_SIZE (KERNEL_DIM * KERNEL_DIM)
#define IMG_SIZE (IMG_DIM * IMG_DIM)

#define BLOCK_DIM 16
#define O_TILE_DIM (BLOCK_DIM - (KERNEL_DIM - 1))

__constant__ int kernel[KERNEL_SIZE];

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<int> dis(0, 100);

__host__ void populateArray(int* A, int size) {
    for (int i = 0; i < size; ++i) {
        A[i] = dis(gen);
    }
}

__host__ bool compare(int* expected, int* actual, int size) {
    for (int i = 0; i < size; ++i) {
        if (expected[i] != actual[i]) {
            std::cout << "Difference on i = " << i
                << ", expected = " << expected[i]
                << ", actual = " << actual[i] << std::endl;
            return false;
        }
    }
    return true;
}

__host__ void print(int* A, int dim) {
    for (int i = 0; i < dim; ++i) {
        std::cout << "[ ";
        for (int j = 0; j < dim; ++j) {
            std::cout << A[i * dim + j] << ", ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << std::endl;
}

__host__ void hostConvolution(int* img, int imgDim, int* kernel, int kernelDim, int* res) {
    int offset = kernelDim / 2;
    for (int r = 0; r < imgDim; ++r) {
        for (int c = 0; c < imgDim; ++c) {
            // Calculate res[r][c]
            int val = 0;
            for (int kr = 0; kr < kernelDim; ++kr) {
                for (int kc = 0; kc < kernelDim; ++kc) {
                    int imgR = r - offset + kr;
                    int imgC = c - offset + kc;
                    if (0 <= imgR && imgR < imgDim && 0 <= imgC && imgC < imgDim) {
                        val += img[imgR * imgDim + imgC] * kernel[kr * kernelDim + kc];
                    }
                }
            }
            res[r * imgDim + c] = val;
        }
    }
}

/* The input is divided into tiles of size O_TILE_DIM, and each block writes to 
 * an area of O_TILE_DIM * O_TILE_DIM. The block has enough threads to read all the inputs
 * necessary to compute this area
 */
__global__ void kernelConvolution(int* input, int* output, int imgDim, int kernelDim) {
    __shared__ int inputTile[BLOCK_DIM * BLOCK_DIM];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int offset = kernelDim / 2;

    int row_o = by * O_TILE_DIM + ty;
    int col_o = bx * O_TILE_DIM + tx;
    int row_i = row_o - offset;
    int col_i = col_o - offset;

    // Load input tile
    if (0 <= row_i && row_i < imgDim && 0 <= col_i && col_i < imgDim) {
        inputTile[ty * BLOCK_DIM + tx] = input[row_i * imgDim + col_i];
    } else {
        inputTile[ty * BLOCK_DIM + tx] = 0;
    }
    __syncthreads();

    // Compute output values
    if (ty < O_TILE_DIM && row_o < imgDim && tx < O_TILE_DIM && col_o < imgDim) {
        int val = 0;
        for (int kr = 0; kr < kernelDim; ++kr) {
            for (int kc = 0; kc < kernelDim; ++kc) {
                int tr = ty + kr;
                int tc = tx + kc;
                val += inputTile[tr * BLOCK_DIM + tc] * kernel[kr * kernelDim + kc];
            }
        }

        output[row_o * imgDim + col_o] = val;
    }
}

__host__ void gpuConvolution(int* img, int imgDim, int* localKernel, int kernelDim, int* actual) {
    // Copy to low latency constant device memory
    cudaMemcpyToSymbol(kernel, localKernel, KERNEL_SIZE * sizeof(int));
    
    int *d_input, *d_output;
    cudaMalloc(&d_input, IMG_SIZE * sizeof(int));
    cudaMalloc(&d_output, IMG_SIZE * sizeof(int));
    cudaMemcpy(d_input, img, IMG_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    int gridDims = (imgDim - 1) / O_TILE_DIM + 1;
    dim3 dimGrid(gridDims, gridDims, 1);
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM, 1);

    kernelConvolution<<<dimGrid, dimBlock>>>(d_input, d_output, imgDim, kernelDim);
    cudaMemcpy(actual, d_output, IMG_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
}

int main() {
    // Generate img
    int* img;
    cudaError_t err = cudaHostAlloc((void**) &img, IMG_SIZE * sizeof(int), cudaHostAllocDefault);
    if (err != cudaSuccess) {
        std::cout << "Failed to allocate host-pinned memory" << std::endl;
        exit(1);
    }
    populateArray(img, IMG_SIZE);

    // Generate kernel
    int localKernel[KERNEL_SIZE];
    populateArray(localKernel, KERNEL_SIZE);

    if constexpr(DEBUG) {
        print(img, IMG_DIM);
        print(localKernel, KERNEL_DIM);
    }

    // Compute correct result
    auto start = std::chrono::high_resolution_clock::now();

    int expected[IMG_SIZE];
    hostConvolution(img, IMG_DIM, localKernel, KERNEL_DIM, expected);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "CPU Convolution Took " << duration.count() << " ms." << std::endl;

    if constexpr(DEBUG) {
        print(expected, IMG_DIM);
    }

    // Compute using GPU
    start = std::chrono::high_resolution_clock::now();

    int actual[IMG_SIZE];
    gpuConvolution(img, IMG_DIM, localKernel, KERNEL_DIM, actual);

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "GPU Convolution Took " << duration.count() << " ms." << std::endl;

    if constexpr(DEBUG) {
        print(actual, IMG_DIM);
    }

    // Compare results
    if (compare(expected, actual, IMG_SIZE)) {
        std::cout << "Success!" << std::endl;
    } else {
        std::cout << "Failure!" << std::endl;
    }
}
