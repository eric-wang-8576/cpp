#include <stdio.h>
#include <iostream>
#include <random>
#include <assert.h>
#include <iomanip>
#include <chrono>
#include <cuda_runtime.h>

// TODO: Organize tiled.cu and naive.cu into different files, modify CMakeLists.txt

#define DEBUG 0

#define M 20000
#define N 10000
#define K 15000

#define TILE_WIDTH 16

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> dis(0.0, 100.0);

struct Matrix {
    std::string name;
    float* m;
    int numBytes;
    int numRows;
    int numCols;

    Matrix(std::string n, int r, int c, bool populate) : name(std::move(n)), numRows(r), numCols(c) {
        int numElements = numRows * numCols;
        m = (float*) malloc(numElements * sizeof(float));
        if (populate) {
            for (int i = 0; i < numElements; ++i) {
                m[i] = dis(gen);
            }
        }
        numBytes = numElements * sizeof(float);
    }

    __host__ void print() {
        std::cout << "Matrix " << name << ":" << std::endl;
        for (int i = 0; i < numRows; ++i) {
            std::cout << "\t";
            for (int j = 0; j < numCols; ++j) {
                std::cout << m[i * numCols + j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
};

__host__ bool compare(const Matrix& C, const Matrix& CGPU) {
    assert(C.numRows == CGPU.numRows && C.numCols == CGPU.numCols);
    for (int r = 0; r < C.numRows; ++r) {
        for (int c = 0; c < C.numCols; ++c) {
            int idx = r * C.numCols + c;
            if (std::abs(C.m[idx] - CGPU.m[idx]) > 5) {
                std::cout << std::setprecision(10) 
                    << "Failure on r = " << r
                    << ", c = " << c
                    << ": " << C.m[idx] << " != " << CGPU.m[idx] << std::endl;
                return false;
            }
        }
    }
    return true;
}

__global__ void naiveMatMul(int m, int n, int k, float* A, float* B, float* C) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < m && c < k) {
        float sum = 0.0;
        for (int i = 0; i < N; ++i) {
            sum += A[r * n + i] * B[c + i * k];
        }
        C[r * k + c] = sum;
    }
}

// Multiples A and B, puts the result into C
__host__ void runNaive(Matrix& A, Matrix& B, Matrix& C) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start); 
    cudaEventCreate(&stop);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, A.numBytes);
    cudaMalloc(&d_B, B.numBytes);
    cudaMalloc(&d_C, C.numBytes);
    cudaMemcpy(d_A, A.m, A.numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.m, B.numBytes, cudaMemcpyHostToDevice);

    // Execute kernel
    dim3 dimGrid((K - 1)/TILE_WIDTH + 1, (M - 1)/TILE_WIDTH + 1, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    cudaEventRecord(start);

    naiveMatMul<<<dimGrid, dimBlock>>>(M, N, K, d_A, d_B, d_C);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Time for GPU naive execution: " << ms << " milliseconds." << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy results back to host
    cudaMemcpy(C.m, d_C, C.numBytes, cudaMemcpyDeviceToHost);
}

__global__ void tiledMatMul(int m, int n, int k, float* A, float* B, float* C) {
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    float CVal = 0;

    for (int tile = 0; tile < (n - 1) / TILE_WIDTH + 1; ++tile) {
        // Load into tile A
        if (row < m && tile * TILE_WIDTH + tx < n) {
            ds_A[ty][tx] = A[row * n + (tile * TILE_WIDTH + tx)];
        } else {
            ds_A[ty][tx] = 0;
        }

        // Load into tile B
        if (tile * TILE_WIDTH + ty < n && col < k) {
            ds_B[ty][tx] = B[(ty + tile * TILE_WIDTH) * k + col];
        } else {
            ds_B[ty][tx] = 0;
        }

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i) {
            CVal += ds_A[ty][i] * ds_B[i][tx];
        }
        __syncthreads();
    }

    if (row < m && col < k) {
        C[row * k + col] = CVal;
    }
}

__host__ void runTiled(Matrix& A, Matrix& B, Matrix& C) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start); 
    cudaEventCreate(&stop);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, A.numBytes);
    cudaMalloc(&d_B, B.numBytes);
    cudaMalloc(&d_C, C.numBytes);
    cudaMemcpy(d_A, A.m, A.numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.m, B.numBytes, cudaMemcpyHostToDevice);

    // Execute kernel
    dim3 dimGrid((K - 1)/TILE_WIDTH + 1, (M - 1)/TILE_WIDTH + 1, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    cudaEventRecord(start);
    tiledMatMul<<<dimGrid, dimBlock>>>(M, N, K, d_A, d_B, d_C);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Time for GPU tiled execution: " << ms << " milliseconds." << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy results back to host
    cudaMemcpy(C.m, d_C, C.numBytes, cudaMemcpyDeviceToHost);
}

int main() {
    // Generate random A and B matrices
    Matrix A("A", M, N, true);
    Matrix B("B", N, K, true);
    Matrix CNaiveTest("C Naive", M, K, false);
    Matrix CTiledTest("C Tiled", M, K, false);

    if constexpr(DEBUG) {
        A.print();
        B.print();
    }

    runNaive(A, B, CNaiveTest);
    if constexpr(DEBUG) {
        CNaiveTest.print();
    }

    runTiled(A, B, CTiledTest);
    if constexpr(DEBUG) {
        CTiledTest.print();
    }

    // Check that the implementation is correct
    if (compare(CNaiveTest, CTiledTest) == true) {
        std::cout << "Success!" << std::endl;
    }
}
