#include <stdio.h>
#include <iostream>
#include <random>
#include <assert.h>
#include <iomanip>
#include <chrono>

#define DEBUG 0

#define M 2000
#define N 1800
#define K 1900

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

__host__ void hostMatMul(const Matrix& A, const Matrix& B, const Matrix& C) {
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < K; ++c) {
            float sum = 0.0;
            for (int i = 0; i < N; ++i) {
                sum += A.m[r * N + i] * B.m[c + i * K];
            }
            C.m[r * K + c] = sum;
        }
    }
}

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

__global__ void kernelMatMul(int m, int n, int k, float* A, float* B, float* C) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;

    if (r < m && c < k) {
        float sum = 0.0;
        for (int i = 0; i < N; ++i) {
            sum += A[r * n + i] * B[c + i * k];
        }
        C[r * k + c] = sum;
    }
}

int main() {
    // Generate random A and B matrices
    Matrix A("A", M, N, true);
    Matrix B("B", N, K, true);
    Matrix C("C Correct", M, K, false);
    Matrix CGPU("C Generated", M, K, false);

    if constexpr(DEBUG) {
        A.print();
        B.print();
    }

    // Generate the correct matrix C
    auto start = std::chrono::steady_clock::now();

    hostMatMul(A, B, C);

    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time for CPU execution: " << elapsed.count() << " milliseconds." << std::endl;
    if constexpr(DEBUG) {
        C.print();
    }

    // Allocate row-major order matrices on the device
    start = std::chrono::steady_clock::now();

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, A.numBytes);
    cudaMalloc(&d_B, B.numBytes);
    cudaMalloc(&d_C, C.numBytes);
    cudaMemcpy(d_A, A.m, A.numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.m, B.numBytes, cudaMemcpyHostToDevice);

    // Execute kernel
    dim3 dimGrid((M - 1)/TILE_WIDTH + 1, (K - 1)/TILE_WIDTH + 1, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    kernelMatMul<<<dimGrid, dimBlock>>>(M, N, K, d_A, d_B, d_C);
    
    // Copy results back to host
    cudaMemcpy(CGPU.m, d_C, C.numBytes, cudaMemcpyDeviceToHost);
    if constexpr(DEBUG) {
        CGPU.print();
    }

    end = std::chrono::steady_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time for GPU execution: " << elapsed.count() << " milliseconds." << std::endl;



    // Check that the implementation is correct
    if (compare(C, CGPU) == true) {
        std::cout << "Success!" << std::endl;
    }
}
