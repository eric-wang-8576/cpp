#include <stdio.h>
#include <iostream>
#include <random>


#define M 5
#define N 4
#define K 7

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> dis(0.0, 100.0);

struct Matrix {
    std::string name;
    float* m;
    int numRows;
    int numCols;

    Matrix(std::string n, int r, int c) : name(std::move(n)), numRows(r), numCols(c) {
        int numElements = numRows * numCols;
        m = (float*) malloc(numElements * sizeof(float));
        for (int i = 0; i < numElements; ++i) {
            m[i] = dis(gen);
        }
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

__host__ void hostMatMul(Matrix& A, Matrix& B, Matrix& C) {
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < K; ++c) {
            float sum = 0;
            for (int i = 0; i < N; ++i) {
                sum += A.m[r * N + i] * B.m[c + i * K];
            }
            C.m[r * K + c] = sum;
        }
    }
}

int main() {
    // Generate random A and B matrices
    Matrix A("A", M, N);
    Matrix B("B", N, K);
    Matrix C("C", M, K);

    A.print();
    B.print();

    hostMatMul(A, B, C);
    C.print();



    // Generate the correct matrix C


    // Generate matrix C based off of GPU CUDA kernel


    // Check that the implementation is correct
}
