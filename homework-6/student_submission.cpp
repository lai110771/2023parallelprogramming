#include "dgemm.h"
#include <cstdio>
#include <cstdlib>
#include <immintrin.h>

void dgemm(float alpha, const float *a, const float *b, float beta, float *c) {
    const __m256 alphav = _mm256_set1_ps(alpha); 
    int ub = MATRIX_SIZE - (MATRIX_SIZE % 8);
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            c[i * MATRIX_SIZE + j] *= beta;
    	    float buffer[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            __m256 partial_sum = _mm256_set1_ps(0);
	        float curr_add = 0;
            for (int k = 0; k < ub; k+=8) {
                //c[i * MATRIX_SIZE + j] += alpha * a[i * MATRIX_SIZE + k] * b[j * MATRIX_SIZE + k];
                __m256 av = _mm256_loadu_ps(a + i * MATRIX_SIZE + k);
	    	    __m256 bv = _mm256_loadu_ps(b + j * MATRIX_SIZE + k);
		        __m256 qv = _mm256_mul_ps(av, bv);
		        __m256 tem = _mm256_mul_ps(alphav, qv);
                partial_sum = _mm256_add_ps(partial_sum, tem);
                //_mm256_store_ps(buffer, tem);
	    	    //for (int q = 0; q < 8; q++) {
	       	    //    curr_add += buffer[q];
	            //}
            }
            _mm256_store_ps(buffer, partial_sum);
            for (int q = 0; q < 8; q++) {
	       	    c[i * MATRIX_SIZE + j] += buffer[q];
	        }

            for (int p = ub; p < MATRIX_SIZE; p++){
                c[i * MATRIX_SIZE + j] += alpha * a[i * MATRIX_SIZE + p] * b[j * MATRIX_SIZE + p];
            }
        }
    }
}

int main(int, char **) {
    float alpha, beta;

    // mem allocations
    int mem_size = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);
    auto a = (float *) malloc(mem_size);
    auto b = (float *) malloc(mem_size);
    auto c = (float *) malloc(mem_size);

    // check if allocated
    if (nullptr == a || nullptr == b || nullptr == c) {
        printf("Memory allocation failed\n");
        if (nullptr != a) free(a);
        if (nullptr != b) free(b);
        if (nullptr != c) free(c);
        return 0;
    }

    generateProblemFromInput(alpha, a, b, beta, c);

    std::cerr << "Launching dgemm step." << std::endl;
    // matrix-multiplication
    dgemm(alpha, a, b, beta, c);

    outputSolution(c);

    free(a);
    free(b);
    free(c);
    return 0;
}
