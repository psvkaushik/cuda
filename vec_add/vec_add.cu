
#include <stdio.h>
__global__ void add_kernel(float *x, float *y, float *z, int N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N) {
        z[i] = x[i] + y[i];
    }
}

void vec_add(float *x, float *y, float *z, int N){

    float *x_d, *y_d, *z_d;
    int size = N * sizeof(float);


    // Allocate space in GPU
    cudaMalloc((void **)&x_d, size);
    cudaMalloc((void **)&y_d, size);
    cudaMalloc((void **)&z_d, size);

    // Copy from Host to Memory
    cudaMemcpy(x_d, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, size, cudaMemcpyHostToDevice);

    const unsigned int threadsPerBlock = 1024;
    const unsigned int blocksPerGrid = (N + threadsPerBlock -1)/threadsPerBlock;


    // Launch the kernel
    add_kernel<<< blocksPerGrid, threadsPerBlock>>>(x_d, y_d, z_d, N);

    // Copy from Memory to Host
    cudaMemcpy(z, z_d, size, cudaMemcpyDeviceToHost);

    //Free the pointers
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);


    
}
void vec_add_cpu(float *x, float *y,float *z, int N){
    for (unsigned int i=0; i<N; i++){
        z[i] = x[i] + y[i];
    }
}

int main(void){
    
    cudaDeviceSynchronize();
    const int n = 1 << 18;
    float A[n];
    float B[n];
    float C[n];

    clock_t start, stop;
    double cpu_time_used;
    double gpu_time_used;

    for (int i = 0; i < n; i++) {
    A[i] = rand();
    B[i] = rand();
    }
    // timing CPU operation

    start = clock();

    vec_add_cpu(A, B, C, n);

    stop = clock();
    cpu_time_used = ((double)(stop - start)) / CLOCKS_PER_SEC;
    printf("Time taken: %f milli-seconds\n", cpu_time_used*1000);

    // timing GPU operation
    start = clock();
    vec_add(A, B, C, n);
    cudaDeviceSynchronize();
    stop = clock();
    gpu_time_used = ((double)(stop - start)) / CLOCKS_PER_SEC;
    printf("Time taken: %f milli-seconds\n", gpu_time_used*1000);

    return 0;


}
