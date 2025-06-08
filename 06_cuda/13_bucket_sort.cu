#include <cstdio>
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void count_bucket(const int* key, int* bucket, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(&bucket[key[i]], 1); 
    }
}

__global__ void bucket_to_key(int* key, const int* bucket, int n, int range) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int count = 0;
        for (int val = 0; val < range; ++val) {
            count += bucket[val];
            if (i < count) {
                key[i] = val;
                break;
            }
        }
    }
}

int main() {
    int n = 50;
    int range = 5;
    std::vector<int> key(n);
    for (int i=0; i<n; i++) {
        key[i] = rand() % range;
        printf("%d ",key[i]);
    }
    printf("\n");

    int* d_key;
    int* d_bucket;
    cudaMalloc(&d_key, n * sizeof(int));
    cudaMalloc(&d_bucket, range * sizeof(int));

    cudaMemcpy(d_key, key.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_bucket, 0, range * sizeof(int)); 

    int threads = 128;
    int blocks = (n + threads - 1) / threads;
    count_bucket<<<blocks, threads>>>(d_key, d_bucket, n);
    cudaDeviceSynchronize();

    bucket_to_key<<<blocks, threads>>>(d_key, d_bucket, n, range);
    cudaDeviceSynchronize();

    cudaMemcpy(key.data(), d_key, n * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i=0; i<n; i++) printf("%d ",key[i]);
    printf("\n");

    cudaFree(d_key);
    cudaFree(d_bucket);
    return 0;
}
