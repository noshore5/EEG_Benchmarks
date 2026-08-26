#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int count = 0;
    cudaGetDeviceCount(&count);
    printf("CUDA devices: %d\n", count);

    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("GPU %d: %s\n", i, prop.name);
    }

    return 0;
}
