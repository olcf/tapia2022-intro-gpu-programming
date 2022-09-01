#include "hip/hip_runtime.h"

#define cudaSuccess hipSuccess
#define cudaError_t hipError_t
#define cudaGetErrorString hipGetErrorString
#define cudaMalloc hipMalloc
#define cudaMemcpy hipMemcpy
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaGetLastError hipGetLastError
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaFree hipFree
