#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16

__constant__ float Mc[64*64];

#define CUDA_ERROR(val) { check_err((val), __FILE__, __LINE__); }
inline void check_err(cudaError_t err, const char* const file, int line)
{
    if (err != cudaSuccess)
    {
        // std::cerr << cudaGetErrorString(err) << " " << file << ":" << line << std::endl;
        fprintf(stderr, "Got GPU Error: %s in file %s: line %d\n", cudaGetErrorString(err), file, line);
        exit(err);
    }
}

#define LAST_CUDA_ERROR() { checkLastErr(__FILE__, __LINE__); }
inline void checkLastErr(const char* const file, const int line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Got GPU Error: %s in file %s: line %d\n", cudaGetErrorString(err), file, line);
        exit(err);
    }
}

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) Mc[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int W_grid = ceil(1.0*Width/TILE_WIDTH);

    int m = blockIdx.x;
    int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;


    if ((h < Height_out) && (w < Width_out)){
      float acc = 0.0f;
      for (int c = 0; c < Channel; c++){
        for (int p = 0; p < K; p++){
          for (int q = 0; q < K; q++){
            acc += in_4d(blockIdx.z, c, h+p, w+q) * mask_4d(m, c, p, q);
          }
        }
      }
      out_4d(blockIdx.z, m, h, w) = acc;
    }
    

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    int size_Input = Height * Width * Batch * Channel;
    int size_Kernel = K * K * Map_out * Channel;
    int size_Output = Height_out * Width_out * Batch * Map_out;

    CUDA_ERROR(cudaMalloc((void **) device_input_ptr, size_Input * sizeof(float)));
    // CUDA_ERROR(cudaMalloc((void **) device_mask_ptr, size_Kernel * sizeof(float)));
    CUDA_ERROR(cudaMalloc((void **) device_output_ptr, size_Output * sizeof(float)));

    CUDA_ERROR(cudaMemcpy(*device_input_ptr, host_input, size_Input * sizeof(float), cudaMemcpyHostToDevice));
    // CUDA_ERROR(cudaMemcpy(*device_mask_ptr, host_mask, size_Kernel * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpyToSymbol(Mc, host_mask, size_Kernel*sizeof(float), 0, cudaMemcpyHostToDevice));
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    int W_grid = ceil(1.0*Width/TILE_WIDTH);
    int H_grid = ceil(1.0*Height/TILE_WIDTH);
    int Y = H_grid * W_grid;

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(Map_out, Y, Batch);

    conv_forward_kernel <<< gridDim, blockDim >>> (device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);

    LAST_CUDA_ERROR();
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int size_Output = Height_out * Width_out * Batch * Map_out;
    CUDA_ERROR(cudaMemcpy(host_output, device_output, size_Output * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_ERROR(cudaFree(device_input));
    // CUDA_ERROR(cudaFree(device_mask));
    CUDA_ERROR(cudaFree(device_output));
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
