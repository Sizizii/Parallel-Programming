#include <wb.h>
#include <math.h>

#define wbCheck(stmt)                                        \
  do                                                         \
  {                                                          \
    cudaError_t err = stmt;                                  \
    if (err != cudaSuccess)                                  \
    {                                                        \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err)); \
      wbLog(ERROR, "Failed to run stmt ", #stmt);            \
      return -1;                                             \
    }                                                        \
  } while (0)

//@@ Define any useful program-wide constants here
#define kernel_size 3 // kernelLength = 27 27**(1/3)
#define radius 1      // (3-1)/2
#define tile_width 3

//@@ Define constant memory for device kernel here
__constant__ float deviceKernel[kernel_size*kernel_size*kernel_size];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size)
{
  //@@ Insert kernel code here

  __shared__ float input_ds[tile_width*tile_width*tile_width];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int bz = blockIdx.z;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  int Width = bx * tile_width + tx;
  int Height = by * tile_width + ty;
  int Depth = bz * tile_width + tz;

  // calculate thread's idx in the block
  int idx = Depth * (x_size * y_size) + Height * x_size + Width;
  int tid = tz * (tile_width*tile_width) + ty * tile_width + tx;
  if((Width < x_size) && (Height < y_size) && (Depth < z_size)){
    input_ds[tid] = input[idx];
  }else{
    input_ds[tid] = 0;
  }
  __syncthreads();

  int this_tile_Start_point_x = blockIdx.x * blockDim.x;
  int this_tile_Start_point_y = blockIdx.y * blockDim.y;
  int this_tile_Start_point_z = blockIdx.z * blockDim.z;
  int next_tile_Start_point_x = (blockIdx.x+1) * blockDim.x;
  int next_tile_Start_point_y = (blockIdx.y+1) * blockDim.y;
  int next_tile_Start_point_z = (blockIdx.z+1) * blockDim.z;
  int input_start_point_x = Width - radius;
  int input_start_point_y = Height - radius;
  int input_start_point_z = Depth - radius;

  /*
  for (int i = 0; i< tile_width*tile_width*tile_width; i++){
    printf("%.7f  ", input_ds[i]);
  }
  */

  float Pvalue = 0;
  if((Width < x_size) && (Height < y_size) && (Depth < z_size)){
    for(int i = 0; i < kernel_size; i++){
      int N_index_z = input_start_point_z + i;
      for(int j = 0; j < kernel_size; j++){
        int N_index_y = input_start_point_y + j;
        for(int k = 0; k < kernel_size; k++){
          int N_index_x = input_start_point_x + k;
          int kernel_idx = i * (kernel_size * kernel_size) + j*kernel_size + k;
          if((N_index_x >= this_tile_Start_point_x) && (N_index_x < next_tile_Start_point_x) && (N_index_y >= this_tile_Start_point_y) && (N_index_y < next_tile_Start_point_y) && (N_index_z >= this_tile_Start_point_z) && (N_index_z < next_tile_Start_point_z)){
            int shared_idx = (tz + i - radius) * (tile_width*tile_width) + (ty + j - radius) * tile_width + (tx + k -radius);
            Pvalue += input_ds[shared_idx]*deviceKernel[kernel_idx];
          }
          else{
            if((N_index_x >= 0) && (N_index_x < x_size) && (N_index_y >= 0) && (N_index_y < y_size) && (N_index_z >= 0) && (N_index_z < z_size)){
              int global_idx = N_index_z * (x_size * y_size) + N_index_y * x_size + N_index_x;
              Pvalue += input[global_idx]*deviceKernel[kernel_idx];
            }
          }
        }
      }
    }
    output[idx] = Pvalue;
  }
  

}

int main(int argc, char *argv[])
{
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  int size_Input = (inputLength - 3) * sizeof(float);
  int size_Kernel = kernelLength * sizeof(float);
  int size_Output = (inputLength - 3) * sizeof(float);

  cudaMalloc((void **)&deviceInput, size_Input);
  cudaMalloc((void **)&deviceOutput, size_Output);

  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, hostInput+3, size_Input, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(deviceKernel, hostKernel, size_Kernel, 0, cudaMemcpyHostToDevice);

  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  // int mark_width = pow(kernelLength, 1.0 / 3);
  dim3 dimBlock(tile_width, tile_width, tile_width);
  dim3 dimGrid(ceil(x_size / (1.0 * tile_width)), ceil(y_size / (1.0 * tile_width)), ceil(z_size / (1.0 * tile_width)));
  conv3d<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);

  //@@ Launch the GPU kernel here

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(hostOutput+3, deviceOutput, size_Output, cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  
  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
