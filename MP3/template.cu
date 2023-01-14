
#include <wb.h>

#define wbCheck(stmt)                                                \
  do                                                                 \
  {                                                                  \
    cudaError_t err = stmt;                                          \
    if (err != cudaSuccess)                                          \
    {                                                                \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                    \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err)); \
      return -1;                                                     \
    }                                                                \
  } while (0)

// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns)
{
  //@@ Insert code to implement matrix multiplication here
  int tile_width = 32;
  __shared__ float subTileA[32][32];
  __shared__ float subTileB[32][32];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int Row = by * tile_width + ty;
  int Col = bx * tile_width + tx;
  float Cvalue = 0;

  for (int m = 0; m < ((numAColumns - 1) / tile_width + 1); ++m)
  {
    if ((Row < numARows) && ((m * tile_width + tx) < numAColumns))
    {
      subTileA[ty][tx] = A[Row * numAColumns + m * tile_width + tx];
    }
    else
    {
      subTileA[ty][tx] = 0;
    }

    if (((m * tile_width + ty) < numBRows) && (Col < numBColumns))
    {
      subTileB[ty][tx] = B[(m * tile_width + ty) * numBColumns + Col];
    }
    else
    {
      subTileB[ty][tx] = 0;
    }
    __syncthreads();

    if ((Row < numCRows) && (Col < numCColumns))
    {
      for (int k = 0; k < tile_width; ++k)
      {
        Cvalue += subTileA[ty][k] * subTileB[k][tx];
      }
    }
    __syncthreads();
  }
  if ((Row < numCRows) && (Col < numCColumns))
  {
    C[Row * numCColumns + Col] = Cvalue;
  }
}

int main(int argc, char **argv)
{
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(numCRows * numCColumns * sizeof(float));

  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  int size_A = numARows * numAColumns * sizeof(float);
  int size_B = numBRows * numBColumns * sizeof(float);
  int size_C = numCRows * numCColumns * sizeof(float);

  cudaMalloc((void **)&deviceA, size_A);
  cudaMalloc((void **)&deviceB, size_B);
  cudaMalloc((void **)&deviceC, size_C);

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, size_B, cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  int tile_width = 32;
  dim3 dimGrid(ceil((1.0 * numCColumns) / tile_width), ceil((1.0 * numCRows) / tile_width), 1);
  dim3 dimBlock(tile_width, tile_width, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiply<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, numARows,
                                        numAColumns, numBRows, numBColumns, numCRows, numCColumns);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, size_C, cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
