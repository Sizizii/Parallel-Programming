// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 512

//@@ insert code here
__global__ void f2char(float* input, unsigned char* output, int size){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    output[idx] = (unsigned char) (255 * input[idx]);
  }
}

__global__ void rgb2gray(unsigned char* input, unsigned char* output, int size){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    output[idx] = (unsigned char) (0.21 * input[3*idx] + 0.71 * input[3*idx+1] + 0.07 * input[3*idx+2]);
  }
}

__global__ void histo_kernel(unsigned char* input, unsigned int* output, int size){
  __shared__ unsigned int histo_private[HISTOGRAM_LENGTH];

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadIdx.x < HISTOGRAM_LENGTH)
  {
    histo_private[threadIdx.x] = 0;
  }
  __syncthreads();

  if (idx < size)
  {
    atomicAdd(&(histo_private[input[idx]]), 1);
  }
  __syncthreads();

  if (threadIdx.x < HISTOGRAM_LENGTH)
  {
    atomicAdd(&(output[threadIdx.x]), histo_private[threadIdx.x]);
  }
  
}

__global__ void cdf_kernel(unsigned int* input, float* output, int size){
  __shared__ float cdf[HISTOGRAM_LENGTH];

  /* Load elements from global */
  int tid = threadIdx.x;
  
  if(tid < HISTOGRAM_LENGTH){  cdf[tid] = input[tid]; }
  if((tid + blockDim.x) < HISTOGRAM_LENGTH){  cdf[tid + blockDim.x] = input[tid + blockDim.x]; }

  /* first scan */
  int stride = 1;
  while (stride < (2 * blockDim.x))
  {
    __syncthreads();
    int index = (tid + 1) * stride * 2 - 1;
    if ((index < HISTOGRAM_LENGTH) && ((index-stride) >= 0))
    {
      cdf[index] += cdf[index-stride];
    }
    stride *= 2;
  }

  /* second scan */
  stride = HISTOGRAM_LENGTH/4;
  while (stride > 0)
  {
    __syncthreads();
    int index = (tid + 1) * stride * 2 - 1;
    if ((index + stride) < HISTOGRAM_LENGTH)
    {
       cdf[index + stride] += cdf[index];
    }
    stride /= 2;
  }
  
  /* store back to global */
  __syncthreads();
  if(tid < HISTOGRAM_LENGTH){  output[tid] = cdf[tid]/size; }
  if((tid + blockDim.x) < HISTOGRAM_LENGTH){  output[tid + blockDim.x] = cdf[tid + blockDim.x]/size; }
}

__global__ void equalization(unsigned char* imageChar, float* cdf, int size){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    float x = 255.0*(cdf[imageChar[idx]] - cdf[0])/(1.0 - cdf[0]);
    float clamp = min(max(x, 0.0), 255.0);
    imageChar[idx] = (unsigned char) (clamp);
  }
  
}

__global__ void tofloat(unsigned char* input, float* output, int size){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
  {
    output[idx] = (float) (input[idx]/255.0);
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *deviceInputImage;
  unsigned char *deviceChar;
  unsigned char *deviceGray;
  unsigned int *deviceHisto;
  float *deviceCDF;
  float *deviceOutputImage;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  int imageSize = imageWidth * imageHeight * imageChannels;
  int graySize = imageWidth * imageHeight;
  cudaMalloc((void **)&deviceInputImage, imageSize * sizeof(float));
  cudaMalloc((void **)&deviceChar, imageSize * sizeof(unsigned char));
  cudaMalloc((void **)&deviceGray, graySize * sizeof(unsigned char));
  cudaMalloc((void **)&deviceHisto, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMalloc((void **)&deviceCDF, HISTOGRAM_LENGTH * sizeof(float));
  cudaMalloc((void **)&deviceOutputImage, imageSize * sizeof(float));

  cudaMemcpy(deviceInputImage, hostInputImageData, imageSize * sizeof(float),cudaMemcpyHostToDevice);
  cudaMemset(deviceHisto, 0, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMemset(deviceCDF, 0, HISTOGRAM_LENGTH * sizeof(float));

  dim3 dimGrid1(ceil(1.0*imageSize/BLOCK_SIZE),1,1);
  dim3 dimBlock1(BLOCK_SIZE,1,1);
  f2char<<< dimGrid1, dimBlock1 >>>(deviceInputImage, deviceChar, imageSize);
  cudaDeviceSynchronize();

  dim3 dimGrid2(ceil(1.0*graySize/BLOCK_SIZE),1,1);
  rgb2gray<<< dimGrid2, dimBlock1 >>>(deviceChar, deviceGray, graySize);
  cudaDeviceSynchronize();

  dim3 dimGrid3(ceil(1.0*graySize/HISTOGRAM_LENGTH), 1, 1);
  dim3 dimBlock2(HISTOGRAM_LENGTH, 1, 1);
  histo_kernel<<< dimGrid3, dimBlock2 >>>(deviceGray, deviceHisto, graySize);
  cudaDeviceSynchronize();

  dim3 dimGrid4(1, 1, 1);
  dim3 dimBlock3(HISTOGRAM_LENGTH/2, 1, 1);
  cdf_kernel<<< dimGrid4, dimBlock3 >>>(deviceHisto, deviceCDF, graySize);
  cudaDeviceSynchronize();

  equalization<<< dimGrid1, dimBlock1 >>>(deviceChar, deviceCDF, imageSize);
  cudaDeviceSynchronize();

  tofloat<<< dimGrid1, dimBlock1 >>>(deviceChar, deviceOutputImage, imageSize);
  cudaDeviceSynchronize();

  cudaMemcpy(hostOutputImageData, deviceOutputImage, imageSize * sizeof(float),cudaMemcpyDeviceToHost);
  wbImage_setData(outputImage, hostOutputImageData);
  wbSolution(args, outputImage);

  //@@ insert code here
  cudaFree(deviceInputImage);
  cudaFree(deviceChar);
  cudaFree(deviceGray);
  cudaFree(deviceHisto);
  cudaFree(deviceCDF);
  cudaFree(deviceOutputImage);
  free(hostInputImageData);
  free(hostOutputImageData);
  return 0;
}
