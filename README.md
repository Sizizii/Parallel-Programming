# ECE408 Applied Parallel Programming

MP0: Device Query
MP1: Vector Addition
MP2: Simple Matrix Multiply
MP3: Tiled Matrix Multiply
MP4: 3D Convolution
MP5.1: List Reduction
MP5.2: Scan
MP6: Histogramming
MP7: Sparse Matrix Multiply

Project: Implementation and optimization of the forward-pass of a CNN using CUDA. Optimizations include:
* Tiled shared memory convolution
* Weight matrix (kernel values) in constant memory
* Fixed point (FP16) arithmetic
* Sweeping various parameters to find best values (block sizes, amount of thread coarsening)
* Input channel reduction: tree
* Input channel reduction: atomics
