# Flash Attention - I
This chapter assumes that you know about attention mechanism. If not please see this video, which provides a lot of info about how to model, train a GPT-2 from ground up, [Andrej Karpathy video](https://www.youtube.com/watch?v=l8pRSuU81PU).
This chapter compromises of:
1) CuBLAS.
2) Some Functions And Classes To Know.
3) Finding Maximum And Sum Of An Array.

## CuBLAS
The official documentation of [Cuda](https://docs.nvidia.com/cuda/cublas/) is very detailed and well explained. I would request everyone to go through it. I believe it is self sufficent. 

## Some Functions and Classes To Knoww

### cudaSetDevice()

`As of CUDA 12.0, the cudaInitDevice() and cudaSetDevice() calls initialize the runtime and the primary context associated with the specified device. Absent these calls, the runtime will implicitly use device 0 and self-initialize as needed to process other runtime API requests. One needs to keep this in mind when timing runtime function calls and when interpreting the error code from the first call into the runtime. Before 12.0, cudaSetDevice() would not initialize the runtime and applications would often use the no-op runtime call cudaFree(0) to isolate the runtime initialization from other api activity (both for the sake of timing and error handling).`

`A host thread can set the device it operates on at any time by calling cudaSetDevice(). Device memory allocations and kernel launches are made on the currently set device; streams and events are created in association with the currently set device. If no call to cudaSetDevice() is made, the current device is device 0.`

`Multiple host threads can use the device (by calling cudaSetDevice() on this device, when using the runtime API, or by making current a context associated to the device, when using the driver API) at the same time.`

`This means, in particular, that a host thread using the runtime API without explicitly calling cudaSetDevice() might be associated with a device other than device 0 if device 0 turns out to be in prohibited mode or in exclusive-process mode and used by another process. cudaSetValidDevices() can be used to set a device from a prioritized list of devices.`

`Only the device on which a kernel is running will be controllable from that kernel. This means that device APIs such as cudaSetDevice() are not supported by the device runtime. The active device as seen from the GPU (returned from cudaGetDevice()) will have the same device number as seen from the host system. The cudaDeviceGetAttribute() call may request information about another device as this API allows specification of a device ID as a parameter of the call. Note that the catch-all cudaGetDeviceProperties() API is not offered by the device runtime - properties must be queried individually`

The above informations are taken from here, [cuda-c-programming-guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/).

So initially we associate a device to a host. Meaning when we launch a kernel from our host, the operations will happend on the device which we have set using *cudaSetDevice()*
```c++
cudaCheck(cudaSetDevice(0));
```
Since I have only one gpu, I set use the first GPU indexed by 0.

### cudaGetDeviceProperties()
This is to get the properties of a particular device. First parameter is class of *cudaDeviceProp*, the second is the device ID. 

```c++
cudaDeviceProp deviceProp;
cudaGetDeviceProperties(&deviceProp, deviceIdx);
```

After calling *cudaGetDeviceProperties()*, the device properties will be stored in *deviceProp*.

### Setting up cuBLAS

`The application must initialize a handle to the cuBLAS library context by calling the cublasCreate() function. Then, the handle is explicitly passed to every subsequent library function call. Once the application finishes using the library, it must call the function cublasDestroy() to release the resources associated with the cuBLAS library context.`   
The above information are taken from here, [cuBLAS](https://docs.nvidia.com/cuda/cublas/).   

```c++
cublasHandle_t cublas_handle;
cublasCreate(&cublas_handle);
```

### CuBLAS Error

Cuda functions return *​cudaError_t*, which defines the CUDA error types. 
CuBLAS functions return *cublasStatus_t*, which defines the cuBLAS error types. 

So to if a check Cublas error has occured, we use the follwoing code:  

```c++
// cuBLAS error checking
void cublasCheck(cublasStatus_t status, const char *file, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, line);
        exit(EXIT_FAILURE);
    }
}

#define cublasCheck(status) { cublasCheck((status), __FILE__, __LINE__); }
```

### `cublasSgemmStridedBatched` in CUDA

More information can be found here [cublas-t-gemmstridedbatched](https://docs.nvidia.com/cuda/cublas/#cublas-t-gemmstridedbatched) and [Nvidia blog](https://developer.nvidia.com/blog/cublas-strided-batched-matrix-multiply/).

The `cublasSgemmStridedBatched` function in CUDA is part of the cuBLAS library. It performs **batched** matrix-matrix multiplication. Mathematically, it computes:

$$
C[i] = \alpha \cdot (A[i]^T \times B[i]) + \beta \cdot C[i], \quad \text{for } i = 0, \dots, \text{batchCount x t} - 1
$$  
$$
A[i]\text{ is a matrix inside one batch. If } A \text{ has dim Batch x m x k, then } A[i] \text{ has dim m x K.} 
$$

#### Function Prototype

```c++
cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  int m, int n, int k,
                                  const float           *alpha,
                                  const float           *A, int lda,
                                  long long int          strideA,
                                  const float           *B, int ldb,
                                  long long int          strideB,
                                  const float           *beta,
                                  float                 *C, int ldc,
                                  long long int          strideC,
                                  int batchCount);
```
Performs:
```
C[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i]
```
for i = 0 to batchCount - 1, where op(X) is either the matrix X, its transpose, or its conjugate transpose, depending on the specified operation.

Parameters
- **handle**: handle to the cuBLAS library context. Which is defined in the previous section.
- **transa**: Specifies whether matrix A is transposed or not.
    - CUBLAS_OP_N: 
        No transpose.   
        The matrix A[i] is used as-is.

    - CUBLAS_OP_T:
        Transpose.   
        The matrix A[i] is transposed.

    - CUBLAS_OP_C:
        Conjugate transpose.
        The matrix A[i] is conjugate-transposed.
- **transb**: Specifies whether matrix B is transposed or not. 
- **m**: The number of rows of matrix C[i] and matrix A[i].
- **n**: The number of columns of matrix C[i] and matrix B[i].
- **k**: The number of columns of matrix A[i] and rows of matrix B[i].
- **alpha**: Scalar multiplier for the product of matrices A[i] and B[i].
- **A**: Pointer to the first matrix A in device memory.
- **lda**: Leading dimension of matrix A. if A has dimension batch_size x m x k. The lda = k.
- **strideA**: alue of type long long int that gives the offset in number of elements between A[i] and A[i+1].
- **B**: Pointer to the first matrix B in device memory.
- **ldb**: Leading dimension of matrix B. If B has dimension batch_size x k x n. The lda = n.
- **strideB**: Value of type long long int that gives the offset in number of elements between B[i] and B[i+1].
- **beta**: Scalar multiplier for matrix C. If beta == 0, C does not have to be a valid input.
- **C**: pointer to the C matrix corresponding to the first instance of the batch, with dimensions ldc x n with ldc>=max(1,m). Matrices C[i] should not overlap; otherwise, undefined behavior is expected.
- **ldc**: leading dimension of two-dimensional array used to store each matrix C[i].
- **strideC**: Value of type long long int that gives the offset in number of elements between C[i] and C[i+1].
- **batchCount**: Number of batches. 

## Finding Maximum And Sum Of An Array.

Given an  Array A of size 1024, write a program to calculate the maximum. Based on what we know lets write a code:
We will write a kernel to find the max. Initiall max_value will be -inf, each thread will correspond to a value in Array A and compare its value with the max_value simultaneously, if max_value is less the value represented by the thread will get stored in the max_value. Logically this should work, becuase as soon as the thread representing the max value is compared with the max_value its value will get stored.   
The below code demonstrates this idea

```c++
#include <stdio.h>
#include <limits.h>
#include <iostream>
#include <cuda_runtime.h>

__global__ void findmax(int *A, int *C){
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    //C has INT_MIN value. 
    if(A[idx] > *C)
        *C = A[idx];
}

int main(){
    const int n = 1024;
    int h_A[n];
    for(int i = 0;i < n; i ++)
        h_A[i] = i;
    int h_C = 0;
    int *d_A, *d_C;
    
    cudaMalloc(&d_A,sizeof(int)*n);
    cudaMalloc(&d_C,sizeof(int)*1);
    
    cudaMemset(d_C, INT_MIN, sizeof(int));
    cudaMemcpy(d_A,h_A,n*sizeof(int),cudaMemcpyHostToDevice);
    
    findmax<<<1,n>>>(d_A,d_C);
    
    cudaMemcpy(&h_C,d_C,1*sizeof(int),cudaMemcpyDeviceToHost);
    std::cout<<"Maximum value: "<<h_C;
}
```

The above code will give incorrect results. C will have junk value stored in it. Since all the threads run parallely, if two threads simultaneously try to write to C, then the value stored in it will be junk. We don't have any mechanism in place here to make sure that thread locking(only one thread writes into C at a time) is happening.

How to we find the max then? 
Lets start by finding the max within a wrap. Recall that whenever a instruction is executed it is wrap wide, that is 32 threads will have the same instruction. To find the maximum within a wrap we will make use of **__shfl_down_sync()** function. It is a warp-level primitive. It is warp shuffle function used for intra-warp communication, allowing threads within a warp to share data efficiently without resorting to global memory. 

```c++
T __shfl_down_sync(unsigned mask, T var, unsigned int delta, int width=warpSize);
```
T can be int, unsigned int, long, unsigned long, long long, unsigned long long, float or double.
- unsigned mask: specifies which threads inside a wrap are participating. 
    - If mask = 0xffffffff: then all the threads communicate. 
    - If mask = 0x0000000F: Only the first 4 threads (lanes 0 to 3) communicate.
    - If mask = 0x0000FFFF: Only the first 16 threads (lanes 0 to 15) communicate.
- T var: The value inside each thread which needs to be communicated.
- unsigned int delta: The number of threads to shift down. 
    - For example: if delta = 16. Then thread 0(also called lane 0) communicates with thread 16( 16 + 0), 1(also called lane 1) with 17(16 + 1), 2(also called lane 2) with 18(16 + 2), 3(also called lane 3) with 19(16 + 3)... 15 with 32(16 + 15).
- width: Optional. Specifies the width of the shuffle (default is warpSize).

More about this can be found here,[using-cuda-warp-level-primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/). 

### Lets write a program to find the max within an array of size = 32(wrap size).
The code is here, the explanation is given after this.

```c++
#include <stdio.h>
#include <limits.h>
#include <iostream>
#include <cuda_runtime.h>

__global__ void findmax(int *A, int *C) {
    int idx = threadIdx.x;
    

    if (idx < 32)
        max_val = A[idx];

    // Perform warp-level reduction using shuffle instructions
    for (int offset = 16; offset > 0; offset /= 2)
        max_val = max(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));

    // Store the result from thread 0
    if (idx == 0)
        *C = max_val;
}

int main() {
    const int n = 32;
    int h_A[n];
    for (int i = 0; i < n; i++)
        h_A[i] = i; 

    int h_C = 0;
    int *d_A, *d_C;

    cudaMalloc(&d_A, sizeof(int) * n);
    cudaMalloc(&d_C, sizeof(int));

    cudaMemcpy(d_A, h_A, sizeof(int) * n, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, sizeof(int));

    findmax<<<1, 32>>>(d_A, d_C); // Launch kernel with 32 threads (1 block, 32 threads)

    cudaMemcpy(&h_C, d_C, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Maximum value: " << h_C << std::endl;

   
    cudaFree(d_A);
    cudaFree(d_C);
}
```

Explanation: This is the main part of the code.
```c++
if (idx < 32)
        max_val = A[idx];
for (int offset = 16; offset > 0; offset /= 2)
        max_val = max(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
```
initially max_val in each thread will have A[idx].   
   
During the first iteration, that is when offeset = 16, thread 0 communicates with thread 16( 16 + 0) and updates its max_val, 1 with 17(16 + 1) and updates its max_val, 2 with 18(16 + 2) and updates its max_val, 3 with 19(16 + 3) and updates its max_val... 15 with 32(16 + 15) and updates its max_val. After the first iteration,any of the threads from 0 to 15 will have the maximum of the array.   

During the second iteration, that is when offeset = 8, thread 0 communicates with thread 8( 8 + 0) and updates its max_val, 1 with 9(8 + 1) and updates its max_val, 2 with 10(8 + 2) and updates its max_val, 3 with 11(8 + 3) and updates its max_val... 7 with 15(8 + 7) and updates its max_val. After the second iteration,any of the threads from 0 to 7 will have the maximum of the array.    

In the last iteration, when offset = 2. The thread 0 communicates with thread 1 and updates its max_val.    
So the maximum val of the array will be stored in the Thread 0's max_val.   

### Lets write a program to find the max within an array whose size is a power of 2 and greater 31.
 
laneId = threadIdx.x % 32;     // to identify the threadId within a wrap.
warpId = threadIdx.x / 32;     // to identify the wrap to which a thread belongs to.

Here we will make use of shared memory.

- IThis is what we do:   
    - The number of blocks is 1. The threads per block = threads_per_block. Array size = asize. Number of partitions = asize/threads_per_block.
    - The threads in partition 0, compares its values with threads in other partitions. For example lets say there are 4 partitions, threads per partition = 64. Array size =  256.    
        - Thread 0 of partition 0 will compare its value with thread 0 of partition 1, thread 0 of partition 2, thread 0 of partition 3. 
        - Thread 1 of partition 0 will compare its value with thread 1 of partition 1, thread 1 of partition 2, thread 1 of partition 3.
        - Thread 63 of partition 0 will compare its value with thread 63 of partition 1, thread 63 of partition 2, thread 63 of partition 3.
    - Now the maximum will be inside the threads in partition 0.
    - Now we perform within-warp reductions which we discussed above. After this thread 0 within each wrap will have the maximum value, which we will store it in the shared memory. Thread 0 of a wrap is identified by warpId.
    - Now the shared memory will contain the maximum. All we have to do is iterate through thisshared memory to find the maximum.

- How do we decide the size of shared memory?
    - For each wrap inside the partition zero, we will have a **1 \* sizeof(type)** bytes of shared memory. Because remember we have to store the result of the maximum of each wrap, this is where shared memory comes into the picture. 

*I have left it to you to figure out why only the powers of 2, greater than 31 can be used as a array size*
      
```c++

#include <stdio.h>
#include <limits.h>
#include <iostream>
#include <cuda_runtime.h>


__global__ void findmax(int *A, int *C, int asize) {
    // special reduction operations warpReduceMax/warpReduceSum are used for intra-warp reductions
    // shared memory is used for inter-warp reduction
    extern __shared__ int maxvals[];
    int tid = threadIdx.x;
    int warpId = threadIdx.x / 32; // warp index within a block
    int laneId = threadIdx.x % 32; // thread index within a warp

     // the number of warps per block. recall that blockDim.x is block_size
    int warpsPerBlock = blockDim.x / 32;
    
    // first, thread coarsening by directly accessing global memory in series
    int max_val = INT_MIN;
    for(int i = tid;i < asize; i += blockDim.x)
        max_val = max(max_val,A[i]);
    
    __syncthreads();
    // now within-warp reductions for maxval
    for (int offset = 16; offset > 0; offset /= 2)
        max_val = max(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));

    // the 0th thread of each warp writes the maxval of that warp to shared memory
    if (laneId == 0) maxvals[warpId] = max_val;
    __syncthreads();

    // now the 0th thread reduces the maxvals in shared memory, i.e. across warps
    if (tid == 0) {
        int val = maxvals[tid];
        for (int i = 1; i < warpsPerBlock; i++) {
            val = max(val, maxvals[i]);
        }
        // store the final max in the first position
        *C = val;
    }
    __syncthreads();
    
}

int main() {
    const int n = 1024; // always a power of 2 and greater than 31.
    int threads_per_block = 64; // always a power of 2 and greater than 31.
    int h_A[n];
    for (int i = 0; i < n; i++)
        h_A[i] = i; 

    int h_C = 0;
    int *d_A, *d_C;

    cudaMalloc(&d_A, sizeof(int) * n);
    cudaMalloc(&d_C, sizeof(int));

    cudaMemcpy(d_A, h_A, sizeof(int) * n, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, sizeof(int));


    const int space = (threads_per_block + 31) / 32 * sizeof(int);

    findmax<<<1, threads_per_block, space>>>(d_A, d_C,n); // Launch kernel with 32 threads (1 block, 32 threads)

    cudaMemcpy(&h_C, d_C, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout<< "Maximum value: " << h_C << std::endl;

   
    cudaFree(d_A);
    cudaFree(d_C);
}
```
### Finding the row maximum of a (N,C) dimesnion array. N is a power of 2 and greater than 31.

Most of the code from above remains the same except the below changes. 
    - Input is a N*M dim array. N = rows, M = columns.
    - We will represent the final result by max_result instead of C. max_result wil be N dimesnional array.
    - Instead of having one block we will have N blocks, to find max in each of the N rows. 
```c++
#include <stdio.h>
#include <limits.h>
#include <iostream>
#include <cuda_runtime.h>

__global__ void findmax(int *A, int *max_result,int asize) {
    // special reduction operations warpReduceMax/warpReduceSum are used for intra-warp reductions
    // shared memory is used for inter-warp reduction
    extern __shared__ int maxvals[];
    int tid = threadIdx.x;
    int warpId = threadIdx.x / 32; // warp index within a block
    int laneId = threadIdx.x % 32; // thread index within a warp

     // the number of warps per block. recall that blockDim.x is block_size
    int warpsPerBlock = blockDim.x / 32;
    
    // first, thread coarsening by directly accessing global memory in series
    int max_val = INT_MIN;
    int *inp = A + asize * blockIdx.x;
    for(int i = tid;i < asize; i += blockDim.x)
        max_val = max(max_val,inp[i]);
    
    __syncthreads();
    // now within-warp reductions for maxval
    for (int offset = 16; offset > 0; offset /= 2)
        max_val = max(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));

    // the 0th thread of each warp writes the maxval of that warp to shared memory
    if (laneId == 0) maxvals[warpId] = max_val;
    __syncthreads();

    // now the 0th thread reduces the maxvals in shared memory, i.e. across warps
    if (tid == 0) {
        int val = maxvals[tid];
        for (int i = 1; i < warpsPerBlock; i++) {
            val = max(val, maxvals[i]);
        }
        // store the final max in the first position
        max_result[blockIdx.x] = val;
    }
    __syncthreads();
    
}

int main() {
    const int m = 1024, n = 32;
    int threads_per_block = 64; // always a power of 2 and greater than 31.
    
    int h_A[n*m];
    for (int i = 0; i < n; i++)
        for(int j = 0;j < m; j ++)
            h_A[i*m + j] = i*m + j; 

    int h_C[n];
    int *d_A, *d_C;

    cudaMalloc(&d_A, sizeof(int) * n * m);
    cudaMalloc(&d_C, sizeof(int) * n);

    cudaMemcpy(d_A, h_A, sizeof(int) * n * m, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, sizeof(int)* n);


    const int space = (threads_per_block + 31) / 32 * sizeof(int);

    findmax<<<n, threads_per_block, space>>>(d_A, d_C,m); // Launch kernel with 32 threads (1 block, 32 threads)

    cudaMemcpy(&h_C, d_C, sizeof(int) * n, cudaMemcpyDeviceToHost);
    for(int i = 0;i < n; i++)
        std::cout<<"Row Number : "<< i <<". Maximum value: " << h_C[i] << std::endl;

   
    cudaFree(d_A);
    cudaFree(d_C);
}
```
Output of the code:

```
Row Number : 0. Maximum value: 1023
Row Number : 1. Maximum value: 2047
Row Number : 2. Maximum value: 3071
Row Number : 3. Maximum value: 4095
Row Number : 4. Maximum value: 5119
Row Number : 5. Maximum value: 6143
Row Number : 6. Maximum value: 7167
Row Number : 7. Maximum value: 8191
Row Number : 8. Maximum value: 9215
Row Number : 9. Maximum value: 10239
Row Number : 10. Maximum value: 11263
Row Number : 11. Maximum value: 12287
Row Number : 12. Maximum value: 13311
Row Number : 13. Maximum value: 14335
Row Number : 14. Maximum value: 15359
Row Number : 15. Maximum value: 16383
Row Number : 16. Maximum value: 17407
Row Number : 17. Maximum value: 18431
Row Number : 18. Maximum value: 19455
Row Number : 19. Maximum value: 20479
Row Number : 20. Maximum value: 21503
Row Number : 21. Maximum value: 22527
Row Number : 22. Maximum value: 23551
Row Number : 23. Maximum value: 24575
Row Number : 24. Maximum value: 25599
Row Number : 25. Maximum value: 26623
Row Number : 26. Maximum value: 27647
Row Number : 27. Maximum value: 28671
Row Number : 28. Maximum value: 29695
Row Number : 29. Maximum value: 30719
Row Number : 30. Maximum value: 31743
Row Number : 31. Maximum value: 32767
```

### Finding the row sum of a (N,C) dimesnion array. N is a power of 2 and greater than 31.

The code is similar to the above, but wherever max operation is present we do addition instead. 

```c++
#include <stdio.h>
#include <limits.h>
#include <iostream>
#include <cuda_runtime.h>


__global__ void findmax(int *A, int *results,int asize) {
    // special reduction operations warpReduceMax/warpReduceSum are used for intra-warp reductions
    // shared memory is used for inter-warp reduction
    extern __shared__ int row_sums[];
    int tid = threadIdx.x;
    int warpId = threadIdx.x / 32; // warp index within a block
    int laneId = threadIdx.x % 32; // thread index within a warp

     // the number of warps per block. recall that blockDim.x is block_size
    int warpsPerBlock = blockDim.x / 32;
    
    // first, thread coarsening by directly accessing global memory in series
    int row_sum = 0;
    int *inp = A + asize * blockIdx.x;
    for(int i = tid;i < asize; i += blockDim.x)
        row_sum += inp[i];
    
    __syncthreads();
    // now within-warp reductions for maxval
    for (int offset = 16; offset > 0; offset /= 2)
        row_sum += __shfl_down_sync(0xFFFFFFFF, row_sum, offset);

    // the 0th thread of each warp writes the maxval of that warp to shared memory
    if (laneId == 0) row_sums[warpId] = row_sum;
    __syncthreads();

    // now the 0th thread reduces the maxvals in shared memory, i.e. across warps
    if (tid == 0) {
        int val = row_sums[tid];
        for (int i = 1; i < warpsPerBlock; i++) {
            val +=  row_sums[i];
        }
        // store the final max in the first position
        results[blockIdx.x] = val;
    }
    __syncthreads();
    
}

int main() {
    const int m = 1024, n = 32;
    int threads_per_block = 64; // always a power of 2 and greater than 31.
    int h_A[n*m];
    for (int i = 0; i < n; i++)
        for(int j = 0;j < m; j ++)
            h_A[i*m + j] = j + i;

    int h_C[n];
    int *d_A, *d_C;

    cudaMalloc(&d_A, sizeof(int) * n * m);
    cudaMalloc(&d_C, sizeof(int) * n);

    cudaMemcpy(d_A, h_A, sizeof(int) * n * m, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, sizeof(int)* n);


    const int space = (threads_per_block + 31) / 32 * sizeof(int);

    findmax<<<n, threads_per_block, space>>>(d_A, d_C,m); // Launch kernel with 32 threads (1 block, 32 threads)

    cudaMemcpy(&h_C, d_C, sizeof(int) * n, cudaMemcpyDeviceToHost);
    for(int i = 0;i < n; i++)
        std::cout<<"Row Number : "<< i <<". Row sum: " << h_C[i] << std::endl;

   
    cudaFree(d_A);
    cudaFree(d_C);
}
```

Output

```
Row Number : 0. Row sum: 523776
Row Number : 1. Row sum: 524800
Row Number : 2. Row sum: 525824
Row Number : 3. Row sum: 526848
Row Number : 4. Row sum: 527872
Row Number : 5. Row sum: 528896
Row Number : 6. Row sum: 529920
Row Number : 7. Row sum: 530944
Row Number : 8. Row sum: 531968
Row Number : 9. Row sum: 532992
Row Number : 10. Row sum: 534016
Row Number : 11. Row sum: 535040
Row Number : 12. Row sum: 536064
Row Number : 13. Row sum: 537088
Row Number : 14. Row sum: 538112
Row Number : 15. Row sum: 539136
Row Number : 16. Row sum: 540160
Row Number : 17. Row sum: 541184
Row Number : 18. Row sum: 542208
Row Number : 19. Row sum: 543232
Row Number : 20. Row sum: 544256
Row Number : 21. Row sum: 545280
Row Number : 22. Row sum: 546304
Row Number : 23. Row sum: 547328
Row Number : 24. Row sum: 548352
Row Number : 25. Row sum: 549376
Row Number : 26. Row sum: 550400
Row Number : 27. Row sum: 551424
Row Number : 28. Row sum: 552448
Row Number : 29. Row sum: 553472
Row Number : 30. Row sum: 554496
Row Number : 31. Row sum: 555520
```