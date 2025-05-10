# Flash Attention - II
This chapter assumes that you know about attention mechanism. If not please see this video, which provides a lot of info about how to model, train a GPT-2 from ground up, [Andrej Karpathy video](https://www.youtube.com/watch?v=l8pRSuU81PU).
This chapter compromises of:
1) Attention Pytorch Code.
2) Attention From Scratch Using Cuda.


## The Vanilla Self-Attention: Math Refresher

Before we look at the code, let's quickly recap the math for a single attention head.

Given Query ($Q$), Key ($K$), and Value ($V$) matrices for a sequence of length $T$ and feature dimension $d_k$ (per head):

1.  **Scaled Dot-Product Scores:** Calculate the raw attention scores.
    $ S = Q K^T $   
    This results in a $T \times T$ matrix where $S_{ij}$ is the dot product of the i-th query vector and the j-th key vector.

2.  **Scaling:** Scale the scores to prevent gradients from becoming too small.   
    $ S_{\text{scaled}} = \frac{S}{\sqrt{d_k}} $
    Where $d_k$ is the dimension of the key vectors (and query vectors).

3.  **Masking (Optional, but common for decoders):** For causal attention (like in GPT), we prevent a position from attending to future positions. This is usually done by setting the corresponding scores in $S_{\text{scaled}}$ to negative infinity before the softmax.
    If $j > i$, then $(S_{\text{scaled}})_{ij} = -\infty$.

4.  **Softmax:** Apply softmax row-wise to the scaled scores to get probabilities.
    $ P = \text{softmax}(S_{\text{scaled}}) $
    Each row of $P$ now sums to 1, and $P_{ij}$ represents how much attention token $i$ should pay to token $j$.

5.  **Weighted Sum of Values:** Compute the output by taking a weighted sum of the Value vectors using the attention probabilities.
    $ O = P V $
    The output $O$ is a $T \times d_v$ matrix (where $d_v$ is often equal to $d_k$).

**Multi-Head Attention:**
Instead of one big attention calculation, we split Q, K, and V into $NH$ (Number of Heads) smaller pieces along the feature dimension. Each "head" performs the above 5 steps independently. The outputs of all heads are then concatenated and linearly projected back to the original embedding dimension. This allows the model to jointly attend to information from different representation subspaces at different positions.

---

## The Bottleneck: Why is Standard Attention Slow?

The main issue with a naive implementation of attention, especially on GPUs, is **memory bandwidth**.

1.  **Large Intermediate Matrices:** The matrices $S$ ($QK^T$) and $P$ ($\text{softmax}(S)$) can be very large ($T \times T$). For a sequence length $T=1024$, $S$ has over a million elements! For $T=8192$, it's 67 million elements.
2.  **Multiple Memory Accesses:**
    *   Read Q, K from High Bandwidth Memory (HBM, the GPU's main RAM).
    *   Write $S$ to HBM.
    *   Read $S$ from HBM for scaling and softmax.
    *   Write $P$ to HBM.
    *   Read $P$ and V from HBM.
    *   Write final output $O$ to HBM.

All these reads and writes to HBM are slow compared to computations happening on the GPU's cores. The goal of optimized attention (like the one in the code, and more advanced versions like the original FlashAttention paper) is to reduce these HBM accesses by fusing operations and keeping intermediate data in faster on-chip memory.

## Flash Attention.

The main difference between flash attention and the naive attention is how the softmax is computed. In naive implementation the shared memomry is not used, every time the global memory is accessed to compute the max and sum of each row.(Max is computed to subtract from the each element, gives numerical stability). 

In flash attention, shared memory is used. As we have seen how to compute max and sum in the previous chapter, we will use those concepts directly here.   
The version written in this blog post is simplified one which will help understand the original implementation which will be covered in the latter chapter.


## Attention Pytorch Code.

The below is the computation of Attention using pytorch.
We will do the exact same sequence of operations in the cuda code.
```python
def forward(self, qkvr): // x is the qkv matrix
        B, T, three_C = qkvr.size() # batch size, sequence length, embedding dimensionality (n_embd)
        C = three_C // 3

        #### step 1: permute ####
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = split(qkvr) # split qkvr into q, k, v 
        k = k.view(B, T, n_head, C // n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, n_head, C // n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, n_head, C // n_head).transpose(1, 2) # (B, nh, T, hs)

        # manual implementation of attention
        
        #### step 2: Dot product ########
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        ##### step 3: scale and mask ######
        # block_size is the max sequence length
        bias = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size) 
        att = att.masked_fill(bias[:,:,:T,:T] == 0, float('-inf'))
        
        ##### step 4: Perform softmax operation ######
        att = F.softmax(att, dim=-1)
        
        #### Step 5: calculate y #######
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        #### step 6: unpermute final output ######
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        return y
```

The cuda code we write will be made of these sub components, which is defined in the above code:
- Step 0: The int main() function where we define the host/device inputs, outputs and intermediate variables.
- step 1 - Permute: Splitting the input qkv into q, k, v. By calling the permute kernel.
- step 2 - Dot Product: Dot product of q and k to compute preattn.
- step 3 - Scale And Mask: Scale and mask the entries of preattn matrix.
- step 4 - Perform softmax operation.
- step 5 - Calculate y.
- step 6 - Unpermute y.

Lets start with step 0, where we define the host/device inputs, outputs and intermediate variables.

## Flash Attention Cuda Coda

### Step 0: Defining the inputs and outputs.

Here is the Main funtion code.

```c++

int main(int argc, char **argv) {
	
    int B = 8;
    int T = 1024;
    int C = 768;
    int NH = 12;

    int deviceIdx = 0;
	

    cudaCheck(cudaSetDevice(deviceIdx)); // device for calling host thread
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);

    // setup cuBLAS
    cublasCreate(&cublas_handle);
	
    // create host memory of random numbers
    float* out = (float*)malloc(B * T * C * sizeof(float));
    float* preatt = (float*)malloc(B * NH * T * T * sizeof(float));
    float* att = (float*)malloc(B * NH * T * T * sizeof(float));
    //float* inp = make_random_float(B * T * 3 * C, 10.0f);
    float* inp = make_random_float(B * T * 3 * C);

    // move to GPU
    float* d_out;
    float* d_vaccum;
    float* d_qkvr;
    float* d_preatt;
    float* d_att;
    float* d_inp;
    cudaCheck(cudaMalloc(&d_out, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_vaccum, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_qkvr, B * T * 3 * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_preatt, B * NH * T * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_att, B * NH * T * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp, B * T * 3 * C * sizeof(float)));
    cudaCheck(cudaMemcpy(d_inp, inp, B * T * 3 * C * sizeof(float), cudaMemcpyHostToDevice));

    int block_sizes[] = {32, 64, 128, 256, 512}; 
    int block_size = block_sizes[0];
        
    attention_forward(d_out,d_vaccum, d_qkvr, d_preatt, d_att, d_inp, B, T, C, NH, block_size);
    printf("success !!");

    free(out);
    free(preatt);
    free(att);
    free(inp);
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_vaccum));
    cudaCheck(cudaFree(d_qkvr));
    cudaCheck(cudaFree(d_preatt));
    cudaCheck(cudaFree(d_att));
    cudaCheck(cudaFree(d_inp));
    cublasDestroy(cublas_handle);

    return 0;
}
```

**inp**: is the value which has the qkv matrix.      
**d_preattn**: output of Q*(K^T).   
**d_attn**: softmax applied to d_preattn.   
**d_vaccum**: d_attn * V.   
**d_out**: Final output after re-arranging.   

The *attention_forward()* function has all the neccessary kernel calls (That is step 1 to step 6).

```c++

template<class T>
__host__ __device__ T ceil_div(T dividend, T divisor) {
    return (dividend + divisor-1) / divisor;
}

void attention_forward(float* out, float* vaccum, float* qkvr, float* preatt, float* att,
                       const float* inp,
                       int B, int T, int C, int NH,
                       const int block_size) {
    // inp is (B, T, 3C) QKV
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int HS = C / NH; // head size
    
    // ----------------------- step 1: Pemute ------------------------//

    // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
    float *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    int total_threads = B * NH * T * HS;
    int num_blocks =ceil_div(total_threads,block_size);
    permute_kernel<<<num_blocks, block_size>>>(q, k, v, inp, B, T, NH, HS);

    // ----------------------- step 2: Q*(K^T) Product ------------------------//

    // batched matrix multiply with cuBLAS
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasCheck(cublasSgemmStridedBatched(cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            T, T, HS,
                            &alpha,
                            k, HS, T * HS,
                            q, HS, T * HS,
                            &beta,
                            preatt, T, T * T,
                            B * NH));

    // ----------------------- step 3: SCALE AND MASK  ------------------------//

    // multiply all elements of preatt elementwise by scale
    float scale = 1.0f / sqrtf(HS);
    total_threads = B * NH * T * T;
    num_blocks = ceil_div(total_threads, block_size);
    scale_kernel<<<num_blocks, block_size>>>(preatt, scale, B, NH, T);

    // ----------------------- step 4: SOFTMAX ------------------------//

    // softmax. preatt is (B, NH, T, T) but we view it as (B * NH * T, T) and use the softmax kernel
    int softmax_block_size = 256;
    int grid_size = B * NH * T;
    size_t shared_mem_size = 2 * softmax_block_size / 32 * sizeof(float);
    softmax_forward_kernel<<<grid_size, softmax_block_size, shared_mem_size>>>(att, preatt, B * NH * T, T);

    // ----------------------- step 5: y = attn * V ------------------------//

    // new approach: first cuBLAS another batched matmul
    // y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    cublasCheck(cublasSgemmStridedBatched(cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            HS, T, T,
                            &alpha,
                            v, HS, T * HS,
                            att, T, T * T,
                            &beta,
                            vaccum, HS, T * HS,
                            B * NH));

    // ----------------------- step 6: Unpermute Y ------------------------//

    // now unpermute
    // y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    num_blocks = ceil_div(B * T * C, block_size);
    unpermute_kernel<<<num_blocks, block_size>>>(vaccum, out, B, T, NH, HS);
}
```

### Step 1 - Permute: Splitting the input qkv into q, k, v. By calling the permute kernel.

*   **Purpose:** The input `inp` often comes from a previous layer (like a linear projection) and might have Q, K, and V packed together. For efficient processing, especially with batched matrix multiplies used in cuBLAS, it's better to have Q, K, and V separated and with a layout like `(Batch, NumHeads, SequenceLength, HeadDim)`. This kernel performs that rearrangement.

```c++

__global__ void permute_kernel(float* q, float* k, float* v,
                               const float* inp,
                               int B, int N, int NH, int d) {
    // okay so now, this kernel wants Q,K,V to all be of shape (B, NH, N, d)
    // but instead, we have a single tensor QKV (inp) of shape (B, N, 3, NH, d)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Q[b][nh_][n][d_] = inp[b][n][0][nh_][d_]

    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int inp_idx = \
            (b * N * 3 * NH * d)
            +   (n * 3 * NH * d)
            +       (0 * NH * d)
            +          (nh_ * d)
            +                d_;

        q[idx] = inp[inp_idx];
        k[idx] = inp[inp_idx + NH * d];
        v[idx] = inp[inp_idx + 2 * (NH * d)];
    }
}
```

*   **Indexing Logic:**
    *   `idx = blockIdx.x * blockDim.x + threadIdx.x;`: This is the standard way to get a unique global ID for each thread. Each thread will be responsible for writing one element to each of `q`, `k`, and `v`.
    *   The code deconstructs `idx` based on the *target* layout `(B, NH, N, d)` to find the `b, nh_, n, d_` coordinates for the output.
    *   Then, it reconstructs the source indices (`inp_idx_q`, `inp_idx_k`, `inp_idx_v`) based on the *source* layout `(B, N, 3, NH, d)`. The offsets `NH * d` correctly step from the Q part to K part, and K part to V part, within the `[3]` dimension of the input tensor.
*   **Memory Access Pattern:** This kernel performs strided reads from `inp` and (ideally) coalesced writes to `q`, `k`, `v`.

Now can access the Q,KV matrix via the pointers q,k,v.

### Step 2 - Dot Product: Dot product of q and k to compute preattn.

```c++
// ----------------------- step 2: Q*(K^T) Product ------------------------//

    // batched matrix multiply with cuBLAS
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasCheck(cublasSgemmStridedBatched(cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            T, T, HS,
                            &alpha,
                            k, HS, T * HS,
                            q, HS, T * HS,
                            &beta,
                            preatt, T, T * T,
                            B * NH));

```
This does the matrix multiplication of Q and K transpose. We could write our won kernel as we did in post 2, but built in functions are faster and safer for various GPUs.

### Step 3: SCALE and MASK.

```c++
 // multiply all elements of preatt elementwise by scale
    float scale = 1.0f / sqrtf(HS);
    total_threads = B * NH * T * T;
    num_blocks = ceil_div(total_threads, block_size);
    scale_kernel<<<num_blocks, block_size>>>(preatt, scale, B, NH, T);
```

```c++
__global__ void scale_kernel(float* inp, float scale, int B, int NH, int T) {
    // scales the pre-softmax attention scores by scale
    // and sets the autoregressive locations to -INFINITY
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * NH * T * T) {
        int rest = idx % (NH * T * T);
        rest = rest % (T * T);
        int t2 = rest / T;
        int t = rest % T;
        if (t > t2) {
            inp[idx] = -INFINITY;
        } else {
            inp[idx] *= scale;
        }
    }
}
```

*   Here in this step we launcha kernel where each thread handles one element of the `(B, NH, T, T)` `preatt` tensor.
*   It applies the $1/\sqrt{d_k}$ scaling.
*   It also performs causal masking: if `key_idx > query_idx` (i.e., `t1 > t2`), it means the query at `t2` is trying to attend to a key at `t1` which is in the "future". This is set to `-INFINITY` so that after softmax, its probability becomes 0.

### Step 4: Softmax kernel.

This is the main part.

```c++
// ----------------------- step 4: SOFTMAX ------------------------//

    // softmax. preatt is (B, NH, T, T) but we view it as (B * NH * T, T) and use the softmax kernel
    int softmax_block_size = 256;
    int grid_size = B * NH * T;
    size_t shared_mem_size = 2 * softmax_block_size / 32 * sizeof(float);
    softmax_forward_kernel<<<grid_size, softmax_block_size, shared_mem_size>>>(att, preatt, B * NH * T, T);
```
Below is the kernel code. Most of it is to find the max and the sum of each row of T*T matrix, in total we have B*NH*T number of rows for which we need to calculate the max and min. If you have read the previous blog, then the kernel code written here will be easy to understand.

Note that we launch B\*NH\*T number of blocks which are of size softmax_block_size. Each block will be responsible for computing the softmax for a row, in total we have B\*NH\*T number of rows.

Here the shared memory size is 2 * softmax_block_size / 32 * sizeof(float), beacuse each wrap will calculate max and sum between 32 threads and store it in one shared memory. In total we have softmax_block_size/32 number of wraps. Multiplied by 2 beacause we need separate memory location for max and sum.

```c++
__global__ void softmax_forward_kernel(float* out, const float* inp, int N, int C) {
    // out is (N, C) just like inp. Each row of inp will get softmaxed.
    // same as kernel3, but can handle any block size (multiple of 32)
    // each row of C elements is handled by block_size threads
    // furthermore, each block_size threads get executed in warps of 32 threads

    // special reduction operations warpReduceMax/warpReduceSum are used for intra-warp reductions
    // shared memory is used for inter-warp reduction
    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int warpId = threadIdx.x / 32; // warp index within a block
    int laneId = threadIdx.x % 32; // thread index within a warp

    // the number of warps per block. recall that blockDim.x is block_size
    int warpsPerBlock = blockDim.x / 32;

    // shared[] must be allocated to have 2 * warpsPerBlock elements
    // first half for max values, the second half for sum values
    float* maxvals = shared;
    float* sumvals = &shared[warpsPerBlock];

    // one row of inp, i.e. inp[idx, :] of shape (C,)
    const float* x = inp + idx * C;

    // first, thread coarsening by directly accessing global memory in series
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x) {
        maxval = fmaxf(maxval, x[i]);
    }
    // now within-warp reductions for maxval
    maxval = warpReduceMax(maxval);

    // the 0th thread of each warp writes the maxval of that warp to shared memory
    if (laneId == 0) maxvals[warpId] = maxval;
    __syncthreads();

    // now the 0th thread reduces the maxvals in shared memory, i.e. across warps
    if (tid == 0) {
        float val = maxvals[tid];
        for (int i = 1; i < warpsPerBlock; i++) {
            val = fmaxf(val, maxvals[i]);
        }
        // store the final max in the first position
        maxvals[0] = val;
    }
    __syncthreads();
    // broadcast the max to all threads
    float offset = maxvals[0];

    // compute expf and write the result to global memory
    for (int i = tid; i < C; i += blockDim.x) {
        // subtract max for numerical stability
        out[idx * C + i] = expf(x[i] - offset);
    }

    // okay now we calculated exp(x - max(x))
    // step 2: sum all the values and divide by the sum

    // thread coarsening for sum
    x = out + idx * C;
    float sumval = 0.0f;
    for (int i = tid; i < C; i += blockDim.x) {
        sumval += x[i];
    }
    // within-warp reduction for sumval
    sumval = warpReduceSum(sumval);

    // write sumval to shared memory
    if (laneId == 0) sumvals[warpId] = sumval;
    __syncthreads();

    // inter-thread reduction of sum
    if (tid == 0) {
        float val = sumvals[tid];
        for (int i = 1; i < warpsPerBlock; ++i) {
            val += sumvals[i];
        }
        sumvals[0] = val;
    }
    __syncthreads();
    // broadcast the sum to all threads
    float sum = sumvals[0];

    // divide the whole row by the sum
    for (int i = tid; i < C; i += blockDim.x) {
        out[idx * C + i] = x[i] / sum;
    }
}
```

*   **Numerical Stability:** Softmax $\frac{\exp(x_i)}{\sum \exp(x_j)}$ can overflow if $x_i$ are large. Subtracting the max $m$ from all $x_i$ before `exp` ($\frac{\exp(x_i - m)}{\sum \exp(x_j - m)}$) gives the same result but is numerically stable.
*   **Two-Pass Style Reduction (Max then Sum):**
    1.  **Find Max:** Each block calculates the maximum value in its assigned row.
        *   **Thread Coarsening:** Each thread iterates over `C_cols / blockDim.x` elements to find its local max.
        *   **Warp Reduction:** `warpReduceMax` finds the max within each warp.
        *   **Shared Memory Reduction:** The first thread of each warp writes its warp's max to shared memory. Then, the first thread of the *block* reduces these values in shared memory to get the true row maximum.
        *   `__syncthreads()`: Barrier synchronization. Ensures all threads in a block reach this point before any proceed. Crucial when writing to and then reading from shared memory.
    2.  **Calculate $\exp(x_i - \text{max})$ and Sum:**
        *   Each thread calculates $\exp(x_i - \text{offset})$ for its elements and writes to the `out` buffer (temporarily), while also accumulating its local sum.
        *   Then, a similar reduction process (warp reduction, shared memory reduction) is used to find the sum of these $\exp$ values.
    3.  **Divide:** Each thread divides its $\exp(x_i - \text{offset})$ values (read from `out`) by the `total_sum`.

### Step 5: Ouput of attention matrix and V.

This is self explanatory.   
The ouput is stored in vaccum.

```c++ 
// ----------------------- step 6: y = attn * V ------------------------//

    // new approach: first cuBLAS another batched matmul
    // y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    cublasCheck(cublasSgemmStridedBatched(cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            HS, T, T,
                            &alpha,
                            v, HS, T * HS,
                            att, T, T * T,
                            &beta,
                            vaccum, HS, T * HS,
                            B * NH));
```                        

### Step 6: Last step, Unpermute Y.

The vaccum tensor currently has the shape (B, NH, T, HS). The final output of an attention layer is usually (B, T, C). This requires a transpose from (B, NH, T, HS) to (B, T, NH, HS) and then a reshape/view where the (NH, HS) dimensions become contiguous to form C.

 ```c++
 // ----------------------- step 6: Unpermute Y ------------------------//

    // now unpermute
    // y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    num_blocks = ceil_div(B * T * C, block_size);
    unpermute_kernel<<<num_blocks, block_size>>>(vaccum, out, B, T, NH, HS);
```

```c++
__global__ void unpermute_kernel(const float* inp, float *out, int B, int N, int NH, int d) {
   // out has shape (B, nh, N, d) but we need to unpermute it to (B, N, nh, d)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // out[b][n][nh_][d_] <- inp[b][nh_][n][d_]
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
        out[other_idx] = inp[idx];
    }
}

```

### Entire code.

Putting it all together, the entire code is presented below.
Try running it using the below command:
```
nvcc -lcublas flash_attention.cu -o flash_attention.exe
```
```
./flash_attention_blog.exe
```

```c++

#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <float.h>
#include <cublas_v2.h>
#include <cmath>

cublasHandle_t cublas_handle;

template<class T>
__host__ __device__ T ceil_div(T dividend, T divisor) {
    return (dividend + divisor-1) / divisor;
}


float* make_random_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    return arr;
}

void cuda_check(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
               cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}
#define cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))

// cuBLAS error checking
void cublasCheck(cublasStatus_t status, const char *file, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, line);
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(status) { cublasCheck((status), __FILE__, __LINE__); }

__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}


__global__ void permute_kernel(float* q, float* k, float* v,
                               const float* inp,
                               int B, int N, int NH, int d) {
    // okay so now, this kernel wants Q,K,V to all be of shape (B, NH, N, d)
    // but instead, we have a single tensor QKV (inp) of shape (B, N, 3, NH, d)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Q[b][nh_][n][d_] = inp[b][n][0][nh_][d_]

    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int inp_idx = \
            (b * N * 3 * NH * d)
            +   (n * 3 * NH * d)
            +       (0 * NH * d)
            +          (nh_ * d)
            +                d_;

        q[idx] = inp[inp_idx];
        k[idx] = inp[inp_idx + NH * d];
        v[idx] = inp[inp_idx + 2 * (NH * d)];
    }
}

__global__ void unpermute_kernel(const float* inp, float *out, int B, int N, int NH, int d) {
   // out has shape (B, nh, N, d) but we need to unpermute it to (B, N, nh, d)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // out[b][n][nh_][d_] <- inp[b][nh_][n][d_]
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
        out[other_idx] = inp[idx];
    }
}


__global__ void scale_kernel(float* inp, float scale, int B, int NH, int T) {
    // scales the pre-softmax attention scores by scale
    // and sets the autoregressive locations to -INFINITY
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * NH * T * T) {
        int rest = idx % (NH * T * T);
        rest = rest % (T * T);
        int t2 = rest / T;
        int t = rest % T;
        if (t > t2) {
            inp[idx] = -INFINITY;
        } else {
            inp[idx] *= scale;
        }
    }
}



__global__ void softmax_forward_kernel(float* out, const float* inp, int N, int C) {
    // out is (N, C) just like inp. Each row of inp will get softmaxed.
    // same as kernel3, but can handle any block size (multiple of 32)
    // each row of C elements is handled by block_size threads
    // furthermore, each block_size threads get executed in warps of 32 threads

    // special reduction operations warpReduceMax/warpReduceSum are used for intra-warp reductions
    // shared memory is used for inter-warp reduction
    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int warpId = threadIdx.x / 32; // warp index within a block
    int laneId = threadIdx.x % 32; // thread index within a warp

    // the number of warps per block. recall that blockDim.x is block_size
    int warpsPerBlock = blockDim.x / 32;

    // shared[] must be allocated to have 2 * warpsPerBlock elements
    // first half for max values, the second half for sum values
    float* maxvals = shared;
    float* sumvals = &shared[warpsPerBlock];

    // one row of inp, i.e. inp[idx, :] of shape (C,)
    const float* x = inp + idx * C;

    // first, thread coarsening by directly accessing global memory in series
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x) {
        maxval = fmaxf(maxval, x[i]);
    }
    // now within-warp reductions for maxval
    maxval = warpReduceMax(maxval);

    // the 0th thread of each warp writes the maxval of that warp to shared memory
    if (laneId == 0) maxvals[warpId] = maxval;
    __syncthreads();

    // now the 0th thread reduces the maxvals in shared memory, i.e. across warps
    if (tid == 0) {
        float val = maxvals[tid];
        for (int i = 1; i < warpsPerBlock; i++) {
            val = fmaxf(val, maxvals[i]);
        }
        // store the final max in the first position
        maxvals[0] = val;
    }
    __syncthreads();
    // broadcast the max to all threads
    float offset = maxvals[0];

    // compute expf and write the result to global memory
    for (int i = tid; i < C; i += blockDim.x) {
        // subtract max for numerical stability
        out[idx * C + i] = expf(x[i] - offset);
    }

    // okay now we calculated exp(x - max(x))
    // step 2: sum all the values and divide by the sum

    // thread coarsening for sum
    x = out + idx * C;
    float sumval = 0.0f;
    for (int i = tid; i < C; i += blockDim.x) {
        sumval += x[i];
    }
    // within-warp reduction for sumval
    sumval = warpReduceSum(sumval);

    // write sumval to shared memory
    if (laneId == 0) sumvals[warpId] = sumval;
    __syncthreads();

    // inter-thread reduction of sum
    if (tid == 0) {
        float val = sumvals[tid];
        for (int i = 1; i < warpsPerBlock; ++i) {
            val += sumvals[i];
        }
        sumvals[0] = val;
    }
    __syncthreads();
    // broadcast the sum to all threads
    float sum = sumvals[0];

    // divide the whole row by the sum
    for (int i = tid; i < C; i += blockDim.x) {
        out[idx * C + i] = x[i] / sum;
    }
}

void attention_forward(float* out, float* vaccum, float* qkvr, float* preatt, float* att,
                       const float* inp,
                       int B, int T, int C, int NH,
                       const int block_size) {
    // inp is (B, T, 3C) QKV
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int HS = C / NH; // head size

    // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
    float *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    int total_threads = B * NH * T * HS;
    int num_blocks =ceil_div(total_threads,block_size);
    permute_kernel<<<num_blocks, block_size>>>(q, k, v, inp, B, T, NH, HS);

    // batched matrix multiply with cuBLAS
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasCheck(cublasSgemmStridedBatched(cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            T, T, HS,
                            &alpha,
                            k, HS, T * HS,
                            q, HS, T * HS,
                            &beta,
                            preatt, T, T * T,
                            B * NH));

    // multiply all elements of preatt elementwise by scale
    float scale = 1.0f / sqrtf(HS);
    total_threads = B * NH * T * T;
    num_blocks = ceil_div(total_threads, block_size);
    scale_kernel<<<num_blocks, block_size>>>(preatt, scale, B, NH, T);

    // softmax. preatt is (B, NH, T, T) but we view it as (B * NH * T, T) and use the softmax kernel
    int softmax_block_size = 256;
    int grid_size = B * NH * T;
    size_t shared_mem_size = 2 * softmax_block_size / 32 * sizeof(float);
    softmax_forward_kernel<<<grid_size, softmax_block_size, shared_mem_size>>>(att, preatt, B * NH * T, T);

    // new approach: first cuBLAS another batched matmul
    // y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    cublasCheck(cublasSgemmStridedBatched(cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            HS, T, T,
                            &alpha,
                            v, HS, T * HS,
                            att, T, T * T,
                            &beta,
                            vaccum, HS, T * HS,
                            B * NH));

    // now unpermute
    // y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    num_blocks = ceil_div(B * T * C, block_size);
    unpermute_kernel<<<num_blocks, block_size>>>(vaccum, out, B, T, NH, HS);
}

int main(int argc, char **argv) {
	
	int B = 8;
    int T = 1024;
    int C = 768;
    int NH = 12;

    int deviceIdx = 0;
	

    cudaCheck(cudaSetDevice(deviceIdx)); // device for calling host thread
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);

    // setup cuBLAS
    cublasCreate(&cublas_handle);
	
    // create host memory of random numbers
    float* out = (float*)malloc(B * T * C * sizeof(float));
    float* preatt = (float*)malloc(B * NH * T * T * sizeof(float));
    float* att = (float*)malloc(B * NH * T * T * sizeof(float));
    //float* inp = make_random_float(B * T * 3 * C, 10.0f);
    float* inp = make_random_float(B * T * 3 * C);

    // move to GPU
    float* d_out;
    float* d_vaccum;
    float* d_qkvr;
    float* d_preatt;
    float* d_att;
    float* d_inp;
    cudaCheck(cudaMalloc(&d_out, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_vaccum, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_qkvr, B * T * 3 * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_preatt, B * NH * T * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_att, B * NH * T * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp, B * T * 3 * C * sizeof(float)));
    cudaCheck(cudaMemcpy(d_inp, inp, B * T * 3 * C * sizeof(float), cudaMemcpyHostToDevice));

    int block_sizes[] = {32, 64, 128, 256, 512}; 
    int block_size = block_sizes[0];
        
    attention_forward(d_out,d_vaccum, d_qkvr, d_preatt, d_att, d_inp, B, T, C, NH, block_size);
    printf("success !!");

    free(out);
    free(preatt);
    free(att);
    free(inp);
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_vaccum));
    cudaCheck(cudaFree(d_qkvr));
    cudaCheck(cudaFree(d_preatt));
    cudaCheck(cudaFree(d_att));
    cudaCheck(cudaFree(d_inp));
    cublasDestroy(cublas_handle);

    return 0;
}
```

## How This Code Reduces Time (Compared to Naive GPU code)

This code isn't the full "FlashAttention" algorithm (which uses sophisticated tiling to be I/O-aware for the $S$ and $P$ matrices, often avoiding writing them to HBM at all). However, it employs several key optimization strategies common in high-performance GPU computing:

1.  **Kernel Fusion (Implicit and Explicit):**
    *   **Explicit:** Operations like scaling and masking are combined into a single `scale_kernel`.
    *   **Implicit within `softmax_forward_kernel`:** This kernel is a prime example of fusion. It performs max calculation, subtraction for numerical stability, exponentiation, sum accumulation, and final division all within one kernel launch for each row being softmaxed. This significantly reduces:
        *   **Kernel Launch Overhead:** Each kernel launch has a small CPU-GPU synchronization cost. Fewer launches mean less overhead.
        *   **Memory Traffic:** Intermediate results (like `max_val` and `sum_val` during softmax reduction) are kept in faster on-chip memory (registers, shared memory) instead of being written to and read back from slower global HBM (High Bandwidth Memory).

2.  **Optimized Primitives & Libraries:**
    *   **cuBLAS:** `cublasSgemmStridedBatched` is used for matrix multiplications ($QK^T$ and $PV$). cuBLAS routines are highly optimized by NVIDIA for their hardware, leveraging internal tiling, register blocking, and instruction scheduling for near-peak performance on GEMMs (General Matrix Multiplies).
    *   **Warp Shuffles:** `warpReduceMax` and `warpReduceSum` utilize direct register-to-register communication between threads within the same warp (typically 32 threads). This is much faster for small-scale reductions than going through shared memory or global memory.

3.  **Strategic Shared Memory Usage:** The `softmax_forward_kernel` effectively uses shared memory as a user-managed cache for inter-warp reductions (aggregating max values and sum values from all warps within a block). Shared memory is an on-chip memory space with much higher bandwidth and lower latency than global HBM.

4.  **Coalesced Memory Access (Attempted):** Kernels like `permute_kernel` and `unpermute_kernel` rearrange data. The goal is often to prepare data so that subsequent operations (especially the memory-intensive GEMMs and element-wise kernels) can access global memory in a coalesced pattern. When threads in a warp access contiguous memory locations, the GPU can service these requests in a single (or few) wide memory transaction(s), maximizing effective bandwidth. The success of coalescing depends on the specific access patterns generated by the thread indexing.

5.  **Numerical Stability in Softmax:** Subtracting the maximum value from each element in a row before applying `exp()` in the softmax calculation is crucial for avoiding numerical overflow (if values are large) or underflow/loss of precision (if values are very negative). This ensures the computation remains accurate.

**How "True" FlashAttention (Dao et al.) Goes Further (Conceptual):** (We will cover this in the next chapter)   
The original FlashAttention algorithm takes these ideas, especially memory hierarchy management, to an extreme for the attention calculation itself.
*   **Tiling:** It breaks down the Q, K, V matrices into smaller blocks (tiles).
*   **SRAM Utilization:** It loads blocks of Q, K, V into the GPU's fast on-chip SRAM.
*   **Online Softmax & Recomputation:** It computes one block of the attention output $O$ using these SRAM-local blocks. Crucially, the corresponding blocks of $S = QK^T$ and $P = \text{softmax}(S)$ are computed *on-the-fly but may never be written to HBM*. The softmax normalization (finding max and sum) is done in a streaming/"online" fashion across blocks of $K$ and $V$ for a given block of $Q$. Some values might be recomputed to avoid HBM writes.
*   **I/O Awareness:** This drastically reduces reads/writes of the potentially huge $S$ and $P$ matrices to/from HBM. The memory access complexity for these intermediate matrices moves from $O(T^2)$ (for sequence length $T$) towards something closer to $O(T \cdot d^2 / M_{\text{SRAM}})$, where $M_{\text{SRAM}}$ is the size of SRAM. This makes the attention computation significantly less memory-bound and more compute-bound.

## Miscellaneous Code

Below is the code to check if our implementation matches the true output which is implemeted using cpu only in the function *attention_forward_cpu()*. And also to benchmark time using different *block_size*.

```c++
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <float.h>
#include <cublas_v2.h>
#include <cmath>

cublasHandle_t cublas_handle;

template<class T>
__host__ __device__ T ceil_div(T dividend, T divisor) {
    return (dividend + divisor-1) / divisor;
}


float* make_random_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    return arr;
}

void cuda_check(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
               cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}
#define cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))

// cuBLAS error checking
void cublasCheck(cublasStatus_t status, const char *file, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, line);
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(status) { cublasCheck((status), __FILE__, __LINE__); }

__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}


template<class D, class T>
void validate_result(D* device_result, const T* cpu_reference, const char* name, std::size_t num_elements, T tolerance=1e-4) {
    D* out_gpu = (D*)malloc(num_elements * sizeof(D));
    cudaCheck(cudaMemcpy(out_gpu, device_result, num_elements * sizeof(D), cudaMemcpyDeviceToHost));
    int nfaults = 0;
#ifndef ENABLE_BF16
    float epsilon = FLT_EPSILON;
#else
    float epsilon = 0.079;
#endif
    for (int i = 0; i < num_elements; i++) {
        // Skip masked elements
        if(!isfinite(cpu_reference[i]))
            continue;

        // print the first few comparisons
        if (i < 5) {
            printf("%f %f\n", cpu_reference[i], (T)out_gpu[i]);
        }
        // effective tolerance is based on expected rounding error (epsilon),
        // plus any specified additional tolerance
        float t_eff = tolerance + fabs(cpu_reference[i]) * epsilon;
        // ensure correctness for all elements.
        if (fabs(cpu_reference[i] - (T)out_gpu[i]) > t_eff) {
            printf("Mismatch of %s at %d: CPU_ref: %f vs GPU: %f\n", name, i, cpu_reference[i], (T)out_gpu[i]);
            nfaults ++;
            if (nfaults >= 10) {
                free(out_gpu);
                exit(EXIT_FAILURE);
            }
        }
    }

    if (nfaults > 0) {
        free(out_gpu);
        exit(EXIT_FAILURE);
    }

    free(out_gpu);
}


__global__ void permute_kernel(float* q, float* k, float* v,
                               const float* inp,
                               int B, int N, int NH, int d) {
    // okay so now, this kernel wants Q,K,V to all be of shape (B, NH, N, d)
    // but instead, we have a single tensor QKV (inp) of shape (B, N, 3, NH, d)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Q[b][nh_][n][d_] = inp[b][n][0][nh_][d_]

    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int inp_idx = \
            (b * N * 3 * NH * d)
            +   (n * 3 * NH * d)
            +       (0 * NH * d)
            +          (nh_ * d)
            +                d_;

        q[idx] = inp[inp_idx];
        k[idx] = inp[inp_idx + NH * d];
        v[idx] = inp[inp_idx + 2 * (NH * d)];
    }
}

__global__ void unpermute_kernel(const float* inp, float *out, int B, int N, int NH, int d) {
   // out has shape (B, nh, N, d) but we need to unpermute it to (B, N, nh, d)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // out[b][n][nh_][d_] <- inp[b][nh_][n][d_]
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
        out[other_idx] = inp[idx];
    }
}


__global__ void scale_kernel(float* inp, float scale, int B, int NH, int T) {
    // scales the pre-softmax attention scores by scale
    // and sets the autoregressive locations to -INFINITY
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * NH * T * T) {
        int rest = idx % (NH * T * T);
        rest = rest % (T * T);
        int t2 = rest / T;
        int t = rest % T;
        if (t > t2) {
            inp[idx] = -INFINITY;
        } else {
            inp[idx] *= scale;
        }
    }
}



__global__ void softmax_forward_kernel4(float* out, const float* inp, int N, int C) {
    // out is (N, C) just like inp. Each row of inp will get softmaxed.
    // same as kernel3, but can handle any block size (multiple of 32)
    // each row of C elements is handled by block_size threads
    // furthermore, each block_size threads get executed in warps of 32 threads

    // special reduction operations warpReduceMax/warpReduceSum are used for intra-warp reductions
    // shared memory is used for inter-warp reduction
    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int warpId = threadIdx.x / 32; // warp index within a block
    int laneId = threadIdx.x % 32; // thread index within a warp

    // the number of warps per block. recall that blockDim.x is block_size
    int warpsPerBlock = blockDim.x / 32;

    // shared[] must be allocated to have 2 * warpsPerBlock elements
    // first half for max values, the second half for sum values
    float* maxvals = shared;
    float* sumvals = &shared[warpsPerBlock];

    // one row of inp, i.e. inp[idx, :] of shape (C,)
    const float* x = inp + idx * C;

    // first, thread coarsening by directly accessing global memory in series
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x) {
        maxval = fmaxf(maxval, x[i]);
    }
    // now within-warp reductions for maxval
    maxval = warpReduceMax(maxval);

    // the 0th thread of each warp writes the maxval of that warp to shared memory
    if (laneId == 0) maxvals[warpId] = maxval;
    __syncthreads();

    // now the 0th thread reduces the maxvals in shared memory, i.e. across warps
    if (tid == 0) {
        float val = maxvals[tid];
        for (int i = 1; i < warpsPerBlock; i++) {
            val = fmaxf(val, maxvals[i]);
        }
        // store the final max in the first position
        maxvals[0] = val;
    }
    __syncthreads();
    // broadcast the max to all threads
    float offset = maxvals[0];

    // compute expf and write the result to global memory
    for (int i = tid; i < C; i += blockDim.x) {
        // subtract max for numerical stability
        out[idx * C + i] = expf(x[i] - offset);
    }

    // okay now we calculated exp(x - max(x))
    // step 2: sum all the values and divide by the sum

    // thread coarsening for sum
    x = out + idx * C;
    float sumval = 0.0f;
    for (int i = tid; i < C; i += blockDim.x) {
        sumval += x[i];
    }
    // within-warp reduction for sumval
    sumval = warpReduceSum(sumval);

    // write sumval to shared memory
    if (laneId == 0) sumvals[warpId] = sumval;
    __syncthreads();

    // inter-thread reduction of sum
    if (tid == 0) {
        float val = sumvals[tid];
        for (int i = 1; i < warpsPerBlock; ++i) {
            val += sumvals[i];
        }
        sumvals[0] = val;
    }
    __syncthreads();
    // broadcast the sum to all threads
    float sum = sumvals[0];

    // divide the whole row by the sum
    for (int i = tid; i < C; i += blockDim.x) {
        out[idx * C + i] = x[i] / sum;
    }
}

void attention_forward_cpu(float* out, float* preatt, float* att,
                       const float* inp,
                       int B, int T, int C, int NH) {
    // input is (B, T, 3C) Q,K,V
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int C3 = C*3;
    int hs = C / NH; // head size
    float scale = 1.0 / sqrtf(hs);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                const float* query_t = inp + b * T * C3 + t * C3 + h * hs;
                float* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
                float* att_bth = att + b*NH*T*T + h*T*T + t*T;

                // pass 1: calculate query dot key and maxval
                float maxval = -FLT_MAX;
                for (int t2 = 0; t2 <= t; t2++) {
                    const float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

                    // (query_t) dot (key_t2)
                    float val = 0.0f;
                    for (int i = 0; i < hs; i++) {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;
                    if (val > maxval) {
                        maxval = val;
                    }

                    preatt_bth[t2] = val;
                }
                // pad with -INFINITY outside of autoregressive region for debugging comparisons
                for (int t2 = t+1; t2 < T; t2++) {
                    preatt_bth[t2] = -INFINITY;
                }

                // pass 2: calculate the exp and keep track of sum
                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float expv = expf(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                // pass 3: normalize to get the softmax
                for (int t2 = 0; t2 < T; t2++) {
                    if (t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                    } else {
                        // causal attention mask. not strictly necessary to set to zero here
                        // only doing this explicitly for debugging and checking to PyTorch
                        att_bth[t2] = 0.0f;
                    }
                }

                // pass 4: accumulate weighted values into the output of attention
                float* out_bth = out + b * T * C + t * C + h * hs;
                for (int i = 0; i < hs; i++) { out_bth[i] = 0.0f; }
                for (int t2 = 0; t2 <= t; t2++) {
                    const float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    float att_btht2 = att_bth[t2];
                    for (int i = 0; i < hs; i++) {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
            }
        }
    }
}


void attention_forward_cuda(float* out, float* vaccum, float* qkvr, float* preatt, float* att,
                       const float* inp,
                       int B, int T, int C, int NH,
                       const int block_size) {
    // inp is (B, T, 3C) QKV
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int HS = C / NH; // head size

    // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
    float *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    int total_threads = B * NH * T * HS;
    int num_blocks =ceil_div(total_threads,block_size);
    permute_kernel<<<num_blocks, block_size>>>(q, k, v, inp, B, T, NH, HS);

    // batched matrix multiply with cuBLAS
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasCheck(cublasSgemmStridedBatched(cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            T, T, HS,
                            &alpha,
                            k, HS, T * HS,
                            q, HS, T * HS,
                            &beta,
                            preatt, T, T * T,
                            B * NH));

    // multiply all elements of preatt elementwise by scale
    float scale = 1.0f / sqrtf(HS);
    total_threads = B * NH * T * T;
    num_blocks = ceil_div(total_threads, block_size);
    scale_kernel<<<num_blocks, block_size>>>(preatt, scale, B, NH, T);

    // softmax. preatt is (B, NH, T, T) but we view it as (B * NH * T, T) and use the softmax kernel
    int softmax_block_size = 256;
    int grid_size = B * NH * T;
    size_t shared_mem_size = 2 * softmax_block_size / 32 * sizeof(float);
    softmax_forward_kernel4<<<grid_size, softmax_block_size, shared_mem_size>>>(att, preatt, B * NH * T, T);

    // new approach: first cuBLAS another batched matmul
    // y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    cublasCheck(cublasSgemmStridedBatched(cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            HS, T, T,
                            &alpha,
                            v, HS, T * HS,
                            att, T, T * T,
                            &beta,
                            vaccum, HS, T * HS,
                            B * NH));

    // now unpermute
    // y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    num_blocks = ceil_div(B * T * C, block_size);
    unpermute_kernel<<<num_blocks, block_size>>>(vaccum, out, B, T, NH, HS);
}

// kernel version dispatch
void attention_forward(float* out, float* vaccum,
                       float* qkvr, float* preatt, float* att,
                       float* inp,
                       int B, int T, int C, int NH,
                       const int block_size) {

            attention_forward_cuda(out, vaccum, qkvr, preatt, att, inp, B, T, C, NH, block_size);
}


template<class Kernel, class... KernelArgs>
float benchmark_kernel(int repeats, Kernel kernel, KernelArgs&&... kernel_args) {
    cudaEvent_t start, stop;
    // prepare buffer to scrub L2 cache between benchmarks
    // just memset a large dummy array, recommended by
    // https://stackoverflow.com/questions/31429377/how-can-i-clear-flush-the-l2-cache-and-the-tlb-of-a-gpu
    // and apparently used in nvbench.
    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaCheck(cudaGetDeviceProperties(&deviceProp, deviceIdx));
    void* flush_buffer;
    cudaCheck(cudaMalloc(&flush_buffer, deviceProp.l2CacheSize));

    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    float elapsed_time = 0.f;
    for (int i = 0; i < repeats; i++) {
        // clear L2
        cudaCheck(cudaMemset(flush_buffer, 0, deviceProp.l2CacheSize));
        // now we can start recording the timing of the kernel
        cudaCheck(cudaEventRecord(start, nullptr));
        kernel(std::forward<KernelArgs>(kernel_args)...);
        cudaCheck(cudaEventRecord(stop, nullptr));
        cudaCheck(cudaEventSynchronize(start));
        cudaCheck(cudaEventSynchronize(stop));
        float single_call;
        cudaCheck(cudaEventElapsedTime(&single_call, start, stop));
        elapsed_time += single_call;
    }

    cudaCheck(cudaFree(flush_buffer));

    return elapsed_time / repeats;
}

int main(int argc, char **argv) {
	
	int B = 8;
    int T = 1024;
    int C = 768;
    int NH = 12;

    int deviceIdx = 0;
	

    cudaCheck(cudaSetDevice(deviceIdx)); // device for calling host thread
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);

    // setup cuBLAS (and cuDNN if needed)
    cublasCreate(&cublas_handle);
    int enable_tf32 = deviceProp.major >= 8 ? 1 : 0;
    printf("enable_tf32: %d\n", enable_tf32);
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));
	
	
    // create host memory of random numbers
    float* out = (float*)malloc(B * T * C * sizeof(float));
    float* preatt = (float*)malloc(B * NH * T * T * sizeof(float));
    float* att = (float*)malloc(B * NH * T * T * sizeof(float));
    //float* inp = make_random_float(B * T * 3 * C, 10.0f);
    float* inp = make_random_float(B * T * 3 * C);

    // move to GPU
    float* d_out;
    float* d_vaccum;
    float* d_qkvr;
    float* d_preatt;
    float* d_att;
    float* d_inp;
    cudaCheck(cudaMalloc(&d_out, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_vaccum, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_qkvr, B * T * 3 * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_preatt, B * NH * T * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_att, B * NH * T * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp, B * T * 3 * C * sizeof(float)));
    cudaCheck(cudaMemcpy(d_inp, inp, B * T * 3 * C * sizeof(float), cudaMemcpyHostToDevice));

    int block_sizes[] = {32, 64, 128, 256, 512};

    // Lower accuracy requirements for FP16 (1e-4f also too much for TF32 on kernels 3 & 4)
    float accuracy_threshold = 1e-3f;

    // first check the correctness of the kernel
    attention_forward_cpu(out, preatt, att, inp, B, T, C, NH);
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        attention_forward(d_out,d_vaccum, d_qkvr, d_preatt, d_att, d_inp, B, T, C, NH, block_size);
        // all kernels should produce the correct output out
        // todo - make accuracy threshold dynamic and depend on FP16 vs FP32?
        validate_result(d_out, out, "out", B * T * C, accuracy_threshold);
        // but as for preatt and att, things get a bit more complicated:
        validate_result(d_att, att, "att", B * NH * T * T, accuracy_threshold);
        
        validate_result(d_preatt, preatt, "preatt", B * NH * T * T, accuracy_threshold);
        }
    std::cout<<"All results match. Starting benchmarks.\n\n";


    // benchmark speed of the kernel
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        int repeat_times = 100;

        float elapsed_time = benchmark_kernel(repeat_times, attention_forward,
                                              d_out, d_vaccum, d_qkvr, d_preatt, d_att,
                                              d_inp, B, T, C, NH, block_size);

        printf("block_size %4d | time %f ms\n", block_size, elapsed_time);
    }

    // free memory
    free(out);
    free(preatt);
    free(att);
    free(inp);
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_vaccum));
    cudaCheck(cudaFree(d_qkvr));
    cudaCheck(cudaFree(d_preatt));
    cudaCheck(cudaFree(d_att));
    cudaCheck(cudaFree(d_inp));
    cublasDestroy(cublas_handle);

    return 0;
}
```

## Conclusion

We've journeyed through a CUDA C++ implementation of an optimized multi-head attention mechanism. We saw how:
*   The mathematical formulation of attention translates into a sequence of GPU operations.
*   Data permutations (e.g., `permute_kernel`, `unpermute_kernel`) are important for preparing data for efficient processing by subsequent stages, particularly for aligning with the expectations of libraries like cuBLAS or for achieving better memory access patterns.
*   Specialized libraries like cuBLAS are leveraged for computationally dominant tasks like matrix multiplications, providing highly optimized building blocks.
*   Custom CUDA kernels (`scale_kernel`, `softmax_forward_kernel`) are written to fuse multiple logical operations and exploit GPU architectural features like warp shuffles and shared memory. This minimizes kernel launch overhead and intermediate data movement to/from slow global memory.
*   Even without the full tiling and recomputation strategy of the I/O-aware FlashAttention algorithm, these techniques significantly reduce memory bottlenecks and improve performance compared to a more naive GPU implementation.


See you on the next chapter!