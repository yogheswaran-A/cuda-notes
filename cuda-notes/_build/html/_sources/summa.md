
## Matrix Multiplication Using cuBLAS.

We wrote our own matrix multiplication kernel in the previous chapter. CuBLAS provides a in built function called *cublasSgemmStridedBatched*.    
All of the information about this can be found here, [cublasSgemmStridedBatched](https://docs.nvidia.com/cuda/cublas/#cublas-t-gemmstridedbatched) and [Nvidia blog](https://developer.nvidia.com/blog/cublas-strided-batched-matrix-multiply/).   
We will make use of it. 

### Definition of `cublasSgemmStridedBatched`
In our code A,B and C are stored contiguously.    
A has a dimesnion of Batchsize x number_of_heads x seqeunce_length x d.     
B has a dimesnion of Batchsize x number_of_heads x seqeunce_length x d.   
C will have dimension of Batchsize x number_of_heads x seqeunce_length x seqeunce_length.    
While performing multiplication we will do it by thinking that A and B has (Batchsize x number_of_heads) number of batches, in each batch we will have a matrix of dimension seqeunce_length x d. The output C will have (Batchsize x number_of_heads) number of batches, in each batch we will have a matrix of dimension seqeunce_length x seqeunce_length.

The `cublasSgemmStridedBatched` function performs batched matrix-matrix multiplication. Mathematically, it computes:

$$
C[i] = \alpha \cdot (A[i]^T \times B[i]) + \beta \cdot C[i], \quad \text{for } i = 0, \dots, \text{Batchsize x number-of-heads} - 1
$$

#### Parameters:
Remember we want to multiply A[i]xB[i] for each of Batchsize x number_of_heads times.
Where:   
   
- A[i], B[i], and C[i] are the i-th matrices in the batch(Batchsize x number_of_heads). 
- α and β are scalar multipliers.
- A[i] x B[i] denotes the matrix multiplication of A[i] and B[i].
- Batchsize x number_of_heads is the number of such matrix multiplications processed in parallel.


```
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
                                  int batchCount)
```

#### Parameters:   
- cublasHandle_t handle: it is the handle which we have discussed about in the earlier section.   
- cublasOperation_t transa : the  parameter specifies whether the matrices A[i] should be as it is, transposed or conjugate-transposed before multiplication.
    transa can take one of the following values:

    - CUBLAS_OP_N:
        No transpose.   
        The matrix A[i] is used as-is.

    - CUBLAS_OP_T:
        Transpose.   
        The matrix A[i]A[i] is transposed, effectively swapping it dimensions to k×mk×m.

    - CUBLAS_OP_C:
        Conjugate transpose.
        The matrix A[i]A[i] is conjugate-transposed (useful for complex numbers).

    If `transa = CUBLAS_OP_T`, the operation modifies the multiplication such that \( A[i] \) is transposed:

$$
C[i] = \alpha \cdot (A[i]^T \times B[i]) + \beta \cdot C[i], \quad \text{for } i = 0, \dots, \text{batchCount x t} - 1
$$  


- m : number of rows of matrix A[i] and C[i]. In attention it will be 

- n : number of columns of op(B[i]) and C[i].

- k : number of columns of op(A[i]) and rows of op(B[i]).