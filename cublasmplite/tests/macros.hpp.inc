#include <iostream>

#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(err)                                                                      \
do {                                                                                           \
    cublasStatus_t err_ = (err);                                                               \
    if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
        std::printf("cuBLAS error %s:%d, error %d (%s)\n", __FILE__, __LINE__, (int)err_, cublasGetStatusString(err_)); \
        exit(EXIT_FAILURE);                                                                    \
    }                                                                                          \
} while (0)
#endif

#ifndef CUDA_CHECK
#define CUDA_CHECK(cmd)                                                                         \
  do {                                                                                             \
    cudaError_t retval = cmd;                                                                      \
    if (retval != 0) {                                                                             \
      printf("Failed: CUDART error %s:%d, error %d (%s)\n", __FILE__, __LINE__, (int)retval, cudaGetErrorString(retval)); \
      exit(EXIT_FAILURE);                                                                          \
    }                                                                                              \
  } while (0);
#endif

#ifndef ASSERT
#define ASSERT(cmd)                                                                                \
  do {                                                                                             \
    bool retval = cmd;                                                                             \
    if (!retval) {                                                                                 \
      printf("Failed: ASSERT failed at %s:%d\n", __FILE__, __LINE__);                              \
      exit(EXIT_FAILURE);                                                                          \
    }                                                                                              \
  } while (0);
#endif

#ifndef ASSERT_EQ
#define ASSERT_EQ(lhs, rhs)                                                                        \
  do {                                                                                             \
    auto __lhs = lhs;                                                                              \
    auto __rhs = rhs;                                                                              \
    if (__lhs != __rhs) {                                                                          \
      printf("Failed: ASSERT_EQ failed at %s:%d, got:\n", __FILE__, __LINE__);                    \
      std::cout << __lhs << " != " << __rhs << "\n";                                              \
      exit(EXIT_FAILURE);                                                                          \
    }                                                                                              \
  } while (0);
#endif

#ifndef MPI_CHECK
#define MPI_CHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)
#endif

#ifndef NCCL_CHECK
#define NCCL_CHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)
#endif