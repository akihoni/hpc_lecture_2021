/***************************************
source /etc/profile.d/modules.sh
module load vim cmake gcc cuda/11.2.146 openmpi nccl cudnn intel
nvcc mpi_cuda.cu -lmpi
./a.out
*****************************************/
#include <mpi.h>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <chrono>
using namespace std;

__global__ void matmul(float *A, float *B, float *C, int N) {
  int i = blockIdx.y;
  int j = threadIdx.x + blockDim.x * blockIdx.x;
  float sum = 0.0f;
  extern __shared__ float A_s[];
  for (int ks=0; ks<N; ks+=blockDim.x) {
    __syncthreads();
    A_s[threadIdx.x] = A[N*i+ks+threadIdx.x];
    __syncthreads();
    for (int k=ks; k<ks+blockDim.x; k++) {
      sum += A_s[k-ks] * B[N*k+j];
    }
  }
  C[N*i+j] = sum;
}

int main(int argc, char **argv){
  int mpisize, mpirank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);

  const int N = 1024, M = 512; 
  int gpu_size = N * N * sizeof(float);
  float *A, *B, *C;   
  cudaMallocManaged(&A, gpu_size);
  cudaMallocManaged(&B, gpu_size);
  cudaMallocManaged(&C, gpu_size);

  for (int i=0; i<N; i++){
    for (int j=0; j<N; j++){
      A[N*i+j] = drand48();
      B[N*i+j] = drand48();
      C[N*i+j] = 0;
    }
  }
  double comp_time = 0, comm_time = 0;
  for(int irank=0; irank<mpisize; irank++) {
    dim3 grid(N/M, N);
    auto tic = chrono::steady_clock::now();
    matmul<<<grid,M,M*sizeof(float)>>>(A, B, C, N);
    cudaDeviceSynchronize();
    auto toc = chrono::steady_clock::now();
    comp_time += chrono::duration<double>(toc - tic).count();
    tic = chrono::steady_clock::now();
    comm_time += chrono::duration<double>(tic - toc).count();
  }
#pragma opt parallel for
  for (int i=0; i<N; i++){
    for (int j=0; j<N; j++){
      for (int k=0; k<N; k++){
        C[N*i+j] -= A[N*i+k] * B[N*k+j];
      }
    }
  }
  double err = 0;
  for (int i=0; i<N; i++){
    for (int j=0; j<N; j++){
      err += fabs(C[N*i+j]);
    }   
  }
  if(mpirank==0) {
    double time = comp_time+comm_time;
    printf("N    : %d\n",N);
    printf("comp : %lf s\n", comp_time);
    printf("comm : %lf s\n", comm_time);
    printf("total: %lf s (%lf GFlops)\n",time,2.*N*N*N/time/1e9);
    printf("error: %lf\n",err/N/N);
									      }
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  MPI_Finalize();
}
