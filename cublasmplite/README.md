# Quickstart - standalone AG+GEMM or GEMM+RS tests

## Requirements
- MPI (`MPI_HOME`)
- NVSHMEM (`NVSHMEM_HOME`)
- NCCL (`NCCL_HOME`)
- CUDA Toolkit (`CUDACXX=...`) for NVCC and cuBLAS

## Build
```
mkdir cublasmplite/build
cd cublasmplite/build
CXX=g++-11 MPI_HOME=/usr/ cmake -DCMAKE_CUDA_ARCHITECTURES=80-real -DNVSHMEM_HOME=/home/scratch.lcambier_ent/nvshmem-2.11.0/ -DNCCL_HOME=/home/scratch.lcambier_ent/nccl_2.21.5-1+cuda12.4_x86_64/ ..
make -j8
```

## Run

### AG+GEMM
```
$ mpirun -np 2 -tag-output -x NVSHMEM_REMOTE_TRANSPORT=none -x NVSHMEM_BOOTSTRAP=MPI -x NVSHMEM_DISABLE_NCCL=1 ./tests/ag_gemm -m 32 -n 64 -k 128
[1,1]<stdout>:MPI Hello from 1/2
[1,0]<stdout>:MPI Hello from 0/2
[1,0]<stdout>:AG+GEMM:
[1,0]<stdout>:num_ranks 2
[1,0]<stdout>:m 32
[1,0]<stdout>:n 64
[1,0]<stdout>:k 128
[1,0]<stdout>:cycles 10
[1,0]<stdout>:skip 5
[1,0]<stdout>:Performance:
[1,0]<stdout>:NVSHMEM (max) 0.047718 ms
[1,0]<stdout>:NVSHMEM (average) 0.046848 ms
[1,0]<stdout>:NCCL (max) 0.026214 ms
[1,0]<stdout>:NCCL (average) 0.026163 ms
[1,0]<stdout>:PASSED
```

### GEMM+RS
```
$ mpirun -np 2 -tag-output -x NVSHMEM_REMOTE_TRANSPORT=none -x NVSHMEM_BOOTSTRAP=MPI -x NVSHMEM_DISABLE_NCCL=1 ./tests/gemm_rs -m 32 -n 64 -k 128
[1,0]<stdout>:MPI Hello from 0/2
[1,1]<stdout>:MPI Hello from 1/2
[1,0]<stdout>:GEMM+RS:
[1,0]<stdout>:num_ranks 2
[1,0]<stdout>:m 32
[1,0]<stdout>:n 64
[1,0]<stdout>:k 128
[1,0]<stdout>:cycles 10
[1,0]<stdout>:skip 5
[1,0]<stdout>:Performance:
[1,0]<stdout>:NVSHMEM (max) 0.051712 ms
[1,0]<stdout>:NVSHMEM (average) 0.051610 ms
[1,0]<stdout>:NCCL (max) 0.021914 ms
[1,0]<stdout>:NCCL (average) 0.021914 ms
[1,0]<stdout>:PASSED
```

# Quickstart - inside of TransformerEngine

## Requirements
- MPI (`MPI_HOME`)
- NVSHMEM (`NVSHMEM_HOME`)
- NCCL (`NCCL_HOME`)
- CUDA Toolkit (`CUDACXX=...`) for NVCC and cuBLAS

MPI and NCCL are only used by the tests.

## Git clone, copy NVSHMEM + start docker

Clone repo
```
git clone -b lcambier/ub_nvshmem --recurse-submodules ssh://git@gitlab-master.nvidia.com:12051/lcambier/TransformerEngine.git TransformerEngine
cd TransformerEngine
```

Install NVSHMEM
```
wget https://developer.download.nvidia.com/compute/redist/nvshmem/2.11.0/builds/cuda12/txz/agnostic/x64/libnvshmem_2.11.0-5+cuda12.0_x86_64.txz
tar -xvf libnvshmem_2.11.0-5+cuda12.0_x86_64.txz
```
NVSHMEM is now installed in `libnvshmem_2.11.0-5+cuda12.0_x86_64`.

Start docker
```
docker run -it -v $(pwd):/workdir --privileged --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus all gitlab-master.nvidia.com:5005/dl/dgx/pytorch:master-py3-devel bash -i
```

## Build cuBLASMplite

`cuBLASMplite` (naming is hard) is a small library that encapsultes NVSHMEM ops and a little more.

```
mkdir -p /workdir/cublasmplite/build
cd /workdir/cublasmplite/build
NVSHMEM_HOME=/workdir/libnvshmem_2.11.0-5+cuda12.0_x86_64 cmake -DCMAKE_INSTALL_PREFIX=/workdir/cublasmplite/install  -DCMAKE_CUDA_ARCHITECTURES=90-real ..
make install -j8
```
cuBLASMplite is now installed in `/workdir/cublasmplite/install/`.

## Build TE + cuBLASMplite

```
cd /workdir
CPATH=/workdir/cublasmplite/install/include:$CPATH CUBLASMPLITE_HOME=/workdir/cublasmplite/install/ NVTE_FRAMEWORK=pytorch NVTE_WITH_USERBUFFERS=1 MPI_HOME=/opt/hpcx/ompi/ pip install --verbose -e .[test]
```

## Run tests

With UB
```
root@64642578d1c9:/workdir# UB_SKIPMC=1 LD_LIBRARY_PATH=/workdir/libnvshmem_2.11.0-5+cuda12.0_x86_64/lib:/workdir/cublasmplite/install/lib/:$LD_LIBRARY_PATH torchrun --nproc-per-node=4 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics --p2p --comm-type ag
W0708 20:46:45.577000 139799205713024 torch/distributed/run.py:778]
W0708 20:46:45.577000 139799205713024 torch/distributed/run.py:778] *****************************************
W0708 20:46:45.577000 139799205713024 torch/distributed/run.py:778] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
W0708 20:46:45.577000 139799205713024 torch/distributed/run.py:778] *****************************************
Rank 3/4, hello from test_gemm.py
Rank 2/4, hello from test_gemm.py
Rank 0/4, hello from test_gemm.py
Rank 1/4, hello from test_gemm.py
UB_TIMEOUT is set to 110 sec, 155100000000 cycles, freq: 1410000khz
MC NOT initialized and used
!!! [CommGemmOverlap] communicator initialized
!!! [CommGemmOverlap] registered buffer 1

[GLOBAL] NUMERICAL CHECK PASSED: max error = 0.0005281120538711548

[rank:0] Avg. GPU time for p2p all-gather + GEMM: 97.55648040771484 ms
[rank:2] Avg. GPU time for p2p all-gather + GEMM: 122.86463928222656 ms
[rank:3] Avg. GPU time for p2p all-gather + GEMM: 118.12351989746094 ms
[rank:1] Avg. GPU time for p2p all-gather + GEMM: 110.62477111816406 ms

```

With NVSHMEM
```
root@64642578d1c9:/workdir# NVTE_NVSHMEM=1 NVSHMEM_DISABLE_NCCL=1 NVSHMEM_REMOTE_TRANSPORT=none LD_LIBRARY_PATH=/workdir/libnvshmem_2.11.0-5+cuda12.0_x86_64/lib:/workdir/cublasmplite/install/lib/:$LD_LIBRARY_PATH torchrun --nproc-per-node=4 tests/pytorch/distributed/run_gemm_with_overlap.py --check-numerics --p2p --comm-type ag
W0708 20:46:10.589000 140139100075136 torch/distributed/run.py:778]
W0708 20:46:10.589000 140139100075136 torch/distributed/run.py:778] *****************************************
W0708 20:46:10.589000 140139100075136 torch/distributed/run.py:778] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
W0708 20:46:10.589000 140139100075136 torch/distributed/run.py:778] *****************************************
Rank 2/4, hello from test_gemm.py
Rank 3/4, hello from test_gemm.py
Rank 1/4, hello from test_gemm.py
Rank 0/4, hello from test_gemm.py
UID bootstrap network already initialized using:  eno1:10.112.216.230<0>

UID bootstrap network already initialized using:  eno1:10.112.216.230<0>

UID bootstrap network already initialized using:  eno1:10.112.216.230<0>
UID bootstrap network already initialized using:  eno1:10.112.216.230<0>

UID bootstrap network already initialized using:  eno1:10.112.216.230<0>



[GLOBAL] NUMERICAL CHECK PASSED: max error = 0.0005281120538711548

[rank:0] Avg. GPU time for p2p all-gather + GEMM: 104.06195068359375 ms
[rank:3] Avg. GPU time for p2p all-gather + GEMM: 97.46841430664062 ms
[rank:1] Avg. GPU time for p2p all-gather + GEMM: 100.1707534790039 ms
[rank:2] Avg. GPU time for p2p all-gather + GEMM: 118.84544372558594 ms

```