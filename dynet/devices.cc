#include "dynet/devices.h"

#include <boost/algorithm/string.hpp>
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>

#include "dynet/cuda.h"
#include "dynet/dynet.h"
#include "dynet/expr.h"

using namespace std;

namespace dynet {

DeviceMempoolSizes::DeviceMempoolSizes(size_t total_size) {
  used[0] = total_size/3;
  used[1] = total_size/3;
  used[2] = total_size/3;
}

DeviceMempoolSizes::DeviceMempoolSizes(size_t fx_s, size_t dEdfs_s, size_t ps_s) {
  used[0] = fx_s;
  used[1] = dEdfs_s;
  used[2] = ps_s;
}

DeviceMempoolSizes::DeviceMempoolSizes(const std::string & descriptor) {
  vector<string> strs;
  boost::algorithm::split(strs, descriptor, boost::is_any_of(","));
  if(strs.size() == 1) {
    size_t total_size = stoi(strs[0]);
    used[0] = total_size/3;
    used[1] = total_size/3;
    used[2] = total_size/3;
  } else if(strs.size() == 3) {
    used[0] = stoi(strs[0]);
    used[1] = stoi(strs[1]);
    used[2] = stoi(strs[2]);
  }
}

Device::~Device() {}

DeviceMempoolSizes Device::mark(ComputationGraph *cg) {
  cg->incremental_forward({cg, (VariableIndex)(cg->nodes.size() - 1)}); // needed so that we actually allocate the needed memory
                             // for all existing nodes.
  return DeviceMempoolSizes(pools[0]->used, pools[1]->used, pools[2]->used);
}

void Device::revert(const DeviceMempoolSizes & cp) {
  assert(cp.used[0] <= pools[0]->used);
  pools[0]->used = cp.used[0];
  assert(cp.used[1] <= pools[1]->used);
  pools[1]->used = cp.used[1];
  assert(cp.used[2] <= pools[2]->used);
  pools[2]->used = cp.used[2];
}

void Device::allocate_tensor(DeviceMempool mp, Tensor & tens) {
  assert(mp != DeviceMempool::NONE);
  assert(pools[(int)mp] != nullptr);
  tens.v = (float*)pools[(int)mp]->allocate(tens.d.size() * sizeof(float));
  tens.mem_pool = mp;
}

#if HAVE_CUDA
Device_GPU::Device_GPU(int my_id, const DeviceMempoolSizes & mbs, int device_id) :
    Device(my_id, DeviceType::GPU, &gpu_mem), cuda_device_id(device_id), gpu_mem(device_id) {
  CUDA_CHECK(cudaSetDevice(device_id));
  CUBLAS_CHECK(cublasCreate(&cublas_handle));
  CUBLAS_CHECK(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE));
  kSCALAR_MINUSONE = (float*)gpu_mem.malloc(sizeof(float));
  kSCALAR_ONE = (float*)gpu_mem.malloc(sizeof(float));
  kSCALAR_ZERO = (float*)gpu_mem.malloc(sizeof(float));
  float minusone = -1;
  CUDA_CHECK(cudaMemcpyAsync(kSCALAR_MINUSONE, &minusone, sizeof(float), cudaMemcpyHostToDevice));
  float one = 1;
  CUDA_CHECK(cudaMemcpyAsync(kSCALAR_ONE, &one, sizeof(float), cudaMemcpyHostToDevice));
  float zero = 0;
  CUDA_CHECK(cudaMemcpyAsync(kSCALAR_ZERO, &zero, sizeof(float), cudaMemcpyHostToDevice));

  // Initialize the Eigen device
  estream = new Eigen::CudaStreamDevice(device_id);
  edevice = new Eigen::GpuDevice(estream);

  // this is the big memory allocation.
  for(size_t i = 0; i < 3; ++i)
    pools[i] = new AlignedMemoryPool((mbs.used[i] << 20), &gpu_mem);
}

Device_GPU::~Device_GPU() {}
#endif

Device_CPU::Device_CPU(int my_id, const DeviceMempoolSizes & mbs, bool shared) :
    Device(my_id, DeviceType::CPU, &cpu_mem), shmem(mem) {
  if (shared) shmem = new SharedAllocator();
  kSCALAR_MINUSONE = (float*) mem->malloc(sizeof(float));
  *kSCALAR_MINUSONE = -1;
  kSCALAR_ONE = (float*) mem->malloc(sizeof(float));
  *kSCALAR_ONE = 1;
  kSCALAR_ZERO = (float*) mem->malloc(sizeof(float));
  *kSCALAR_ZERO = 0;

  // Initialize the Eigen device
  edevice = new Eigen::DefaultDevice;

  // this is the big memory allocation.
  for(size_t i = 0; i < 2; ++i)
    pools[i] = new AlignedMemoryPool((mbs.used[i] << 20), &cpu_mem);
  pools[2] = new AlignedMemoryPool((mbs.used[2] << 20), shmem);
}

Device_CPU::~Device_CPU() {}

} // namespace dynet
