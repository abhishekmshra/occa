#include <iostream>

#include <occa.hpp>
#include <occa/modes/hip/utils.hpp>

#include <hip_runtime_api.h>

int main(int argc, char **argv) {
  int entries = 5;

  //---[ Init CUDA ]------------------
  int hipDeviceID = 0;
  hipStream_t hipStream;
  void *hip_a, *hip_b, *hip_ab;

  // Default: hipStream = 0
  hipStreamCreate(&hipStream);

  hipMalloc(&hip_a , entries * sizeof(float));
  hipMalloc(&hip_b , entries * sizeof(float));
  hipMalloc(&hip_ab, entries * sizeof(float));

  //  ---[ Get CUDA Info ]----
  //hipDevice_t hipDevice;
  //hipCtx_t hipCtx;

  //hipDeviceGet(&hipDevice, hipDeviceID);
  //hipCtxGetCurrent(&hipCtx);
  //  ========================
  //====================================

  float *a  = new float[entries];
  float *b  = new float[entries];
  float *ab = new float[entries];

  //occa::device device; // = occa::hip::wrapDevice(hipDevice, hipCtx);

  occa::kernel addVectors;
  occa::memory o_a, o_b, o_ab;

  for (int i = 0; i < entries; ++i) {
    a[i]  = i;
    b[i]  = 1 - i;
    ab[i] = 0;
  }

  //device.setup("mode: 'HIP'");

  occa::properties deviceProps;
  deviceProps["mode"] = "HIP";
  deviceProps["device_id"] = 0;
  occa::device device(deviceProps);

  o_a  = occa::hip::wrapMemory(device, hip_a , entries * sizeof(float));
  o_b  = occa::hip::wrapMemory(device, hip_b , entries * sizeof(float));
  o_ab = occa::hip::wrapMemory(device, hip_ab, entries * sizeof(float));

  addVectors = device.buildKernel("addVectors.okl",
                                  "addVectors");

  o_a.copyFrom(a);
  o_b.copyFrom(b);

  addVectors(entries, o_a, o_b, o_ab);

  o_ab.copyTo(ab);

  for (int i = 0; i < entries; ++i) {
    std::cout << i << ": " << ab[i] << '\n';
  }

  delete [] a;
  delete [] b;
  delete [] ab;

  return 0;
}
