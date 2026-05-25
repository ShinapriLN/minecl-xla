#pragma once

#include <CL/cl.h>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace minecl{

struct OpenClDeviceInfo {
  int id;

  cl_platform_id platform;
  cl_device_id device;

  std::string name;
  std::string vendor;
  std::string device_version;
  std::string driver_version;

  cl_device_type type;
  cl_uint compute_units;
  cl_ulong global_mem_size;
};

absl::StatusOr<std::vector<OpenClDeviceInfo>> EnumerateOpenClDevices();

class OpenClRuntime {
 public:
  static absl::StatusOr<std::shared_ptr<OpenClRuntime>> Create(
    cl_platform_id platform, cl_device_id device
  );

  absl::StatusOr<cl_mem> Allocate(size_t byte_size);

  absl::Status CopyHostToDevice(cl_mem dst, const void* src, size_t byte_size);
  absl::Status CopyDeviceToHost(void* dst, cl_mem src, size_t byte_size);

  absl::Status Finish();

  cl_context context() const;
  cl_command_queue queue() const;
  cl_device_id device() const;

 private:
  cl_platform_id platform_;
  cl_device_id device_;
  cl_context context_;
  cl_command_queue queue_;
};

}// namespace minecl