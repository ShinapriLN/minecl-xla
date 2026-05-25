#pragma once

#include <CL/cl.h>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

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