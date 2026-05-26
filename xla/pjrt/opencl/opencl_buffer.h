#pragma once

#include <memory>   
#include <CL/cl.h>
#include "absl/status/status.h"

#include "xla/pjrt/opencl/opencl_runtime.h"

namespace minecl {

class OpenClBuffer {
 public:
  OpenClBuffer(
    std::shared_ptr<OpenClRuntime> runtime,
    cl_mem mem, size_t byte_size
  );

  ~OpenClBuffer();

  cl_mem mem() const;
  size_t byte_size() const;

  absl::Status CopyFromHost(
    const void* data, 
    size_t offset, 
    size_t byte_size
  );

  absl::Status CopyToHost(
    void* data, 
    size_t offset, 
    size_t byte_size
  );

 private:
  std::shared_ptr<OpenClRuntime> runtime_;
  cl_mem mem_;
  size_t byte_size_;
};

} // namespace minecl
