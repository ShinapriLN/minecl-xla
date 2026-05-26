#include <memory>   
#include <CL/cl.h>
#include "absl/status/status.h"

#include "xla/pjrt/opencl/opencl_buffer.h"
#include "xla/pjrt/opencl/opencl_runtime.h"


minecl::OpenClBuffer::OpenClBuffer(
    std::shared_ptr<OpenClRuntime> runtime, 
    cl_mem mem, size_t byte_size
): 
    runtime_(runtime),
    mem_(mem), 
    byte_size_(byte_size) {}

    
minecl::OpenClBuffer::~OpenClBuffer(){
    clReleaseMemObject(mem_);
}

cl_mem minecl::OpenClBuffer::mem() const { return mem_; }

size_t minecl::OpenClBuffer::byte_size() const { return byte_size_; }

absl::Status minecl::OpenClBuffer::CopyFromHost(
    const void* data, 
    size_t offset,
    size_t byte_size
){
    cl_int errcode;
    errcode = clEnqueueWriteBuffer(
        runtime_->queue(),
        mem_, CL_TRUE,
        offset, byte_size,
        data, 0,
        NULL, NULL
    );
    if(errcode != CL_SUCCESS){
        return absl::InternalError("CopyFromHost failed");
    }

    return absl::OkStatus();
}

absl::Status minecl::OpenClBuffer::CopyToHost(
    void* data, 
    size_t offset,
    size_t byte_size
){
    cl_int errcode;
    errcode = clEnqueueReadBuffer(
        runtime_->queue(),
        mem_, CL_TRUE,
        offset, byte_size,
        data, 0,
        NULL, NULL
    );
    if(errcode != CL_SUCCESS){
        return absl::InternalError("CopyToHost failed");
    }

    return absl::OkStatus();
}
