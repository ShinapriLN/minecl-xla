#include "xla/pjrt/opencl/opencl_runtime.h"
#include <CL/cl.h>
#include <vector>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace minecl {

absl::StatusOr<std::vector<OpenClDeviceInfo>> EnumerateOpenClDevices(){
    std::vector<OpenClDeviceInfo> opencl_device_info;

    cl_uint num_platforms;
    cl_int status; 
    
    status = clGetPlatformIDs(0, NULL, &num_platforms);
    if (status != CL_SUCCESS || num_platforms == 0){
        // no platforms found
        return opencl_device_info;
    }

    cl_platform_id *platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_platforms);
    if (platforms == NULL) {
        // handle error allocation failed
        return opencl_device_info;
    }

    status = clGetPlatformIDs(num_platforms, platforms, NULL);
    if (status != CL_SUCCESS){
        // handle error status not succeed
        free(platforms);
        return opencl_device_info;
    }

    cl_uint num_devices = 0;
    for (cl_uint platform_idx = 0; platform_idx < num_platforms; platform_idx++){
        
        cl_uint num_platform_devices = 0;
        status = clGetDeviceIDs(platforms[platform_idx], CL_DEVICE_TYPE_ALL, 0, NULL, &num_platform_devices);
        if (status != CL_SUCCESS || num_platform_devices == 0) {
            // skip if not success or not found devices for this platforms
            continue;
        }

        cl_device_id *platform_devices = (cl_device_id *)malloc(sizeof(cl_device_id) * num_platform_devices);
        if (platform_devices == NULL){
            // skip if allocation not succeed
            continue;
        }

        status = clGetDeviceIDs(platforms[platform_idx], CL_DEVICE_TYPE_ALL, num_platform_devices, platform_devices, NULL);
        if (status != CL_SUCCESS){
            // skip if failed
            free(platform_devices);
            continue;
        }

        for (cl_uint device_idx = 0; device_idx < num_platform_devices; device_idx++){
            OpenClDeviceInfo device_info;
            device_info.id = int(num_devices + device_idx);
            device_info.platform = platforms[platform_idx];
            device_info.device = platform_devices[device_idx];

            opencl_device_info.push_back(device_info);
        }
        
        num_devices += num_platform_devices;
        free(platform_devices);
    }

    for (cl_uint device_idx = 0; device_idx < num_devices; device_idx++){
        OpenClDeviceInfo device_info = opencl_device_info[device_idx];
        size_t param_value_size;

        param_value_size = 0;
        status = clGetDeviceInfo(device_info.device, CL_DEVICE_NAME, 0, NULL, &param_value_size);
        if (status != CL_SUCCESS){
            // handle err
            continue;
        }

        char *name = (char *)malloc(param_value_size);
        if(name == NULL){
            // handle allocation failed
            continue;
        }
        
        status = clGetDeviceInfo(device_info.device, CL_DEVICE_NAME, param_value_size, name, NULL);
        if (status != CL_SUCCESS){
            // handle err
            free(name);
            continue;
        }

        param_value_size = 0;
        status = clGetDeviceInfo(device_info.device, CL_DEVICE_VENDOR, 0, NULL, &param_value_size);
        if (status != CL_SUCCESS){
            // handle err
            continue;
        }

        char *vendor = (char *)malloc(param_value_size);
        if(vendor == NULL){
            // handle allocation failed
            continue;
        }
        
        status = clGetDeviceInfo(device_info.device, CL_DEVICE_VENDOR, param_value_size, vendor, NULL);
        if (status != CL_SUCCESS){
            // handle err
            free(vendor);
            continue;
        }

        param_value_size = 0;
        status = clGetDeviceInfo(device_info.device, CL_DEVICE_VERSION, 0, NULL, &param_value_size);
        if (status != CL_SUCCESS){
            // handle err
            continue;
        }

        char *device_version = (char *)malloc(param_value_size);
        if(device_version == NULL){
            // handle allocation failed
            continue;
        }
        
        status = clGetDeviceInfo(device_info.device, CL_DEVICE_VERSION, param_value_size, device_version, NULL);
        if (status != CL_SUCCESS){
            // handle err
            free(device_version);
            continue;
        }

        param_value_size = 0;
        status = clGetDeviceInfo(device_info.device, CL_DRIVER_VERSION, 0, NULL, &param_value_size);
        if (status != CL_SUCCESS){
            // handle err
            continue;
        }

        char *driver_version = (char *)malloc(param_value_size);
        if(driver_version == NULL){
            // handle allocation failed
            continue;
        }
        
        status = clGetDeviceInfo(device_info.device, CL_DRIVER_VERSION, param_value_size, driver_version, NULL);
        if (status != CL_SUCCESS){
            // handle err
            free(driver_version);
            continue;
        }

        cl_device_type type;
        status = clGetDeviceInfo(device_info.device, CL_DEVICE_TYPE, sizeof(cl_device_type), &type, NULL);
        if (status != CL_SUCCESS){
            // handle err
            continue;
        }

        cl_uint compute_units;
        status = clGetDeviceInfo(device_info.device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &compute_units, NULL);
        if (status != CL_SUCCESS){
            // handle err
            continue;
        }

        cl_ulong global_mem_size;
        status = clGetDeviceInfo(device_info.device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global_mem_size, NULL);
        if (status != CL_SUCCESS){
            // handle err
            continue;
        }

        opencl_device_info[device_idx].name = name;
        opencl_device_info[device_idx].vendor = vendor;
        opencl_device_info[device_idx].device_version = device_version;
        opencl_device_info[device_idx].driver_version = driver_version;

        opencl_device_info[device_idx].type = type;
        opencl_device_info[device_idx].compute_units = compute_units;
        opencl_device_info[device_idx].global_mem_size = global_mem_size;

        free(name);
        free(vendor);
        free(device_version);
        free(driver_version);
        
    }

    fprintf(stderr, "[minecl] EnumerateOpenClDevices return %zu devices\n",
        opencl_device_info.size());

    return opencl_device_info;
}

static absl::StatusOr<std::shared_ptr<OpenClRuntime>> OpenClRuntime::Create(
    cl_platform_id platform, 
    cl_device_id device
){

    cl_context_properties context_props[] = {
        CL_CONTEXT_PLATFORM,
        reinterpret_cast<cl_context_properties>(platform),
        0,
    };

    cl_int errcode;
    cl_context context;
    cl_command_queue command_queue;

    context = clCreateContext(
        context_props,
        1, &device,
        NULL, NULL,
        &errcode
    )   
    if(errcode != CL_SUCCESS){
        return absl::InternalError("Create::clCreateContext failed");
    }
    
    cl_queue_properties queue_props[] = {
        CL_QUEUE_PROPERTIES,
        0, 0
    }

    queue = clCreateCommandQueueWithProperties(
        context,
        device,
        queue_props,
        &errcode
    )
    if(errcode != CL_SUCCESS){
        clReleaseContext(context);
        return absl::InternalError("Create::clCreateCommandQueueWithProperties failed");
    }

    return std::make_shared<OpenClRuntime>(
        device, platform, context, queue
    )
} 

absl::StatusOr<cl_mem> OpenClRuntime::Allocate(
    size_t byte_size, 
    cl_mem_flags flag
){
    cl_int errcode;
    cl_mem buffer;

    buffer = clCreateBuffer(
        context, flag,
        byte_size, nullptr,
        &errcode
    )
    if (errcode != CL_SUCCESS){
        return absl::InternalError("Allocate failed");
    }

    return buffer;
}

absl::Status OpenClRuntime::CopyHostToDevice(
    cl_mem dst, 
    const void* src, 
    size_t byte_size
){
    cl_int errcode;
    errcode = clEnqueueWriteBuffer(
        queue, dst,
        CL_TRUE,
        0, byte_size,
        src, 0,
        nullptr, nullptr
    )
    if (errcode != CL_SUCCESS) {
        return absl::InternalError("CopyHostToDevice failed");
    }

    return absl::OkStatus();
}

absl::Status OpenClRuntime::CopyDeviceToHost(
    void* dst, 
    cl_mem src, 
    size_t byte_size
){
    cl_int errcode;
    errcode = clEnqueueReadBuffer(
        queue,
        src,
        CL_TRUE,
        0, byte_size,
        dst, 0,
        nullptr, nullptr
    )
    if(errcode != CL_SUCCESS){
        return absl::InternalError("CopyDeviceToHost failed");
    }

    return absl::OkStatus();
}

absl::Status OpenClRuntime::Finish(){
    cl_int errcode;
    errcode = clFinish(queue);
    if (errcode != CL_SUCCESS){
        return absl::InternalError("Finish failed");
    }

    return absl::OkStatus();
}

cl_context OpenClRuntime::context() const{ return context_; }
cl_command_queue OpenClRuntime::queue() const{ return queue_; }
cl_device_id OpenClRuntime::device() const{ return device_; }

} //namespace minecl
