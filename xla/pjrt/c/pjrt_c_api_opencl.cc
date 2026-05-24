#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_opencl_internal.h"

extern "C" const PJRT_Api* GetPjrtApi() {
  return minecl::GetOpenClPjrtApi();
}