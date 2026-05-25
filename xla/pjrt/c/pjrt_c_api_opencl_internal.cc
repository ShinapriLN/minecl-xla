#include "xla/pjrt/c/pjrt_c_api_opencl_internal.h"

#include "xla/pjrt/opencl/opencl_client.h"

#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/tsl/platform/statusor.h"

#include "absl/status/status.h"
#include "xla/pjrt/c/pjrt_c_api_status_utils.h"

namespace minecl {

PJRT_Error* PJRT_Client_Create(PJRT_Client_Create_Args* args) {
  PJRT_RETURN_IF_ERROR(pjrt::ActualStructSizeIsGreaterOrEqual(
      "PJRT_Client_Create_Args",
      PJRT_Client_Create_Args_STRUCT_SIZE,
      args->struct_size));

  PJRT_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::PjRtClient> client,
      GetOpenClPjRtClient());

  args->client = pjrt::CreateWrapperClient(std::move(client));

  return nullptr;
}

PJRT_Error* PJRT_OpenClTopology_Create(
    PJRT_TopologyDescription_Create_Args* args) {
  return pjrt::StatusToPjRtError(
    absl::UnimplementedError("OpenCL topology create not supported yet"));
}

const PJRT_Api* GetOpenClPjrtApi() {
  static const PJRT_Api api = pjrt::CreatePjrtApi(
      PJRT_Client_Create,
      nullptr,  // execute_context_create_fn
      PJRT_OpenClTopology_Create,
      pjrt::PJRT_Plugin_Initialize_NoOp
  );
  return &api;
}

}  // namespace minecl
