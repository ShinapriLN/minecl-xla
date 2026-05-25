#pragma once

#include <memory>

#include "absl/status/statusor.h"
#include "xla/pjrt/pjrt_client.h"

namespace minecl {

absl::StatusOr<std::unique_ptr<xla::PjRtClient>> GetOpenClPjRtClient();

}  // namespace minecl