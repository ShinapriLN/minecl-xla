

cc_library(
    name = "opencl_client",
    srcs = [
        "opencl_client.cc",
        "opencl_runtime.cc",
        "opencl_buffer.cc"
    ],
    hdrs = [
        "opencl_client.h",
        "opencl_runtime.h",
        "opencl_buffer.h"
    ],
    visibility = ["//xla/pjrt/c:__pkg__"],
    deps = [
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_device_description",
        "@opencl_headers//:headers",
        "@opencl_lib//:opencl",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)