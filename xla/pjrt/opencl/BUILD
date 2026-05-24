cc_library(
    name = "opencl_client",
    srcs = ["opencl_client.cc"],
    hdrs = ["opencl_client.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/pjrt:pjrt_client",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)