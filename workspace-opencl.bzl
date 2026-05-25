


def workspace():
    native.new_local_repository(
        name = "opencl_headers",
        path = "/usr/include/CL",
        build_file_content = """cc_library(
    name = "headers",
    hdrs = glob(["*.h"]),
    include_prefix = "CL",
    visibility = ["//visibility:public"],
)
""",
    )

    native.new_local_repository(
        name = "opencl_lib",
        path = "/usr/lib",
        build_file_content = """cc_import(
    name = "opencl",
    shared_library = "libOpenCL.so",
    visibility = ["//visibility:public"],
)
""",
    )

opencl_workspace = workspace