import os
import jax._src.xla_bridge as xb

def initialize():
    path = os.path.join(os.path.dirname(__file__), "jax_opencl_pjrt.so")
    xb.register_plugin("opencl", priority=500, library_path=path, options=None)