# REQUIRES: esi-cosim
# XFAIL: *
# RUN: rm -rf %t
# RUN: mlir-opt  %S/dot.linalg.mlir --empty-tensor-to-alloc-tensor --one-shot-bufferize="allow-return-allocs bufferize-function-boundaries" --buffer-results-to-out-params --convert-linalg-to-affine-loops --lower-affine --convert-scf-to-cf --canonicalize > dot.cf.mlir
# RUN: hlstool dot.cf.mlir --with-esi --dynamic-hw -ir -ir-output-level 2 > dot.hw.mlir
# RUN: %PYTHON% %s %t 2>&1
# RUN: esi-cosim-runner.py --no-aux-files --tmpdir %t --schema %t/runtime/schema.capnp %s `ls %t/hw/*.sv | grep -v driver.sv`
# PY: from dot_prod_system import run_cosim
# PY: run_cosim(tmpdir, rpcschemapath, simhostport)

from pycde import Input, Module, generator, types
from pycde.common import Clock
from pycde.system import System
from pycde.esi import FromServer, ToFromServer, ServiceDecl, CosimBSP
from pycde.constructs import Wire

# import torch
# import torch_mlir
import numpy as np

import pathlib
import sys
import time
from typing import List

__dir__ = pathlib.Path(__file__).parent

################################################################################
# Harware design
################################################################################

# Run this offline, lower it to CF/Arit/Memref, commit output. Avoids needing
# 'torch_mlir' and provides a bit more stability.

# class DotModule(torch.nn.Module):

#   def forward(self, a, b):
#     return torch.matmul(a, b)

# shape = torch_mlir.TensorPlaceholder([5], torch.int32)
# torch_module = torch_mlir.compile(DotModule(), [shape, shape],
#                                   output_type="linalg-on-tensors")


class Gasket(Module):
  """Wrap the accelerator IP module. Instantiate the requiste memories. Wire the
  memories to the host and the host to the module control signals."""

  clk = Clock()
  rst = Input(types.i1)

  @generator
  def generate(ports):
    # Get a 'go' signal from the host.
    go = TorchControl.go("go")

    # Instantiate the accelerator IP, passing it the 'go' signal from the host.
    # All other communication is done through ESI services.
    DotAccelIP(clock=ports.clk, reset=ports.rst, in3=go)

    # Implement the three memories which the IP needs with simple SystemVerilog
    # unpacked arrays.
    dot_a.instantiate_builtin("sv_mem", inputs=[ports.clk, ports.rst])
    dot_b.instantiate_builtin("sv_mem", inputs=[ports.clk, ports.rst])
    dot_x.instantiate_builtin("sv_mem", inputs=[ports.clk, ports.rst])

    # Give the host access to the memories. Write access to the 'a' and 'b'
    # memories and read access to the 'x' memory.
    dot_a.write(TorchControl.a_write("a"))
    dot_b.write(TorchControl.b_write("b"))
    read_address = Wire(dot_x.read.to_server_type)
    read_data = dot_x.read(read_address)
    read_address.assign(TorchControl.x_read(read_data, "x"))


if __name__ == "__main__":
  system = System(CosimBSP(Gasket),
                  name="PyTorchDotProd",
                  output_directory=sys.argv[1])

  # Import the torch_mlir module.
  syms = system.import_mlir(open("dot.hw.mlir").read())

  # Grab references to the imported IP and requested memories which must
  # implemented in a gasket/wrapper.
  DotAccelIP = syms["forward_esi_wrapper"]
  dot_a: ServiceDecl = syms["in0"]
  dot_b: ServiceDecl = syms["in1"]
  dot_x: ServiceDecl = syms["in2"]

  # Define an interface (API) for software.
  @ServiceDecl
  class TorchControl:
    go = FromServer(types.i0)
    a_write = FromServer(dot_a.write.to_server_type)
    b_write = FromServer(dot_b.write.to_server_type)
    x_read = ToFromServer(dot_x.read.to_client_type, dot_x.read.to_server_type)

  # Compile and package up.
  system.compile()
  system.package()

################################################################################
# Software runtime
################################################################################


def write_vector(vec: List[int], port):
  for i, v in enumerate(vec):
    port.write({"address": i, "data": v})


def hw_dotprod(hw, a: List[int], b: List[int]):
  # Write the two vectors to device memories.
  write_vector(a, hw.bsp.a_write[0])
  write_vector(b, hw.bsp.b_write[0])

  # Tell the dot module to go!
  hw.bsp.go[0].write()

  # Wait for unit to compute.
  #       (Hack around missing functionality in XRT bridge.)
  time.sleep(0.01)

  # Read the result.
  x = hw.bsp.x_read[0]()
  print(f"{a} x {b} = {x}")
  return x


def rand_vec():
  return [np.random.randint(0, 100) for _ in range(5)]


def run_cosim(tmpdir=".", schema_path="schema.capnp", rpchostport=None):
  import os
  sys.path.append(os.path.join(tmpdir, "runtime"))
  import PyTorchDotProd as esi_sys
  from PyTorchDotProd.common import Cosim
  if rpchostport is None:
    port = open("cosim.cfg").read().split(':')[1].strip()
    rpchostport = f"localhost:{port}"

  # Connect to RTL simulator via cosimulation.
  acc_conn = Cosim(schema_path, rpchostport)

  # Instantiate the accelerator host API with the backend connection.
  hw = esi_sys.top(acc_conn)

  # Run a simple dot product check.
  hwdp = hw_dotprod(hw, [1, 1, 1, 1, 1], [1, 1, 1, 1, 1])
  assert hwdp == 5, "Basic sanity test FAILED"

  # Instantiate PyTorch module for golden model.
  for _ in range(25):
    a = rand_vec()
    b = rand_vec()

    # Compute with our accelerator.
    hdp = hw_dotprod(hw, a, b)

    # Compute known good result.
    swdp = np.dot(a, b)

    assert hdp == swdp, f"  INCORRCT result. Correct is {swdp}"
