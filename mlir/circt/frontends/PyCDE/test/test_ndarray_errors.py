# RUN: %PYTHON% py-split-input-file.py %s | FileCheck %s

from pycde import Module, Input, generator
from pycde.testing import unittestmodule
from pycde.types import types, dim
from pycde.ndarray import NDArray

# Missing assignment


@unittestmodule()
class M1(Module):
  in1 = Input(types.i32)

  @generator
  def build(ports):
    m = NDArray((10), dtype=types.i32, name='m1')
    for i in range(9):
      m[i] = ports.in1
    # CHECK: ValueError: Unassigned sub-matrices:
    # CHECK: {{[[]}}{{[[]}}9{{[]]}}{{[]]}}
    m.to_circt()


# -----

# dtype mismatch


@unittestmodule()
class M1(Module):
  in1 = Input(types.i33)

  @generator
  def build(ports):
    m = NDArray((32), dtype=types.i32, name='m1')
    # CHECK: ValueError: Width mismatch between provided BitVectorValue (Bits<33>) and target shape (Bits<32>).
    m[0] = ports.in1


# -----

# Invalid constructor


@unittestmodule()
class M1(Module):
  in1 = Input(dim(types.i32, 10))

  @generator
  def build(ports):
    # CHECK: ValueError: Must specify either shape and dtype, or initialize from a value, but not both.
    NDArray((10, 32), from_value=ports.in1, dtype=types.i1, name='m1')


# -----

# Cast mismatch


@unittestmodule()
class M1(Module):
  in1 = Input(types.i31)

  @generator
  def build(ports):
    m = NDArray((32, 32), dtype=types.i1, name='m1')
    # CHECK: ValueError: Width mismatch between provided BitVectorValue (Bits<31>) and target shape (Bits<1>[32]).
    m[0] = ports.in1
