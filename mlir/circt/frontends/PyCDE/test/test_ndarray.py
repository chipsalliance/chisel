# RUN: %PYTHON% py-split-input-file.py %s | FileCheck %s

from pycde import Input, Output, generator, Module
from pycde.testing import unittestmodule

from pycde.ndarray import NDArray
from pycde.dialects import hw
from pycde.types import types, dim

# ndarray transposition via injected ndarray on a ListValue
# Putting this as first test in case users use this file as a reference.
# The majority of tests in this file uses the ndarray directly, but most users
# will probably want to use the Numpy features directly on the ListValue.


@unittestmodule()
class M1(Module):
  in1 = Input(dim(types.i32, 4, 8))
  out = Output(dim(types.i32, 8, 4))

  @generator
  def build(ports):
    ports.out = ports.in1.transpose((1, 0))


# -----

# Composing multiple ndarray transformations


@unittestmodule()
class M1(Module):
  in1 = Input(dim(types.i32, 4, 8))
  out = Output(dim(types.i32, 2, 16))

  @generator
  def build(ports):
    ports.out = ports.in1.transpose((1, 0)).reshape((16, 2))


# -----


@unittestmodule()
class M2(Module):
  in0 = Input(dim(types.i32, 16))
  in1 = Input(types.i32)
  t_c = dim(types.i32, 8, 4)
  c = Output(t_c)

  @generator
  def build(ports):
    # a 32xi32 ndarray.
    m = NDArray((32,), dtype=types.i32, name='m2')
    for i in range(16):
      m[i] = ports.in1
    m[16:32] = ports.in0
    m = m.reshape((4, 8))
    ports.c = m.to_circt()


# -----
@unittestmodule()
class M5(Module):
  in0 = Input(dim(types.i32, 16))
  in1 = Input(types.i32)
  t_c = dim(types.i32, 32)
  c = Output(t_c)

  @generator
  def build(ports):
    # a 32x32xi1 ndarray.
    # A dtype of i32 is fairly expensive wrt. the size of the output IR, but
    # but allows for assigning indiviudal bits.
    m = NDArray((32, 32), dtype=types.i1, name='m1')

    # Assign individual bits to the first 32 bits.
    for i in range(32):
      m[0, i] = hw.ConstantOp(types.i1, 1)

    # Fill the next 15 values with an i32. The ndarray knows how to convert
    # from i32 to <32xi1> to comply with the ndarray dtype.
    for i in range(1, 16):
      m[i] = ports.in1

    # Fill the upportmost 16 rows with the input array of in0 : 16xi32
    m[16:32] = ports.in0

    # We don't provide a method of reshaping the ndarray wrt. its dtype.
    # that is: 32x32xi1 => 32xi32
    # This has massive overhead in the generated IR, and can be easily
    # achieved by a bitcast.
    ports.c = hw.BitcastOp(M5.t_c, m.to_circt())


# -----

# ndarray from value


@unittestmodule()
class M1(Module):
  in1 = Input(dim(types.i32, 10, 10))
  out = Output(dim(types.i32, 10, 10))

  @generator
  def build(ports):
    m = NDArray(from_value=ports.in1, name='m1')
    ports.out = m.to_circt()


# -----

# No barrier wire


@unittestmodule()
class M1(Module):
  in1 = Input(dim(types.i32, 10))
  out = Output(dim(types.i32, 10))

  @generator
  def build(ports):
    m = NDArray(from_value=ports.in1, name='m1')
    ports.out = m.to_circt(create_wire=False)


# -----

# Concatenation using both explicit NDArrays as well as ListValues.


@unittestmodule()
class M1(Module):
  in1 = Input(dim(types.i32, 10))
  in2 = Input(dim(types.i32, 10))
  in3 = Input(dim(types.i32, 10))
  out = Output(dim(types.i32, 30))

  @generator
  def build(ports):
    # Explicit ndarray.
    m = NDArray(from_value=ports.in1, name='m1')
    # Here we could do a sequence of transformations on 'm'...
    # Finally, concatenate [in2, m, in3]
    ports.out = ports.in2.concatenate((m, ports.in3))


# -----

# Rolling


@unittestmodule()
class M1(Module):
  in1 = Input(dim(types.i32, 10))
  out = Output(dim(types.i32, 10))

  @generator
  def build(ports):
    ports.out = ports.in1.roll(3)


# -----

# CHECK-LABEL:   msft.module @M1 {} () -> (out: !hw.array<3xarray<3xi32>>) attributes {fileName = "M1.sv"} {
# CHECK:           %[[VAL_0:.*]] = hw.constant 0 : i32
# CHECK:           %[[VAL_1:.*]] = hw.constant 1 : i32
# CHECK:           %[[VAL_2:.*]] = hw.constant 2 : i32
# CHECK:           %[[VAL_3:.*]] = hw.constant 3 : i32
# CHECK:           %[[VAL_4:.*]] = hw.constant 4 : i32
# CHECK:           %[[VAL_5:.*]] = hw.constant 5 : i32
# CHECK:           %[[VAL_6:.*]] = hw.constant 6 : i32
# CHECK:           %[[VAL_7:.*]] = hw.constant 7 : i32
# CHECK:           %[[VAL_8:.*]] = hw.constant 8 : i32
# CHECK:           %[[VAL_9:.*]] = hw.array_create %[[VAL_8]], %[[VAL_7]], %[[VAL_6]] : i32
# CHECK:           %[[VAL_10:.*]] = hw.array_create %[[VAL_5]], %[[VAL_4]], %[[VAL_3]] : i32
# CHECK:           %[[VAL_11:.*]] = hw.array_create %[[VAL_2]], %[[VAL_1]], %[[VAL_0]] : i32
# CHECK:           %[[VAL_12:.*]] = hw.array_create %[[VAL_9]], %[[VAL_10]], %[[VAL_11]] : !hw.array<3xi32>
# CHECK:           %[[VAL_13:.*]] = sv.wire  : !hw.inout<array<3xarray<3xi32>>>
# CHECK:           sv.assign %[[VAL_13]], %[[VAL_12]] : !hw.array<3xarray<3xi32>>
# CHECK:           %[[VAL_14:.*]] = sv.read_inout %[[VAL_13]] : !hw.inout<array<3xarray<3xi32>>>
# CHECK:           msft.output %[[VAL_14]] : !hw.array<3xarray<3xi32>>
# CHECK:         }


@unittestmodule()
class M1(Module):
  out = Output(dim(types.i32, 3, 3))

  @generator
  def build(ports):
    m = NDArray((3, 3), dtype=types.i32)
    for i in range(3):
      for j in range(3):
        m[i][j] = types.i32(i * 3 + j)
    ports.out = m.to_circt()
