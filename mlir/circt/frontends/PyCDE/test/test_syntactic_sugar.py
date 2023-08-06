# RUN: %PYTHON% %s | FileCheck %s

from pycde import (Output, Input, generator, types, dim, Module)
from pycde.testing import unittestmodule

# CHECK-LABEL:  msft.module @Top {} () attributes {fileName = "Top.sv"} {
# CHECK:    %c7_i12 = hw.constant 7 : i12
# CHECK:    hw.struct_create (%c7_i12) : !hw.struct<foo: i12>
# CHECK:    %c42_i8 = hw.constant 42 : i8
# CHECK:    %c45_i8 = hw.constant 45 : i8
# CHECK:    hw.array_create %c45_i8, %c42_i8 : i8
# CHECK:    %c5_i8 = hw.constant 5 : i8
# CHECK:    %c7_i12_0 = hw.constant 7 : i12
# CHECK:    hw.struct_create (%c7_i12_0) : !hw.typealias<@pycde::@bar, !hw.struct<foo: i12>>
# CHECK:    %Taps.taps = msft.instance @Taps @Taps()  : () -> !hw.array<3xi8>
# CHECK:    msft.output
# CHECK-LABEL:  msft.module @Taps {} () -> (taps: !hw.array<3xi8>) attributes {fileName = "Taps.sv"} {
# CHECK:    %c-53_i8 = hw.constant -53 : i8
# CHECK:    %c100_i8 = hw.constant 100 : i8
# CHECK:    %c23_i8 = hw.constant 23 : i8
# CHECK:    [[R0:%.+]] = hw.array_create %c23_i8, %c100_i8, %c-53_i8 : i8
# CHECK:    msft.output [[R0]] : !hw.array<3xi8>


class Taps(Module):
  taps = Output(dim(8, 3))

  @generator
  def build(ports):
    ports.taps = [203, 100, 23]


BarType = types.struct({"foo": types.i12}, "bar")


@unittestmodule()
class Top(Module):

  @generator
  def build(_):
    types.struct({"foo": types.i12})({"foo": 7})
    dim(types.i8, 2)([42, 45])
    types.i8(5)

    BarType({"foo": 7})

    Taps()


# -----

# CHECK:  msft.module @ComplexPorts {} (%clk: i1, %data_in: !hw.array<3xi32>, %sel: i2, %struct_data_in: !hw.struct<foo: i36>) -> (a: i32, b: i32, c: i32)
# CHECK:    %c0_i2 = hw.constant 0 : i2
# CHECK:    [[REG0:%.+]] = hw.array_get %data_in[%c0_i2] {sv.namehint = "data_in__0"} : !hw.array<3xi32>
# CHECK:    [[REGR1:%data_in__0__reg1]] = seq.compreg sym @data_in__0__reg1 [[REG0]], %clk : i32
# CHECK:    [[REGR2:%data_in__0__reg2]] = seq.compreg sym @data_in__0__reg2 [[REGR1]], %clk : i32
# CHECK:    [[REG1:%.+]] = hw.array_get %data_in[%sel] : !hw.array<3xi32>
# CHECK:    [[REG2:%.+]] = hw.struct_extract %struct_data_in["foo"] {sv.namehint = "struct_data_in__foo"} : !hw.struct<foo: i36>
# CHECK:    [[REG3:%.+]] = comb.extract [[REG2]] from 0 {sv.namehint = "struct_data_in__foo_0upto32"} : (i36) -> i32
# CHECK:    msft.output [[REGR2]], [[REG1]], [[REG3]] : i32, i32, i32


@unittestmodule()
class ComplexPorts(Module):
  clk = Input(types.i1)
  data_in = Input(dim(32, 3))
  sel = Input(types.i2)
  struct_data_in = Input(types.struct({"foo": types.i36}))

  a = Output(types.i32)
  b = Output(types.i32)
  c = Output(types.i32)

  @generator
  def build(self):
    assert len(self.data_in) == 3
    self.a = self.data_in[0].reg(self.clk).reg(self.clk)
    self.b = self.data_in[self.sel]
    self.c = self.struct_data_in.foo[:-4]
