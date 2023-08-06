# RUN: %PYTHON% py-split-input-file.py %s | FileCheck %s

from pycde import generator, dim, Clock, Input, Output, Module, types
from pycde.signals import Signal
from pycde.constructs import Mux
from pycde.testing import unittestmodule
from pycde.types import Bits

# CHECK-LABEL: msft.module @ComplexMux {} (%Clk: i1, %In: !hw.array<5xarray<4xi3>>, %Sel: i1) -> (Out: !hw.array<4xi3>, OutArr: !hw.array<2xarray<4xi3>>, OutInt: i1, OutSlice: !hw.array<3xarray<4xi3>>)
# CHECK:         %c3_i3 = hw.constant 3 : i3
# CHECK:         %0 = hw.array_get %In[%c3_i3] {sv.namehint = "In__3"} : !hw.array<5xarray<4xi3>>
# CHECK:         %In__3__reg1 = seq.compreg sym @In__3__reg1 %0, %Clk : !hw.array<4xi3>
# CHECK:         %In__3__reg2 = seq.compreg sym @In__3__reg2 %In__3__reg1, %Clk : !hw.array<4xi3>
# CHECK:         %In__3__reg3 = seq.compreg sym @In__3__reg3 %In__3__reg2, %Clk : !hw.array<4xi3>
# CHECK:         %c1_i3 = hw.constant 1 : i3
# CHECK:         [[R1:%.+]] = hw.array_get %In[%c1_i3] {sv.namehint = "In__1"} : !hw.array<5xarray<4xi3>>
# CHECK:         [[R3:%.+]] = comb.mux bin %Sel, [[R1]], %In__3__reg3 {sv.namehint = "mux_Sel_In__3__reg3_In__1"} : !hw.array<4xi3>
# CHECK:         %c0_i3 = hw.constant 0 : i3
# CHECK:         [[R4:%.+]] = hw.array_get %In[%c0_i3] {sv.namehint = "In__0"} : !hw.array<5xarray<4xi3>>
# CHECK:         %c1_i3_0 = hw.constant 1 : i3
# CHECK:         [[R5:%.+]] = hw.array_get %In[%c1_i3_0] {sv.namehint = "In__1"} : !hw.array<5xarray<4xi3>>
# CHECK:         [[R6:%.+]] = hw.array_create [[R5]], [[R4]] : !hw.array<4xi3>
# CHECK:         [[R7:%.+]] = hw.array_slice %In[%c0_i3_1] {sv.namehint = "In_0upto3"} : (!hw.array<5xarray<4xi3>>) -> !hw.array<3xarray<4xi3>>
# CHECK:         %c0_i3_2 = hw.constant 0 : i3
# CHECK:         [[R8:%.+]] = hw.array_get %In[%c0_i3_2] {sv.namehint = "In__0"} : !hw.array<5xarray<4xi3>>, i3
# CHECK:         [[R9:%.+]] = hw.array_get [[R8]][%c0_i2] {sv.namehint = "In__0__0"} : !hw.array<4xi3>
# CHECK:         %c0_i2_3 = hw.constant 0 : i2
# CHECK:         [[R10:%.+]] = comb.concat %c0_i2_3, %Sel {sv.namehint = "Sel_padto_3"} : i2, i1
# CHECK:         [[R11:%.+]] = comb.shru bin [[R9]], [[R10]] : i3
# CHECK:         [[R12:%.+]] = comb.extract [[R11]] from 0 : (i3) -> i1
# CHECK:         msft.output [[R3]], [[R6]], [[R12]], [[R7]] : !hw.array<4xi3>, !hw.array<2xarray<4xi3>>, i1, !hw.array<3xarray<4xi3>>


@unittestmodule()
class ComplexMux(Module):

  Clk = Clock()
  In = Input(dim(3, 4, 5))
  Sel = Input(dim(1))
  Out = Output(dim(3, 4))
  OutArr = Output(dim(3, 4, 2))
  OutInt = Output(types.i1)
  OutSlice = Output(dim(3, 4, 3))

  @generator
  def create(ports):
    ports.Out = Mux(ports.Sel, ports.In[3].reg().reg(cycles=2), ports.In[1])

    ports.OutArr = Signal.create([ports.In[0], ports.In[1]])
    ports.OutSlice = ports.In[0:3]

    ports.OutInt = ports.In[0][0][ports.Sel]


# -----

# CHECK-LABEL:  msft.module @Slicing {} (%In: !hw.array<5xarray<4xi8>>, %Sel8: i8, %Sel2: i2) -> (OutIntSlice: i2, OutArrSlice8: !hw.array<2xarray<4xi8>>, OutArrSlice2: !hw.array<2xarray<4xi8>>)
# CHECK:          [[R0:%.+]] = hw.array_get %In[%c0_i3] {sv.namehint = "In__0"} : !hw.array<5xarray<4xi8>>
# CHECK:          [[R1:%.+]] = hw.array_get %0[%c0_i2] {sv.namehint = "In__0__0"} : !hw.array<4xi8>
# CHECK:          [[R2:%.+]] = comb.concat %c0_i6, %Sel2 {sv.namehint = "Sel2_padto_8"} : i6, i2
# CHECK:          [[R3:%.+]] = comb.shru bin [[R1]], [[R2]] : i8
# CHECK:          [[R4:%.+]] = comb.extract [[R3]] from 0 : (i8) -> i2
# CHECK:          [[R5:%.+]] = comb.concat %false, %Sel2 {sv.namehint = "Sel2_padto_3"} : i1, i2
# CHECK:          [[R6:%.+]] = hw.array_slice %In[[[R5]]] : (!hw.array<5xarray<4xi8>>) -> !hw.array<2xarray<4xi8>>
# CHECK:          [[R7:%.+]] = comb.extract %Sel8 from 0 : (i8) -> i3
# CHECK:          [[R8:%.+]] = hw.array_slice %In[[[R7]]] : (!hw.array<5xarray<4xi8>>) -> !hw.array<2xarray<4xi8>>
# CHECK:          msft.output %4, %8, %6 : i2, !hw.array<2xarray<4xi8>>, !hw.array<2xarray<4xi8>>


@unittestmodule()
class Slicing(Module):
  In = Input(dim(8, 4, 5))
  Sel8 = Input(types.i8)
  Sel2 = Input(types.i2)

  OutIntSlice = Output(types.i2)
  OutArrSlice8 = Output(dim(8, 4, 2))
  OutArrSlice2 = Output(dim(8, 4, 2))

  @generator
  def create(ports):
    i = ports.In[0][0]
    ports.OutIntSlice = i.slice(ports.Sel2, 2)
    ports.OutArrSlice2 = ports.In.slice(ports.Sel2, 2)
    ports.OutArrSlice8 = ports.In.slice(ports.Sel8, 2)


# CHECK-LABEL:  msft.module @SimpleMux2 {} (%op: i1, %a: i32, %b: i32) -> (out: i32)
# CHECK-NEXT:     [[r0:%.+]] = comb.mux bin %op, %b, %a
# CHECK-NEXT:     msft.output %0 : i32
@unittestmodule()
class SimpleMux2(Module):
  op = Input(Bits(1))
  a = Input(Bits(32))
  b = Input(Bits(32))
  out = Output(Bits(32))

  @generator
  def construct(self):
    self.out = Mux(self.op, self.a, self.b)


# CHECK-LABEL:  msft.module @SimpleMux4 {} (%op: i2, %a: i32, %b: i32, %c: i32, %d: i32) -> (out: i32)
# CHECK-NEXT:     [[r0:%.+]] = hw.array_create %d, %c, %b, %a
# CHECK-NEXT:     [[r1:%.+]] = hw.array_get [[r0]][%op]
# CHECK-NEXT:     msft.output [[r1]] : i32
@unittestmodule()
class SimpleMux4(Module):
  op = Input(Bits(2))
  a = Input(Bits(32))
  b = Input(Bits(32))
  c = Input(Bits(32))
  d = Input(Bits(32))
  out = Output(Bits(32))

  @generator
  def construct(self):
    self.out = Mux(self.op, self.a, self.b, self.c, self.d)
