# RUN: %PYTHON% %s | FileCheck %s

from pycde import generator, types, dim, Module
from pycde.common import Clock, Input, Output
from pycde.constructs import ControlReg, NamedWire, Reg, Wire, SystolicArray
from pycde.dialects import comb
from pycde.testing import unittestmodule

# CHECK-LABEL: msft.module @WireAndRegTest {} (%In: i8, %InCE: i1, %clk: i1, %rst: i1) -> (Out: i8, OutReg: i8, OutRegRst: i8, OutRegCE: i8)
# CHECK:         [[r0:%.+]] = comb.extract %In from 0 {sv.namehint = "In_0upto7"} : (i8) -> i7
# CHECK:         [[r1:%.+]] = comb.extract %In from 7 {sv.namehint = "In_7upto8"} : (i8) -> i1
# CHECK:         [[r2:%.+]] = comb.concat [[r1]], [[r0]] {sv.namehint = "w1"} : i1, i7
# CHECK:         %in = sv.wire sym @in : !hw.inout<i8>
# CHECK:         {{%.+}} = sv.read_inout %in {sv.namehint = "in"} : !hw.inout<i8>
# CHECK:         sv.assign %in, %In : i8
# CHECK:         [[r1:%.+]] = seq.compreg %In, %clk : i8
# CHECK:         %c0_i8{{.*}} = hw.constant 0 : i8
# CHECK:         [[r5:%.+]] = seq.compreg %In, %clk, %rst, %c0_i8{{.*}}  : i8
# CHECK:         [[r6:%.+]] = seq.compreg.ce %In, %clk, %InCE : i8
# CHECK:         msft.output [[r2]], [[r1]], [[r5]], [[r6]] : i8, i8, i8, i8


@unittestmodule()
class WireAndRegTest(Module):
  In = Input(types.i8)
  InCE = Input(types.i1)
  clk = Clock()
  rst = Input(types.i1)
  Out = Output(types.i8)
  OutReg = Output(types.i8)
  OutRegRst = Output(types.i8)
  OutRegCE = Output(types.i8)

  @generator
  def create(ports):
    w1 = Wire(types.i8, "w1")
    ports.Out = w1
    w1[0:7] = ports.In[0:7]
    w1[7] = ports.In[7]

    NamedWire(ports.In, "in")

    r1 = Reg(types.i8)
    ports.OutReg = r1
    r1.assign(ports.In)

    r_rst = Reg(types.i8, rst=ports.rst, rst_value=0)
    ports.OutRegRst = r_rst
    r_rst.assign(ports.In)

    r_ce = Reg(types.i8, ce=ports.InCE)
    ports.OutRegCE = r_ce
    r_ce.assign(ports.In)


# CHECK-LABEL: %{{.+}} = msft.systolic.array [%{{.+}} : 3 x i8] [%{{.+}} : 2 x i8] pe (%arg0, %arg1) -> (i8) {
# CHECK:         [[SUM:%.+]] = comb.add bin %arg0, %arg1 {sv.namehint = "sum"} : i8
# CHECK:         [[SUMR:%.+]] = seq.compreg sym @sum__reg1 [[SUM]], %clk : i8
# CHECK:         msft.pe.output [[SUMR]] : i8


# CHECK-LABEL: hw.module @SystolicArrayTest<__INST_HIER: none = "INSTANTIATE_WITH_INSTANCE_PATH">(%clk: i1, %col_data: !hw.array<2xi8>, %row_data: !hw.array<3xi8>) -> (out: !hw.array<3xarray<2xi8>>)
# CHECK:         %sum__reg1_0_0 = sv.reg sym @sum__reg1  : !hw.inout<i8>
# CHECK:         sv.read_inout %sum__reg1_0_0 : !hw.inout<i8>
@unittestmodule(print=True, run_passes=True, print_after_passes=True)
class SystolicArrayTest(Module):
  clk = Input(types.i1)
  col_data = Input(dim(8, 2))
  row_data = Input(dim(8, 3))
  out = Output(dim(8, 2, 3))

  @generator
  def build(ports):
    # If we just feed constants, CIRCT pre-computes the outputs in the
    # generated Verilog! Keep these for demo purposes.
    # row_data = dim(8, 3)([1, 2, 3])
    # col_data = dim(8, 2)([4, 5])

    def pe(r, c):
      sum = comb.AddOp(r, c)
      sum.name = "sum"
      return sum.reg(ports.clk)

    pe_outputs = SystolicArray(ports.row_data, ports.col_data, pe)

    ports.out = pe_outputs


# CHECK-LABEL:  msft.module @ControlReg_num_asserts2_num_resets1
# CHECK:          [[r0:%.+]] = hw.array_get %asserts[%false]
# CHECK:          [[r1:%.+]] = hw.array_get %asserts[%true]
# CHECK:          [[r2:%.+]] = comb.or bin [[r0]], [[r1]]
# CHECK:          [[r3:%.+]] = hw.array_get %resets[%c0_i0]
# CHECK:          [[r4:%.+]] = comb.or bin [[r3]]
# CHECK:          %state = seq.compreg [[r6]], %clk, %rst, %false{{.*}}
# CHECK:          [[r5:%.+]] = comb.mux bin [[r4]], %false{{.*}}, %state
# CHECK:          [[r6:%.+]] = comb.mux bin [[r2]], %true{{.*}}, [[r5]]
# CHECK:          msft.output %state
@unittestmodule()
class ControlRegTest(Module):
  clk = Clock()
  rst = Input(types.i1)
  a1 = Input(types.i1)
  a2 = Input(types.i1)
  r1 = Input(types.i1)

  @generator
  def build(ports):
    ControlReg(ports.clk,
               ports.rst,
               asserts=[ports.a1, ports.a2],
               resets=[ports.r1])
