# RUN: %PYTHON% py-split-input-file.py %s | FileCheck %s

from pycde import Input, Output, generator, Module
from pycde.testing import unittestmodule
from pycde.types import types, UInt


# CHECK: msft.module @InfixArith {} (%in0: si16, %in1: ui16)
# CHECK-NEXT:   %0 = hwarith.add %in0, %in1 {{({sv.namehint = ".*"} )?}}: (si16, ui16) -> si18
# CHECK-NEXT:   %1 = hwarith.sub %in0, %in1 {{({sv.namehint = ".*"} )?}}: (si16, ui16) -> si18
# CHECK-NEXT:   %2 = hwarith.mul %in0, %in1 {{({sv.namehint = ".*"} )?}}: (si16, ui16) -> si32
# CHECK-NEXT:   %3 = hwarith.div %in0, %in1 {{({sv.namehint = ".*"} )?}}: (si16, ui16) -> si16
# CHECK-NEXT:   %c-1_i16 = hw.constant -1 {{({sv.namehint = ".*"} )?}}: i16
# CHECK-NEXT:   %4 = hwarith.cast %c-1_i16 {{({sv.namehint = ".*"} )?}}: (i16) -> si16
# CHECK-NEXT:   %5 = hwarith.mul %in0, %4 {{({sv.namehint = ".*"} )?}}: (si16, si16) -> si32
# CHECK-NEXT:   msft.output
@unittestmodule(run_passes=True)
class InfixArith(Module):
  in0 = Input(types.si16)
  in1 = Input(types.ui16)

  @generator
  def construct(ports):
    add = ports.in0 + ports.in1
    sub = ports.in0 - ports.in1
    mul = ports.in0 * ports.in1
    div = ports.in0 / ports.in1
    neg = -ports.in0


# -----


# CHECK: msft.module @InfixLogic {} (%in0: i16, %in1: i16)
# CHECK-NEXT:  comb.and bin %in0, %in1 {{({sv.namehint = ".*"} )?}}: i16
# CHECK-NEXT:  comb.or bin %in0, %in1 {{({sv.namehint = ".*"} )?}}: i16
# CHECK-NEXT:  comb.xor bin %in0, %in1 {{({sv.namehint = ".*"} )?}}: i16
# CHECK-NEXT:  %c-1_i16 = hw.constant -1 {{({sv.namehint = ".*"} )?}}: i16
# CHECK-NEXT:  comb.xor bin %in0, %c-1_i16 {{({sv.namehint = ".*"} )?}}: i16
# CHECK-NEXT:  msft.output
@unittestmodule(run_passes=True)
class InfixLogic(Module):
  in0 = Input(types.i16)
  in1 = Input(types.i16)

  @generator
  def construct(ports):
    and_ = ports.in0 & ports.in1
    or_ = ports.in0 | ports.in1
    xor = ports.in0 ^ ports.in1
    inv = ~ports.in0


# -----


# CHECK: msft.module @SignlessInfixComparison {} (%in0: i16, %in1: i16)
# CHECK-NEXT:    %0 = comb.icmp bin eq %in0, %in1 {{({sv.namehint = ".*"} )?}}: i16
# CHECK-NEXT:    %1 = comb.icmp bin ne %in0, %in1 {{({sv.namehint = ".*"} )?}}: i16
# CHECK-NEXT:    msft.output
@unittestmodule(run_passes=True)
class SignlessInfixComparison(Module):
  in0 = Input(types.i16)
  in1 = Input(types.i16)

  @generator
  def construct(ports):
    eq = ports.in0 == ports.in1
    neq = ports.in0 != ports.in1


# -----


# CHECK: msft.module @InfixComparison {} (%in0: ui16, %in1: ui16)
# CHECK:  %0 = hwarith.icmp eq %in0, %in1 {sv.namehint = "in0_eq_in1"} : ui16, ui16
# CHECK:  %1 = hwarith.icmp ne %in0, %in1 {sv.namehint = "in0_neq_in1"} : ui16, ui16
# CHECK:  %2 = hwarith.icmp lt %in0, %in1 {sv.namehint = "in0_lt_in1"} : ui16, ui16
# CHECK:  %3 = hwarith.icmp gt %in0, %in1 {sv.namehint = "in0_gt_in1"} : ui16, ui16
# CHECK:  %4 = hwarith.icmp le %in0, %in1 {sv.namehint = "in0_le_in1"} : ui16, ui16
# CHECK:  %5 = hwarith.icmp ge %in0, %in1 {sv.namehint = "in0_ge_in1"} : ui16, ui16
# CHECK-NEXT:    msft.output
@unittestmodule(run_passes=False)
class InfixComparison(Module):
  in0 = Input(types.ui16)
  in1 = Input(types.ui16)

  @generator
  def construct(ports):
    eq = ports.in0 == ports.in1
    neq = ports.in0 != ports.in1
    lt = ports.in0 < ports.in1
    gt = ports.in0 > ports.in1
    le = ports.in0 <= ports.in1
    ge = ports.in0 >= ports.in1


# -----


# CHECK:  msft.module @Multiple {} (%in0: si16, %in1: si16) -> (out0: i16)
# CHECK-NEXT:    %0 = hwarith.add %in0, %in1 {{({sv.namehint = ".*"} )?}}: (si16, si16) -> si17
# CHECK-NEXT:    %1 = hwarith.add %0, %in0 {{({sv.namehint = ".*"} )?}}: (si17, si16) -> si18
# CHECK-NEXT:    %2 = hwarith.add %1, %in1 {{({sv.namehint = ".*"} )?}}: (si18, si16) -> si19
# CHECK-NEXT:    %3 = hwarith.cast %2 {{({sv.namehint = ".*"} )?}}: (si19) -> i16
# CHECK-NEXT:    msft.output %3 {{({sv.namehint = ".*"} )?}}: i16
@unittestmodule(run_passes=True)
class Multiple(Module):
  in0 = Input(types.si16)
  in1 = Input(types.si16)
  out0 = Output(types.i16)

  @generator
  def construct(ports):
    ports.out0 = (ports.in0 + ports.in1 + ports.in0 + ports.in1).as_bits(16)


# -----


# CHECK:  msft.module @Casting {} (%in0: i16)
# CHECK-NEXT:    %0 = hwarith.cast %in0 {{({sv.namehint = ".*"} )?}}: (i16) -> si16
# CHECK-NEXT:    %1 = hwarith.cast %in0 {{({sv.namehint = ".*"} )?}}: (i16) -> ui16
# CHECK-NEXT:    %2 = hwarith.cast %0 {{({sv.namehint = ".*"} )?}}: (si16) -> i16
# CHECK-NEXT:    %3 = hwarith.cast %in0 {{({sv.namehint = ".*"} )?}}: (i16) -> si8
# CHECK-NEXT:    %4 = hwarith.cast %in0 {{({sv.namehint = ".*"} )?}}: (i16) -> ui8
# CHECK-NEXT:    %5 = hwarith.cast %0 {{({sv.namehint = ".*"} )?}}: (si16) -> i8
# CHECK-NEXT:    %6 = hwarith.cast %0 {{({sv.namehint = ".*"} )?}}: (si16) -> si24
# CHECK-NEXT:    msft.output
@unittestmodule(run_passes=True)
class Casting(Module):
  in0 = Input(types.i16)

  @generator
  def construct(ports):
    in0s = ports.in0.as_sint()
    in0u = ports.in0.as_uint()
    in0s_i = in0s.as_bits()
    in0s8 = ports.in0.as_sint(8)
    in0u8 = ports.in0.as_uint(8)
    in0s_i8 = in0s.as_bits(8)
    in0s_s24 = in0s.as_sint(24)


# -----


# CHECK: hw.module @Lowering<__INST_HIER: none = "INSTANTIATE_WITH_INSTANCE_PATH">(%in0: i16, %in1: i16) -> (out0: i16)
# CHECK-NEXT:    %0 = comb.add %in0, %in1 {{({sv.namehint = ".*"} )?}}: i16
# CHECK-NEXT:    hw.output %0 {{({sv.namehint = ".*"} )?}}: i16
@unittestmodule(generate=True, run_passes=True, print_after_passes=True)
class Lowering(Module):
  in0 = Input(types.i16)
  in1 = Input(types.i16)
  out0 = Output(types.i16)

  @generator
  def construct(ports):
    ports.out0 = (ports.in0.as_sint() + ports.in1.as_sint()).as_bits(16)


# -----


# CHECK-LABEL:  msft.module @Constants {} (%uin: ui16, %sin: si16) attributes {fileName = "Constants.sv"} {
# CHECK-NEXT:     [[R0:%.+]] = hwarith.constant 1 : ui1
# CHECK-NEXT:     [[R1:%.+]] = hwarith.add %uin, [[R0]] : (ui16, ui1) -> ui17
# CHECK-NEXT:     [[R2:%.+]] = hwarith.constant -1 : si2
# CHECK-NEXT:     [[R3:%.+]] = hwarith.add %uin, [[R2]] : (ui16, si2) -> si18
# CHECK-NEXT:     [[R4:%.+]] = hwarith.constant 1 : ui1
# CHECK-NEXT:     [[R5:%.+]] = hwarith.icmp eq %uin, [[R4]] : ui16, ui1
# CHECK-NEXT:     [[R6:%.+]] = hwarith.constant 1 : ui1
# CHECK-NEXT:     [[R7:%.+]] = hwarith.add %sin, [[R6]] : (si16, ui1) -> si17
# CHECK-NEXT:     [[R8:%.+]] = hwarith.constant -1 : si2
# CHECK-NEXT:     [[R9:%.+]] = hwarith.add %sin, [[R8]] : (si16, si2) -> si17
# CHECK-NEXT:     [[R10:%.+]] = hwarith.constant 1 : ui1
# CHECK-NEXT:     [[R11:%.+]] = hwarith.icmp eq %sin, [[R10]] : si16, ui1
@unittestmodule()
class Constants(Module):
  uin = Input(types.ui16)
  sin = Input(types.si16)

  @generator
  def construct(ports):
    ports.uin + 1
    ports.uin + -1
    ports.uin == 1

    ports.sin + 1
    ports.sin + -1
    ports.sin == 1


# -----


@unittestmodule(generate=True,
                run_passes=True,
                print_after_passes=True,
                debug=True)
class AddInts(Module):
  a = Input(UInt(32))
  b = Input(UInt(32))
  c = Output(UInt(33))

  @generator
  def construct(self):
    self.c = self.a + self.b
