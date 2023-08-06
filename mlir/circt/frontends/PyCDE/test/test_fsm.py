# RUN: %PYTHON% py-split-input-file.py %s | FileCheck %s

from pycde import System, Input, Output, generator, Module
from pycde.common import Clock, Reset
from pycde.dialects import comb
from pycde import fsm
from pycde.types import types
from pycde.testing import unittestmodule

# FSM state transitions example
# CHECK-LABEL: msft.module @FSMUser {} (%a: i1, %b: i1, %c: i1, %clk: i1, %rst: i1) -> (is_a: i1, is_b: i1, is_c: i1)
# CHECK-NEXT:    %0:4 = fsm.hw_instance "F0" @F0(%a, %b, %c), clock %clk, reset %rst : (i1, i1, i1) -> (i1, i1, i1, i1)
# CHECK-NEXT:    msft.output %0#1, %0#2, %0#3 : i1, i1, i1
# CHECK-NEXT:  }
# CHECK-LABEL: fsm.machine @F0(%arg0: i1, %arg1: i1, %arg2: i1) -> (i1, i1, i1, i1) attributes {clock_name = "clk", in_names = ["a", "b", "c"], initialState = "idle", out_names = ["is_idle", "is_A", "is_B", "is_C"], reset_name = "rst"} {
# CHECK-NEXT:    fsm.state @idle output {
# CHECK-NEXT:      %true = hw.constant true
# CHECK-NEXT:      %false = hw.constant false
# CHECK-NEXT:      %false_0 = hw.constant false
# CHECK-NEXT:      %false_1 = hw.constant false
# CHECK-NEXT:      fsm.output %true, %false, %false_0, %false_1 : i1, i1, i1, i1
# CHECK-NEXT:    } transitions {
# CHECK-NEXT:      fsm.transition @A
# CHECK-NEXT:    }
# CHECK-NEXT:    fsm.state @A output {
# CHECK-NEXT:      %false = hw.constant false
# CHECK-NEXT:      %true = hw.constant true
# CHECK-NEXT:      %false_0 = hw.constant false
# CHECK-NEXT:      %false_1 = hw.constant false
# CHECK-NEXT:      fsm.output %false, %true, %false_0, %false_1 : i1, i1, i1, i1
# CHECK-NEXT:    } transitions {
# CHECK-NEXT:      fsm.transition @B guard {
# CHECK-NEXT:        fsm.return %arg0
# CHECK-NEXT:      }
# CHECK-NEXT:    }
# CHECK-NEXT:    fsm.state @B output {
# CHECK-NEXT:      %false = hw.constant false
# CHECK-NEXT:      %false_0 = hw.constant false
# CHECK-NEXT:      %true = hw.constant true
# CHECK-NEXT:      %false_1 = hw.constant false
# CHECK-NEXT:      fsm.output %false, %false_0, %true, %false_1 : i1, i1, i1, i1
# CHECK-NEXT:    } transitions {
# CHECK-NEXT:      fsm.transition @C guard {
# CHECK-NEXT:        %0 = comb.and bin %arg0, %arg1 : i1
# CHECK-NEXT:        %true = hw.constant true
# CHECK-NEXT:        %1 = comb.xor bin %0, %true : i1
# CHECK-NEXT:        %2 = comb.and bin %arg1, %arg2 : i1
# CHECK-NEXT:        %true_0 = hw.constant true
# CHECK-NEXT:        %3 = comb.xor bin %2, %true_0 : i1
# CHECK-NEXT:        %4 = comb.and bin %arg0, %arg2 : i1
# CHECK-NEXT:        %true_1 = hw.constant true
# CHECK-NEXT:        %5 = comb.xor bin %4, %true_1 : i1
# CHECK-NEXT:        %6 = comb.and bin %1, %3, %5 : i1
# CHECK-NEXT:        %true_2 = hw.constant true
# CHECK-NEXT:        %7 = comb.xor bin %6, %true_2 : i1
# CHECK-NEXT:        fsm.return %7
# CHECK-NEXT:      }
# CHECK-NEXT:    }
# CHECK-NEXT:    fsm.state @C output {
# CHECK-NEXT:      %false = hw.constant false
# CHECK-NEXT:      %false_0 = hw.constant false
# CHECK-NEXT:      %false_1 = hw.constant false
# CHECK-NEXT:      %true = hw.constant true
# CHECK-NEXT:      fsm.output %false, %false_0, %false_1, %true : i1, i1, i1, i1
# CHECK-NEXT:    } transitions {
# CHECK-NEXT:      fsm.transition @idle guard {
# CHECK-NEXT:        fsm.return %arg2
# CHECK-NEXT:      }
# CHECK-NEXT:      fsm.transition @A guard {
# CHECK-NEXT:        %true = hw.constant true
# CHECK-NEXT:        %0 = comb.xor bin %arg1, %true : i1
# CHECK-NEXT:        fsm.return %0
# CHECK-NEXT:      }
# CHECK-NEXT:    }
# CHECK-NEXT:  }


class F0(fsm.Machine):
  a = Input(types.i1)
  b = Input(types.i1)
  c = Input(types.i1)

  def maj3(ports):

    def nand(*args):
      return comb.XorOp(comb.AndOp(*args), types.i1(1))

    c1 = nand(ports.a, ports.b)
    c2 = nand(ports.b, ports.c)
    c3 = nand(ports.a, ports.c)
    return nand(c1, c2, c3)

  idle = fsm.State(initial=True)
  (A, B, C) = fsm.States(3)

  idle.set_transitions((A,))
  A.set_transitions((B, lambda ports: ports.a))
  B.set_transitions((C, maj3))
  C.set_transitions((idle, lambda ports: ports.c),
                    (A, lambda ports: comb.XorOp(ports.b, types.i1(1))))


@unittestmodule()
class FSMUser(Module):
  a = Input(types.i1)
  b = Input(types.i1)
  c = Input(types.i1)
  clk = Input(types.i1)
  rst = Input(types.i1)
  is_a = Output(types.i1)
  is_b = Output(types.i1)
  is_c = Output(types.i1)

  @generator
  def construct(ports):
    fsm = F0(a=ports.a, b=ports.b, c=ports.c, clk=ports.clk, rst=ports.rst)
    ports.is_a = fsm.is_A
    ports.is_b = fsm.is_B
    ports.is_c = fsm.is_C


system = System([FSMUser])
system.generate()
system.print()

# ------

# Test alternative clock / reset names.


# CHECK-LABEL:  fsm.machine @FsmClockTest(%arg0: i1) -> (i1, i1) attributes {clock_name = "clock", in_names = ["a"], initialState = "A", out_names = ["is_A", "is_B"], reset_name = "reset"}
@unittestmodule()
class FsmClockTest(fsm.Machine):
  clock = Clock()
  reset = Reset()

  a = Input(types.i1)
  A = fsm.State(initial=True)
  B = fsm.State()

  A.set_transitions((B, lambda ports: ports.a))
  B.set_transitions((A, lambda ports: ports.a))
