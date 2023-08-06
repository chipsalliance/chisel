# RUN: %PYTHON% py-split-input-file.py %s | FileCheck %s

from pycde import System, Input, Output, generator, Module
from pycde.types import dim

# CHECK-LABEL: msft.module @MyMod {} (%in_port: i8) -> (out0: i5, out1: i5) attributes {fileName = "MyMod.sv"} {
# CHECK:         %0 = comb.extract %in_port from 3 {sv.namehint = "in_port_3upto8"} : (i8) -> i5
# CHECK:         %1 = comb.extract %in_port from 0 {sv.namehint = "in_port_0upto5"} : (i8) -> i5
# CHECK:         msft.output %0, %1 : i5, i5
# CHECK:       }


class MyMod(Module):
  in_port = Input(dim(8))
  out0 = Output(dim(5))
  out1 = Output(dim(5))

  @generator
  def construct(mod):
    # Partial lower slice
    mod.out0 = mod.in_port[3:]
    # partial upper slice
    mod.out1 = mod.in_port[:5]


top = System([MyMod])
top.generate()
top.print()
