# RUN: %PYTHON% %s %t | FileCheck %s

from pycde.circt.ir import Module as IrModule
from pycde.circt.dialects import hw

from pycde import Input, Output, System, generator, Module, types
from pycde.module import import_hw_module

import sys

mlir_module = IrModule.parse("""
hw.module @add(%a: i1, %b: i1) -> (out: i1) {
  %0 = comb.add %a, %b : i1
  hw.output %0 : i1
}

hw.module @and(%a: i1, %b: i1) -> (out: i1) {
  %0 = comb.and %a, %b : i1
  hw.output %0 : i1
}
""")

imported_modules = []
for op in mlir_module.body:
  if isinstance(op, hw.HWModuleOp):
    imported_module = import_hw_module(op)
    imported_modules.append(imported_module)


class Top(Module):
  a = Input(types.i1)
  b = Input(types.i1)
  out0 = Output(types.i1)
  out1 = Output(types.i1)

  @generator
  def generate(ports):
    outs = []
    for mod in imported_modules:
      outs.append(mod(a=ports.a, b=ports.b).out)

    ports.out0 = outs[0]
    ports.out1 = outs[1]


system = System([Top], output_directory=sys.argv[1])
system.generate()

# CHECK: msft.module @Top {} (%a: i1, %b: i1) -> (out0: i1, out1: i1)
# CHECK:   %add.out = hw.instance "add" @add(a: %a: i1, b: %b: i1) -> (out: i1)
# CHECK:   %and.out = hw.instance "and" @and(a: %a: i1, b: %b: i1) -> (out: i1)
# CHECK:   msft.output %add.out, %and.out : i1, i1

# CHECK: hw.module @add(%a: i1, %b: i1) -> (out: i1)
# CHECK:   %0 = comb.add %a, %b : i1
# CHECK:   hw.output %0 : i1

# CHECK: hw.module @and(%a: i1, %b: i1) -> (out: i1)
# CHECK:   %0 = comb.and %a, %b : i1
# CHECK:   hw.output %0 : i1
system.print()
