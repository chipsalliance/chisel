# REQUIRES: bindings_python
# RUN: %PYTHON% %s 2>&1 | FileCheck %s

import circt
from circt.dialects import comb, hw

from circt.ir import Context, Location, InsertionPoint, IntegerType, IntegerAttr, Module, MLIRError

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)

  i32 = IntegerType.get_signless(32)
  i31 = IntegerType.get_signless(31)

  m = Module.create()
  with InsertionPoint(m.body):

    def build(module):
      const1 = hw.ConstantOp(IntegerAttr.get(i32, 1))
      const2 = hw.ConstantOp(IntegerAttr.get(i31, 1))

      # CHECK: op requires all operands to have the same type
      div = comb.DivSOp.create(const1.result, const2.result)
      try:
        div.opview.verify()
      except MLIRError as e:
        print(e)

      # CHECK: result type cannot be None
      try:
        comb.DivSOp.create()
      except ValueError as e:
        print(e)

    hw.HWModuleOp(name="test", body_builder=build)
