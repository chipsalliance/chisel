# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import circt
from circt.dialects import hw, sv

from circt import ir

with ir.Context() as ctx, ir.Location.unknown() as loc:
  circt.register_dialects(ctx)
  ctx.allow_unregistered_dialects = True

  sv_attr = sv.SVAttributeAttr.get("fold", "false")
  print(f"sv_attr: {sv_attr} {sv_attr.name} {sv_attr.expression}")
  # CHECK: sv_attr: #sv.attribute<"fold" = "false"> fold false

  sv_attr = sv.SVAttributeAttr.get("no_merge")
  print(f"sv_attr: {sv_attr} {sv_attr.name} {sv_attr.expression}")
  # CHECK: sv_attr: #sv.attribute<"no_merge"> no_merge None

  i1 = ir.IntegerType.get_signless(1)
  i1_inout = hw.InOutType.get(i1)

  m = ir.Module.create()
  with ir.InsertionPoint(m.body):
    wire_op = sv.WireOp(i1_inout, "wire1")
    wire_op.attributes["sv.attributes"] = ir.ArrayAttr.get([sv_attr])
    print(wire_op)
    # CHECK: %wire1 = sv.wire {sv.attributes = [#sv.attribute<"no_merge">]} : !hw.inout<i1>

    reg_op = sv.RegOp(i1_inout, "reg1")
    reg_op.attributes["sv.attributes"] = ir.ArrayAttr.get([sv_attr])
    print(reg_op)
    # CHECK: %reg1 = sv.reg  {sv.attributes = [#sv.attribute<"no_merge">]} : !hw.inout<i1>
