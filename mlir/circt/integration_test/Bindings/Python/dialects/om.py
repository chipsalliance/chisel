# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import circt
from circt.dialects import om
from circt.ir import Context, InsertionPoint, Location, Module
from circt.support import var_to_attribute

from dataclasses import dataclass

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)

  module = Module.parse("""
  module {
    %sym = om.constant #om.ref<<@Root::@x>> : !om.ref

    om.class @Test(%param: i64) {
      om.class.field @field, %param : i64

      %0 = om.object @Child() : () -> !om.class.type<@Child>
      om.class.field @child, %0 : !om.class.type<@Child>

      om.class.field @reference, %sym : !om.ref

      %list = om.constant #om.list<!om.string, ["X" : !om.string, "Y" : !om.string]> : !om.list<!om.string>
      om.class.field @list, %list : !om.list<!om.string>
    }

    om.class @Child() {
      %0 = om.constant 14 : i64
      om.class.field @foo, %0 : i64
    }

    hw.module @Root(%clock: i1) -> () {
      %0 = sv.wire sym @x : !hw.inout<i1>
    }
  }
  """)

  evaluator = om.Evaluator(module)

# Test instantiate failure.

try:
  obj = evaluator.instantiate("Test")
except ValueError as e:
  # CHECK: actual parameter list length (0) does not match
  # CHECK: actual parameters:
  # CHECK: formal parameters:
  # CHECK: unable to instantiate object, see previous error(s)
  print(e)

# Test get field failure.

try:
  obj = evaluator.instantiate("Test", 42)
  obj.foo
except ValueError as e:
  # CHECK: field "foo" does not exist
  # CHECK: see current operation:
  # CHECK: unable to get field, see previous error(s)
  print(e)

# Test instantiate success.

obj = evaluator.instantiate("Test", 42)

# CHECK: 42
print(obj.field)
# CHECK: 14
print(obj.child.foo)
# CHECK: ('Root', 'x')
print(obj.reference)

for (name, field) in obj:
  # CHECK: name: child, field: <circt.dialects.om.Object object
  # CHECK: name: field, field: 42
  # CHECK: name: reference, field: ('Root', 'x')
  print(f"name: {name}, field: {field}")

# CHECK: ['X', 'Y']
print(obj.list)
