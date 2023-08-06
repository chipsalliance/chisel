# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import circt
from circt.support import connect
from circt.dialects import hw

from circt.ir import *
from circt.passmanager import PassManager

import sys

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)

  i32 = IntegerType.get_signless(32)

  # CHECK: !hw.array<5xi32>
  array_i32 = hw.ArrayType.get(i32, 5)
  print(array_i32)
  # CHECK: i32
  print(array_i32.element_type)

  # CHECK: !hw.struct<foo: i32, bar: !hw.array<5xi32>>
  struct = hw.StructType.get([("foo", i32), ("bar", array_i32)])
  print(struct)

  # CHECK: !hw.struct<baz: i32, qux: !hw.array<5xi32>>
  struct = hw.StructType.get([("baz", i32), ("qux", array_i32)])
  print(struct)

  m = Module.create()
  with InsertionPoint(m.body):
    # CHECK: hw.module @MyWidget(%my_input: i32) -> (my_output: i32)
    # CHECK:   hw.output %my_input : i32
    op = hw.HWModuleOp(
        name='MyWidget',
        input_ports=[('my_input', i32)],
        output_ports=[('my_output', i32)],
        body_builder=lambda module: hw.OutputOp([module.my_input]))

    # CHECK: hw.module.extern @FancyThing(%input0: i32) -> (output0: i32)
    extern = hw.HWModuleExternOp(name="FancyThing",
                                 input_ports=[("input0", i32)],
                                 output_ports=[("output0", i32)])

    one_input = hw.HWModuleOp(
        name="one_input",
        input_ports=[("a", i32)],
        parameters=[
            hw.ParamDeclAttr.get("BANKS", i32, IntegerAttr.get(i32, 5))
        ],
        body_builder=lambda m: hw.OutputOp([]),
    )
    two_inputs = hw.HWModuleOp(
        name="two_inputs",
        input_ports=[("a", i32), ("b", i32)],
        body_builder=lambda m: None,
    )
    one_output = hw.HWModuleOp(
        name="one_output",
        output_ports=[("a", i32)],
        body_builder=lambda m: hw.OutputOp(
            [hw.ConstantOp.create(i32, 46).result]),
    )
    two_outputs = hw.HWModuleOp(name="two_outputs",
                                input_ports=[("a", i32)],
                                output_ports=[("x", i32), ("y", i32)],
                                body_builder=lambda m: dict(x=m.a, y=m.a))
    three_outputs = hw.HWModuleOp(name="three_outputs",
                                  input_ports=[("a", i32)],
                                  output_ports=[("x", i32), ("y", i32),
                                                ("z", i32)],
                                  body_builder=lambda m: {
                                      "z": m.a,
                                      "x": m.a,
                                      "y": m.a
                                  })

    # CHECK-LABEL: hw.module @instance_builder_tests
    def instance_builder_body(module):
      # CHECK: %[[INST1_RESULT:.+]] = hw.instance "inst1" @one_output()
      inst1 = one_output.instantiate("inst1")

      # CHECK: hw.instance "inst2" @one_input<BANKS: i32 = 5>(a: %[[INST1_RESULT]]: i32)
      one_input.instantiate("inst2", a=inst1.a)

      # CHECK: hw.instance "inst4" @two_inputs(a: %[[INST1_RESULT]]: i32, b: %[[INST1_RESULT]]: i32)
      inst4 = two_inputs.instantiate("inst4", a=inst1.a)
      connect(inst4.b, inst1.a)

      # CHECK: %[[INST5_RESULT:.+]] = hw.instance "inst5" @MyWidget(my_input: %[[INST5_RESULT]]: i32)
      inst5 = op.instantiate("inst5")
      connect(inst5.my_input, inst5.my_output)

      # CHECK: hw.instance "inst6" @one_input<BANKS: i32 = 2>(a:
      one_input.instantiate("inst6",
                            a=inst1.a,
                            parameters={"BANKS": IntegerAttr.get(i32, 2)})

    instance_builder_tests = hw.HWModuleOp(name="instance_builder_tests",
                                           body_builder=instance_builder_body)

    # CHECK: hw.module @block_args_test(%[[PORT_NAME:.+]]: i32) ->
    # CHECK: hw.output %[[PORT_NAME]]
    hw.HWModuleOp(name="block_args_test",
                  input_ports=[("foo", i32)],
                  output_ports=[("bar", i32)],
                  body_builder=lambda module: hw.OutputOp([module.foo]))

  print(m)

  # CHECK-LABEL: === Verilog ===
  print("=== Verilog ===")

  pm = PassManager.parse("builtin.module(hw.module(hw-cleanup))")
  pm.run(m.operation)
  # CHECK: module MyWidget
  # CHECK: external module FancyThing
  circt.export_verilog(m, sys.stdout)
