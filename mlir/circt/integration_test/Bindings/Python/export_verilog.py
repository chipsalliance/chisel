# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s --check-prefix=INMEMORY
# RUN: FileCheck %s --input-file=test.sv --check-prefix=DIRECTORY

import circt
from circt.dialects import hw

from circt.ir import (Context, Location, InsertionPoint, IntegerType, Module)

import io
import os

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)

  i1 = IntegerType.get_signless(1)

  m = Module.create()
  with InsertionPoint(m.body):
    hw.HWModuleOp(name="test",
                  output_ports=[("out", i1)],
                  body_builder=lambda m: {"out": hw.ConstantOp.create(i1, 1)})

  buffer = io.StringIO()
  circt.export_verilog(m, buffer)
  print(buffer.getvalue())
  # INMEMORY: module test(
  # INMEMORY:   output out
  # INMEMORY: );
  # INMEMORY:   assign out = 1'h1;
  # INMEMORY: endmodule

  cwd = os.getcwd()
  circt.export_split_verilog(m, cwd)
  # DIRECTORY: module test(
  # DIRECTORY:   output out
  # DIRECTORY: );
  # DIRECTORY:   assign out = 1'h1;
  # DIRECTORY: endmodule
