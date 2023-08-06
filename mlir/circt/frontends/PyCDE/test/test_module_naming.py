# RUN: %PYTHON% %s %t 2>&1 | FileCheck %s

import pycde
import pycde.dialects.hw

import sys


@pycde.modparams
def Parameterized(param):

  class TestModule(pycde.Module):
    x = pycde.Input(pycde.types.i1)
    y = pycde.Output(pycde.types.i1)

    @pycde.generator
    def construct(ports):
      ports.y = ports.x

  return TestModule


class UnParameterized(pycde.Module):
  x = pycde.Input(pycde.types.i1)
  y = pycde.Output(pycde.types.i1)

  @pycde.generator
  def construct(ports):
    ports.y = ports.x


class Test(pycde.Module):
  inputs = []
  outputs = []

  @pycde.generator
  def build(_):
    c1 = pycde.dialects.hw.ConstantOp(pycde.types.i1, 1)
    Parameterized(1)(x=c1)
    Parameterized(1)(x=c1)
    Parameterized(2)(x=c1)
    Parameterized(2)(x=c1)
    UnParameterized(x=c1)
    UnParameterized(x=c1)


# CHECK: hw.module @TestModule_param1
# CHECK-NOT: hw.module @TestModule_param1
# CHECK: hw.module @TestModule_param2
# CHECK-NOT: hw.module @TestModule_param2
# CHECK: hw.module @UnParameterized
# CHECK-NOT: hw.module @UnParameterized
t = pycde.System([Test], output_directory=sys.argv[1])
t.generate()
t.run_passes()
t.print()
