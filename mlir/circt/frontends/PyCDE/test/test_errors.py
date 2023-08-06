# RUN: %PYTHON% py-split-input-file.py %s | FileCheck %s

from pycde import Clock, Input, types, System
from pycde.module import AppID, generator, Module, modparams
from pycde.testing import unittestmodule


# CHECK: TypeError: Module parameter definitions cannot have *args
@modparams
def foo(*args):
  pass


# -----


# CHECK: TypeError: Module parameter definitions cannot have **kwargs
@modparams
def bar(**kwargs):
  pass


# -----


@unittestmodule()
class ClkError(Module):
  a = Input(types.i32)

  @generator
  def build(ports):
    # CHECK: ValueError: If 'clk' not specified, must be in clock block
    ports.a.reg()


# -----


@unittestmodule()
class AppIDError(Module):

  @generator
  def build(ports):
    c = types.i32(4)
    # CHECK: ValueError: AppIDs can only be attached to ops with symbols
    c.appid = AppID("c", 0)


# -----


class Test(Module):
  clk = Clock()
  x = Input(types.i32)

  @generator
  def build(ports):
    ports.x.reg(appid=AppID("reg", 5))


t = System([Test], name="Test")
t.generate()

inst = t.get_instance(Test)
# CHECK: reg[8] not found
inst.reg[8]

# -----


@unittestmodule()
class OperatorError(Module):
  a = Input(types.i32)
  b = Input(types.si32)

  @generator
  def build(ports):
    # CHECK: Operator '+' is not supported on non-int or signless signals. RHS operand should be cast .as_sint()/.as_uint() if possible.
    ports.b + ports.a


# -----


@unittestmodule()
class OperatorError2(Module):
  a = Input(types.i32)
  b = Input(types.si32)

  @generator
  def build(ports):
    # CHECK: Comparisons of signed/unsigned integers to Bits<32> not supported. RHS operand should be cast .as_sint()/.as_uint() if possible.
    ports.b == ports.a
