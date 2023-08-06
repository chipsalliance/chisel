# RUN: %PYTHON% py-split-input-file.py %s | FileCheck %s

from pycde import Input, generator, dim, Module
from pycde.constructs import Mux
from pycde.testing import unittestmodule
from pycde.types import Bits, SInt, UInt


@unittestmodule()
class Mux1(Module):

  In = Input(dim(3, 4, 5))
  Sel = Input(dim(8))

  @generator
  def create(ports):
    # CHECK: TypeError: 'Sel' bit width must be clog2 of number of inputs
    Mux(ports.Sel, ports.In[3], ports.In[1])


# -----


@unittestmodule()
class Mux2(Module):

  Sel = Input(dim(8))

  @generator
  def create(ports):
    # CHECK: ValueError: 'Mux' must have 1 or more data input
    Mux(ports.Sel)


# -----


@unittestmodule(print=False)
class WrongInts(Module):

  @generator
  def construct(mod):
    b4 = Bits(4)
    si4 = SInt(4)
    ui4 = UInt(4)

    try:
      # CHECK: Bits can only be created from ints, not str
      b4("foo")
      assert False
    except ValueError as e:
      print(e)

    try:
      # CHECK: 300 overflows type Bits<4>
      b4(300)
      assert False
    except ValueError as e:
      print(e)

    try:
      # CHECK: 15 overflows type SInt<4>
      si4(15)
      assert False
    except ValueError as e:
      print(e)

    try:
      # CHECK: UInt can only store positive numbers, not -1
      ui4(-1)
      assert False
    except ValueError as e:
      print(e)
