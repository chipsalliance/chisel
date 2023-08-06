from pycde import Input, Output, Module, System
from pycde import generator, modparams
from pycde.types import UInt


@modparams
def AddInts(width: int):

  class AddInts(Module):
    a = Input(UInt(width))
    b = Input(UInt(width))
    c = Output(UInt(width + 1))

    @generator
    def build(self):
      self.c = self.a + self.b

  return AddInts


class Top(Module):
  a = Input(UInt(32))
  b = Input(UInt(32))
  c = Output(UInt(33))

  @generator
  def construct(self):
    add_ints_m = AddInts(32)
    add_ints = add_ints_m(a=self.a, b=self.b)
    self.c = add_ints.c


system = System(Top, name="ExampleParams")
system.compile()
