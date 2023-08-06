# RUN: rm -rf %t
# RUN: %PYTHON% %s %t

# This is intended to be a simple 'tutorial' example.  Run it as a test to
# ensure that we keep it up to date (ensure it doesn't crash).

from pycde import dim, generator, Clock, Input, Output, Module
from pycde.types import Bits
import pycde

import sys


class Mux(Module):
  clk = Clock()
  data = Input(dim(8, 14))
  sel = Input(Bits(4))

  out = Output(Bits(8))

  @generator
  def build(ports):
    sel_reg = ports.sel.reg()
    ports.out = ports.data.reg()[sel_reg].reg()


t = pycde.System([Mux], name="MuxDemo", output_directory=sys.argv[1])
t.compile()
