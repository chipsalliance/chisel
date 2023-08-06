# RUN: %PYTHON% py-split-input-file.py %s | FileCheck %s

from pycde import System, Input, Output, generator
from pycde import fsm
from pycde.types import types


class FSM(fsm.Machine):
  # CHECK: ValueError: Input port a has width 2. For now, FSMs only support i1 inputs.
  a = Input(types.i2)
  A = fsm.State(initial=True)


# -----


# CHECK: ValueError: No initial state specified, please create a state with `initial=True`.
class FSM(fsm.Machine):
  a = Input(types.i1)
  A = fsm.State()


# -----


# CHECK: ValueError: Multiple initial states specified (B, A).
class FSM(fsm.Machine):
  a = Input(types.i1)
  A = fsm.State(initial=True)
  B = fsm.State(initial=True)
