#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .dialects import comb
from .signals import BitVectorSignal, Signal

import ctypes
from contextvars import ContextVar
import inspect
from typing import Dict, Optional, Tuple

# 'Else' and 'EndIf' need to know the current 'If' context, so we need to track
# it.
_current_if_stmt = ContextVar("current_pycde_if_stmt")

# Note: these constructs play around with Python stack frames. They use a Python
# C function to flush the changes back to the interpreter.
# PyFrame_LocalsToFast(...) is specific to CPython and as such, is
# implementation-defined. So it _could_ change in subsequent versions of Python.
# It is, however, commonly used to modify stack frames so would break many users
# if it were changed. (So it probably won't be.) Tested with all the Python
# versions for which we produce wheels on PyPI.


class If:
  """Syntactic sugar for creation of muxes with if-then-else-ish behavioral
  syntax.

  ```
  @module
  class IfDemo:
    cond = Input(types.i1)
    out = Output(types.i8)

    @generator
    def build(ports):
      with If(ports.cond):
        v = types.i8(1)
      with Else:
        v = types.i8(0)
      EndIf()
      ports.out = v
  ```"""

  def __init__(self, cond: BitVectorSignal):
    if (cond.type.width != 1):
      raise TypeError("'Cond' bit width must be 1")
    self._cond = cond
    self._muxes: Dict[str, Tuple[Signal, Signal]] = {}

  @staticmethod
  def current() -> Optional[If]:
    return _current_if_stmt.get(None)

  def __enter__(self):
    self._old_system_token = _current_if_stmt.set(self)
    # Keep all the important logic in the _IfBlock class so we can share it with
    # 'Else'.
    self.then = _IfBlock(True)
    self.then.__enter__(stack_level=2)

  def __exit__(self, exc_type, exc_value, traceback):
    if exc_value is not None:
      return
    self.then.__exit__(exc_type, exc_value, traceback, stack_level=2)

  def _finalize(self):
    _current_if_stmt.reset(self._old_system_token)

    # Create the set of muxes from the 'then' and/or 'else' blocks.
    new_locals: Dict[str, Signal] = {}
    for (varname, (else_value, then_value)) in self._muxes.items():
      # New values need to have a value assigned from both the 'then' and 'else'.
      if then_value is None or else_value is None:
        continue
      # And their types must match.
      if then_value.type != else_value.type:
        raise TypeError(
            f"'Then' and 'Else' values must have same type for {varname}" +
            f" ({then_value.type} vs {else_value.type})")
      then_value.name = f"{varname}_thenvalue"
      else_value.name = f"{varname}_elsevalue"
      mux = comb.MuxOp(self._cond, then_value, else_value)
      mux.name = varname
      new_locals[varname] = mux

    # Update the stack frame with the new muxes as locals.
    s = inspect.stack()[2][0]
    s.f_locals.update(new_locals)
    ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(s), ctypes.c_int(1))


class _IfBlock:
  """Track additions and changes to the current stack frame."""

  def __init__(self, is_then: bool):
    self._is_then = is_then

  # Capture the current scope of the stack frame and store a copy.
  def __enter__(self, stack_level=1):
    s = inspect.stack()[stack_level][0]
    self._scope = dict(s.f_locals)

  def __exit__(self, exc_type, exc_val, exc_tb, stack_level=1):
    # Don't do nothing if an exception was thrown.
    if exc_val is not None:
      return

    if_stmt = If.current()
    s = inspect.stack()[stack_level][0]
    lcls_to_del = set()
    new_lcls: Dict[str, Signal] = {}
    for (varname, value) in s.f_locals.items():
      # Only operate on Values.
      if not isinstance(value, Signal):
        continue
      # If the value was in the original scope and it hasn't changed, don't
      # touch it.
      if varname in self._scope and self._scope[varname] is value:
        continue

      # Ensure that an entry for the mux exists. If that variable exists in the
      # outer scope, use it as a default.
      if varname not in if_stmt._muxes:
        if varname in self._scope:
          if_stmt._muxes[varname] = (self._scope[varname], self._scope[varname])
        else:
          if_stmt._muxes[varname] = (None, None)

      # Fill in the correct tuple entry.
      m = if_stmt._muxes[varname]
      if self._is_then:
        if_stmt._muxes[varname] = (m[0], value)
      else:
        if_stmt._muxes[varname] = (value, m[1])

      # Restore the original scope.
      if varname in self._scope:
        new_lcls[varname] = self._scope[varname]
      else:
        lcls_to_del.add(varname)

    # Delete the variables which were not in the original scope.
    for varname in lcls_to_del:
      del s.f_locals[varname]
      ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(s),
                                            ctypes.c_int(1))

    # Restore existing locals to their original values.
    s.f_locals.update(new_lcls)
    ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(s), ctypes.c_int(1))


def Else():
  return _IfBlock(False)


def EndIf():
  c = If.current()
  assert c, "EndIf() called without matching If()"
  c._finalize()
