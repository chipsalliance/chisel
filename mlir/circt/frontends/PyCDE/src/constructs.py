#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .common import Clock, Input, Output
from .dialects import comb, msft, sv
from .module import generator, modparams, Module, _BlockContext
from .signals import ArraySignal, BitsSignal, BitVectorSignal, Signal
from .signals import get_slice_bounds, _FromCirctValue
from .types import dim, types, InOut, Type

from .circt import ir
from .circt.support import BackedgeBuilder
from .circt.dialects import msft as raw_msft

import typing
from typing import List, Union


def NamedWire(type_or_value: Union[Type, Signal], name: str):
  """Create a named wire which is guaranteed to appear in the Verilog output.
  This construct precludes many optimizations (since it introduces an
  optimization barrier) so it should be used sparingly."""

  assert name is not None
  value = None
  type = type_or_value
  if isinstance(type_or_value, Signal):
    type = type_or_value.type
    value = type_or_value

  class NamedWire(type._get_value_class()):

    def __init__(self):
      self.assigned_value = None
      # TODO: We assume here that names are unique within a module, which isn't
      # necessarily the case. We may have to introduce a module-scope list of
      # inner_symbols purely for the purpose of disallowing the SV
      # canonicalizers to eliminate wires!
      uniq_name = _BlockContext.current().uniquify_symbol(name)
      self.wire_op = sv.WireOp(InOut(type), name, sym_name=uniq_name)
      read_val = sv.ReadInOutOp(self.wire_op)
      super().__init__(read_val, type)
      self.name = name

    def assign(self, new_signal: Signal):
      if self.assigned_value is not None:
        raise ValueError("Cannot assign value to Wire twice.")
      if new_signal.type != self.type:
        raise TypeError(
            f"Cannot assign {new_signal.value.type} to {self.value.type}")
      sv.AssignOp(self.wire_op, new_signal.value)
      self.assigned_value = new_signal
      return self

  w = NamedWire()
  if value is not None:
    w.assign(value)
  return w


def Wire(type: Type, name: str = None):
  """Declare a wire. Used to create backedges. Must assign exactly once. If
  'name' is specified, use 'NamedWire' instead."""

  class WireValue(type._get_value_class()):

    def __init__(self):
      self._backedge = BackedgeBuilder.create(type._type,
                                              "wire" if name is None else name,
                                              None)
      super().__init__(self._backedge.result, type)
      if name is not None:
        self.name = name
      self._orig_name = name
      self.assign_parts = None

    def assign(self, new_value: Union[Signal, object]):
      if self._backedge is None:
        raise ValueError("Cannot assign value to Wire twice.")
      if not isinstance(new_value, Signal):
        new_value = type(new_value)
      if new_value.type != self.type:
        raise TypeError(
            f"Cannot assign {new_value.value.type} to {self.value.type}")

      msft.replaceAllUsesWith(self._backedge.result, new_value.value)
      self._backedge.erase()
      self._backedge = None
      self.value = new_value.value
      if self._orig_name is not None:
        self.name = self._orig_name
      return new_value

    def __setitem__(self, idxOrSlice: Union[int, slice], value):
      if self.assign_parts is None:
        self.assign_parts = [None] * self.type.width
      lo, hi = get_slice_bounds(self.type.width, idxOrSlice)
      assert hi <= self.type.width
      width = hi - lo
      assert width == value.type.width
      for i in range(lo, hi):
        assert self.assign_parts[i] is None
        self.assign_parts[i] = value
      if all([p is not None for p in self.assign_parts]):
        concat_operands = [self.assign_parts[0]]
        last = self.assign_parts[0]
        for p in self.assign_parts:
          if p is last:
            continue
          last = p
          concat_operands.append(p)
        concat_operands.reverse()
        self.assign(BitsSignal.concat(concat_operands))

  return WireValue()


def Reg(type: Type,
        clk: Signal = None,
        rst: Signal = None,
        rst_value=0,
        ce: Signal = None):
  """Declare a register. Must assign exactly once."""

  class RegisterValue(type._get_value_class()):

    def assign(self, new_value: Signal):
      if self._wire is None:
        raise ValueError("Cannot assign value to Reg twice.")
      self._wire.assign(new_value)
      self._wire = None

  # Create a wire and register it.
  wire = Wire(type)
  if rst_value is not None and not isinstance(rst_value, Signal):
    rst_value = type(rst_value)
  value = RegisterValue(wire.reg(clk=clk, rst=rst, rst_value=rst_value, ce=ce),
                        type)
  value._wire = wire
  return value


def ControlReg(clk: Signal, rst: Signal, asserts: List[Signal],
               resets: List[Signal]) -> BitVectorSignal:
  """Constructs a 'control register' and returns the output. Asserts are signals
  which causes the output to go high (on the next cycle). Resets do the
  opposite. If both an assert and a reset are active on the same cycle, the
  assert takes priority."""

  @modparams
  def ControlReg(num_asserts: int, num_resets: int):

    class ControlReg(Module):
      clk = Clock()
      rst = Input(types.i1)
      out = Output(types.i1)
      asserts = Input(dim(1, num_asserts))
      resets = Input(dim(1, num_resets))

      @generator
      def generate(ports):
        a = ports.asserts.or_reduce()
        r = ports.resets.or_reduce()
        reg = Reg(types.i1, ports.clk, ports.rst)
        reg.name = "state"
        next_value = Mux(a, Mux(r, reg, types.i1(0)), types.i1(1))
        reg.assign(next_value)
        ports.out = reg

    return ControlReg

  return ControlReg(len(asserts), len(resets))(clk=clk,
                                               rst=rst,
                                               asserts=asserts,
                                               resets=resets).out


def Mux(sel: BitVectorSignal, *data_inputs: typing.List[Signal]):
  """Create a single mux from a list of values."""
  num_inputs = len(data_inputs)
  if num_inputs == 0:
    raise ValueError("'Mux' must have 1 or more data input")
  if num_inputs == 1:
    return data_inputs[0]
  if sel.type.width != (num_inputs - 1).bit_length():
    raise TypeError("'Sel' bit width must be clog2 of number of inputs")

  input_names = [
      i.name if i.name is not None else f"in{idx}"
      for idx, i in enumerate(data_inputs)
  ]
  if num_inputs == 2:
    m = comb.MuxOp(sel, data_inputs[1], data_inputs[0])
  else:
    a = ArraySignal.create(data_inputs)
    a.name = "arr_" + "_".join(input_names)
    m = a[sel]

  m.name = f"mux_{sel.name}_" + "_".join(input_names)
  return m


def SystolicArray(row_inputs: ArraySignal, col_inputs: ArraySignal, pe_builder):
  """Build a systolic array."""

  row_inputs_type = row_inputs.type
  col_inputs_type = col_inputs.type

  dummy_op = ir.Operation.create("dummy", regions=1)
  pe_block = dummy_op.regions[0].blocks.append(
      row_inputs_type.element_type._type, col_inputs_type.element_type._type)
  with ir.InsertionPoint(pe_block):
    result = pe_builder(_FromCirctValue(pe_block.arguments[0]),
                        _FromCirctValue(pe_block.arguments[1]))
    if not isinstance(result, Signal):
      raise TypeError(
          f"pe_builder function must return a `Signal` not {result}")
    pe_output_type = result.type
    msft.PEOutputOp(result)

  sa_result_type = dim(pe_output_type, col_inputs_type.size,
                       row_inputs_type.size)
  array = raw_msft.SystolicArrayOp(sa_result_type._type, row_inputs.value,
                                   col_inputs.value)
  dummy_op.regions[0].blocks[0].append_to(array.regions[0])
  dummy_op.operation.erase()

  return _FromCirctValue(array.peOutputs)
