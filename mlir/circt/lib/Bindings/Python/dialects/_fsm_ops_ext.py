#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from . import fsm as fsm
from .. import support
from ..ir import *


def _get_or_add_single_block(region, args=[]):
  if len(region.blocks) == 0:
    region.blocks.append(*args)
  return region.blocks[0]


class MachineOp:

  def __init__(self,
               name,
               initial_state,
               input_ports,
               output_ports,
               *,
               attributes={},
               loc=None,
               ip=None):
    attributes["sym_name"] = StringAttr.get(name)
    attributes["initialState"] = StringAttr.get(initial_state)

    input_types = []
    output_types = []
    for (i, (_, port_type)) in enumerate(input_ports):
      input_types.append(port_type)

    for (i, (_, port_type)) in enumerate(output_ports):
      output_types.append(port_type)

    attributes["function_type"] = TypeAttr.get(
        FunctionType.get(inputs=input_types, results=output_types))

    OpView.__init__(
        self,
        self.build_generic(attributes=attributes,
                           results=[],
                           operands=[],
                           successors=None,
                           regions=1,
                           loc=loc,
                           ip=ip))

    _get_or_add_single_block(self.body, self.type.inputs)

  @property
  def type(self):
    return FunctionType(TypeAttr(self.attributes["function_type"]).value)

  def instantiate(self, name: str, loc=None, ip=None, **kwargs):
    """ FSM Instantiation function"""
    in_names = support.attribute_to_var(self.attributes['in_names'])
    inputs = [kwargs[port].value for port in in_names]

    # Clock and resets are not part of the input ports of the FSM, but
    # it is at the point of `fsm.hw_instance` instantiation that they
    # must be connected.
    # Attach backedges to these, and associate these backedges to the operation.
    # They can then be accessed at the point of instantiation and assigned.
    clock = support.BackedgeBuilder().create(
        IntegerType.get_signed(1),
        StringAttr(self.attributes['clock_name']).value, self)
    reset = support.BackedgeBuilder().create(
        IntegerType.get_signed(1),
        StringAttr(self.attributes['reset_name']).value, self)

    op = fsm.HWInstanceOp(outputs=self.type.results,
                          inputs=inputs,
                          sym_name=StringAttr.get(name),
                          machine=FlatSymbolRefAttr.get(self.sym_name.value),
                          clock=clock.result,
                          reset=reset.result if reset else None,
                          loc=loc,
                          ip=ip)
    op.backedges = {}

    def set_OpOperand(name, backedge):
      index = None
      for i, operand in enumerate(op.operands):
        if operand == backedge.result:
          index = i
          break
      assert index is not None
      op_operand = support.OpOperand(op, index, op.operands[index], op)
      setattr(op, f'_{name}_backedge', op_operand)
      op.backedges[i] = backedge

    set_OpOperand('clock', clock)
    if reset:
      set_OpOperand('reset', reset)

    return op


class TransitionOp:

  def __init__(self, next_state, *, loc=None, ip=None):
    attributes = {
        "nextState": FlatSymbolRefAttr.get(next_state),
    }
    super().__init__(
        self.build_generic(attributes=attributes,
                           results=[],
                           operands=[],
                           successors=None,
                           regions=2,
                           loc=loc,
                           ip=ip))

  @staticmethod
  def create(to_state):
    op = fsm.TransitionOp(to_state)
    return op

  def set_guard(self, guard_fn):
    """Executes a function to generate a guard for the transition.
    The function is executed within the guard region of this operation."""
    guard_block = _get_or_add_single_block(self.guard)
    with InsertionPoint(guard_block):
      guard = guard_fn()
      guard_type = support.type_to_pytype(guard.type)
      if guard_type.width != 1:
        raise ValueError('The guard must be a single bit')
      fsm.ReturnOp(operand=guard)


class StateOp:

  def __init__(self, name, *, loc=None, ip=None):
    attributes = {}
    attributes["sym_name"] = StringAttr.get(name)

    OpView.__init__(
        self,
        self.build_generic(attributes=attributes,
                           results=[],
                           operands=[],
                           successors=None,
                           regions=2,
                           loc=loc,
                           ip=ip))

  @staticmethod
  def create(name):
    return fsm.StateOp(name)

  @property
  def output(self):
    return _get_or_add_single_block(super().output)

  @property
  def transitions(self):
    return _get_or_add_single_block(super().transitions)


class OutputOp:

  @staticmethod
  def create(*operands):
    return fsm.OutputOp(operands)
