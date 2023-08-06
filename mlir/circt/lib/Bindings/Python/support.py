#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from . import ir

from contextlib import AbstractContextManager
from contextvars import ContextVar
from typing import List

_current_backedge_builder = ContextVar("current_bb")


class ConnectionError(RuntimeError):
  pass


class UnconnectedSignalError(ConnectionError):

  def __init__(self, module: str, port_names: List[str]):
    super().__init__(
        f"Ports {port_names} unconnected in design module {module}.")


def get_value(obj) -> ir.Value:
  """Resolve a Value from a few supported types."""

  if isinstance(obj, ir.Value):
    return obj
  if hasattr(obj, "result"):
    return obj.result
  if hasattr(obj, "value"):
    return obj.value
  return None


def connect(destination, source):
  """A convenient way to use BackedgeBuilder."""
  if not isinstance(destination, OpOperand):
    raise TypeError(
        f"cannot connect to destination of type {type(destination)}. "
        "Must be OpOperand.")
  value = get_value(source)
  if value is None:
    raise TypeError(f"cannot connect from source of type {type(source)}")

  index = destination.index
  destination.operation.operands[index] = value
  if destination.backedge_owner and \
     index in destination.backedge_owner.backedges:
    destination.backedge_owner.backedges[index].erase()
    del destination.backedge_owner.backedges[index]


def var_to_attribute(obj, none_on_fail: bool = False) -> ir.Attribute:
  """Create an MLIR attribute from a Python object for a few common cases."""
  if isinstance(obj, ir.Attribute):
    return obj
  if isinstance(obj, bool):
    return ir.BoolAttr.get(obj)
  if isinstance(obj, int):
    attrTy = ir.IntegerType.get_signless(64)
    return ir.IntegerAttr.get(attrTy, obj)
  if isinstance(obj, str):
    return ir.StringAttr.get(obj)
  if isinstance(obj, list):
    arr = [var_to_attribute(x, none_on_fail) for x in obj]
    if all(arr):
      return ir.ArrayAttr.get(arr)
    return None
  if none_on_fail:
    return None
  raise TypeError(f"Cannot convert type '{type(obj)}' to MLIR attribute")


# There is currently no support in MLIR for querying type types. The
# conversation regarding how to achieve this is ongoing and I expect it to be a
# long one. This is a way that works for now.
def type_to_pytype(t) -> ir.Type:

  if not isinstance(t, ir.Type):
    raise TypeError("type_to_pytype only accepts MLIR Type objects")

  # If it's not the root type, assume it's already been downcasted and don't do
  # the expensive probing below.
  if t.__class__ != ir.Type:
    return t

  from .dialects import esi, hw
  try:
    return ir.IntegerType(t)
  except ValueError:
    pass
  try:
    return ir.NoneType(t)
  except ValueError:
    pass
  try:
    return hw.ArrayType(t)
  except ValueError:
    pass
  try:
    return hw.StructType(t)
  except ValueError:
    pass
  try:
    return hw.TypeAliasType(t)
  except ValueError:
    pass
  try:
    return hw.InOutType(t)
  except ValueError:
    pass
  try:
    return esi.ChannelType(t)
  except ValueError:
    pass

  raise TypeError(f"Cannot convert {repr(t)} to python type")


# There is currently no support in MLIR for querying attribute types. The
# conversation regarding how to achieve this is ongoing and I expect it to be a
# long one. This is a way that works for now.
def attribute_to_var(attr):

  if attr is None:
    return None
  if not isinstance(attr, ir.Attribute):
    raise TypeError("attribute_to_var only accepts MLIR Attributes")

  # If it's not the root type, assume it's already been downcasted and don't do
  # the expensive probing below.
  if attr.__class__ != ir.Attribute and hasattr(attr, "value"):
    return attr.value

  from .dialects import hw, om
  try:
    return ir.BoolAttr(attr).value
  except ValueError:
    pass
  try:
    return ir.IntegerAttr(attr).value
  except ValueError:
    pass
  try:
    return ir.StringAttr(attr).value
  except ValueError:
    pass
  try:
    return ir.FlatSymbolRefAttr(attr).value
  except ValueError:
    pass
  try:
    return ir.TypeAttr(attr).value
  except ValueError:
    pass
  try:
    arr = ir.ArrayAttr(attr)
    return [attribute_to_var(x) for x in arr]
  except ValueError:
    pass
  try:
    dict = ir.DictAttr(attr)
    return {i.name: attribute_to_var(i.attr) for i in dict}
  except ValueError:
    pass
  try:
    return attribute_to_var(om.ReferenceAttr(attr).inner_ref)
  except ValueError:
    pass
  try:
    ref = hw.InnerRefAttr(attr)
    return (ir.StringAttr(ref.module).value, ir.StringAttr(ref.name).value)
  except ValueError:
    pass
  try:
    return list(map(attribute_to_var, om.ListAttr(attr)))
  except ValueError:
    pass

  raise TypeError(f"Cannot convert {repr(attr)} to python value")


def get_self_or_inner(mlir_type):
  from .dialects import hw
  if type(mlir_type) is ir.Type:
    mlir_type = type_to_pytype(mlir_type)
  if isinstance(mlir_type, hw.TypeAliasType):
    return type_to_pytype(mlir_type.inner_type)
  return mlir_type


class BackedgeBuilder(AbstractContextManager):

  class Edge:

    def __init__(self,
                 creator,
                 type: ir.Type,
                 backedge_name: str,
                 op_view,
                 instance_of: ir.Operation,
                 loc: ir.Location = None):
      self.creator: BackedgeBuilder = creator
      self.dummy_op = ir.Operation.create("builtin.unrealized_conversion_cast",
                                          [type],
                                          loc=loc)
      self.instance_of = instance_of
      self.op_view = op_view
      self.port_name = backedge_name
      self.erased = False

    @property
    def result(self):
      return self.dummy_op.result

    def erase(self):
      if self.erased:
        return
      if self in self.creator.edges:
        self.creator.edges.remove(self)
        self.dummy_op.operation.erase()

  def __init__(self):
    self.edges = set()

  @staticmethod
  def current():
    bb = _current_backedge_builder.get(None)
    if bb is None:
      raise RuntimeError("No backedge builder found in context!")
    return bb

  @staticmethod
  def create(*args, **kwargs):
    return BackedgeBuilder.current()._create(*args, **kwargs)

  def _create(self,
              type: ir.Type,
              port_name: str,
              op_view,
              instance_of: ir.Operation = None,
              loc: ir.Location = None):
    edge = BackedgeBuilder.Edge(self, type, port_name, op_view, instance_of,
                                loc)
    self.edges.add(edge)
    return edge

  def __enter__(self):
    self.old_bb_token = _current_backedge_builder.set(self)

  def __exit__(self, exc_type, exc_value, traceback):
    if exc_value is not None:
      return
    _current_backedge_builder.reset(self.old_bb_token)
    errors = []
    for edge in list(self.edges):
      # TODO: Make this use `UnconnectedSignalError`.
      msg = "Backedge:   " + edge.port_name + "\n"
      if edge.instance_of is not None:
        msg += "InstanceOf: " + str(edge.instance_of).split(" {")[0] + "\n"
      if edge.op_view is not None:
        op = edge.op_view.operation
        msg += "Instance:   " + str(op)
      errors.append(msg)

    if errors:
      errors.insert(0, "Uninitialized backedges remain in circuit!")
      raise RuntimeError("\n".join(errors))


class OpOperand:
  __slots__ = ["index", "operation", "value", "backedge_owner"]

  def __init__(self,
               operation: ir.Operation,
               index: int,
               value,
               backedge_owner=None):
    if not isinstance(index, int):
      raise TypeError("Index must be int")
    self.index = index

    if not hasattr(operation, "operands"):
      raise TypeError("Operation must be have 'operands' attribute")
    self.operation = operation

    self.value = value
    self.backedge_owner = backedge_owner

  @property
  def type(self):
    return self.value.type


class NamedValueOpView:
  """Helper class to incrementally construct an instance of an operation that
     names its operands and results"""

  def __init__(self,
               cls,
               data_type=None,
               input_port_mapping=None,
               pre_args=None,
               post_args=None,
               needs_result_type=False,
               **kwargs):
    # Set defaults
    if input_port_mapping is None:
      input_port_mapping = {}
    if pre_args is None:
      pre_args = []
    if post_args is None:
      post_args = []

    # Set result_indices to name each result.
    result_names = self.result_names()
    result_indices = {}
    for i in range(len(result_names)):
      result_indices[result_names[i]] = i

    # Set operand_indices to name each operand. Give them an initial value,
    # either from input_port_mapping or a default value.
    backedges = {}
    operand_indices = {}
    operand_values = []
    operand_names = self.operand_names()
    for i in range(len(operand_names)):
      arg_name = operand_names[i]
      operand_indices[arg_name] = i
      if arg_name in input_port_mapping:
        value = get_value(input_port_mapping[arg_name])
        operand = value
      else:
        backedge = self.create_default_value(i, data_type, arg_name)
        backedges[i] = backedge
        operand = backedge.result
      operand_values.append(operand)

    # Some ops take a list of operand values rather than splatting them out.
    if isinstance(data_type, list):
      operand_values = [operand_values]

    # In many cases, result types are inferred, and we do not need to pass
    # data_type to the underlying constructor. It must be provided to
    # NamedValueOpView in cases where we need to build backedges, but should
    # generally not be passed to the underlying constructor in this case. There
    # are some oddball ops that must pass it, even when building backedges, and
    # these set needs_result_type=True.
    if data_type is not None and (needs_result_type or len(backedges) == 0):
      pre_args.insert(0, data_type)

    self.opview = cls(*pre_args, *operand_values, *post_args, **kwargs)
    self.operand_indices = operand_indices
    self.result_indices = result_indices
    self.backedges = backedges

  def __getattr__(self, name):
    # Check for the attribute in the arg name set.
    if "operand_indices" in dir(self) and name in self.operand_indices:
      index = self.operand_indices[name]
      value = self.opview.operands[index]
      return OpOperand(self.opview.operation, index, value, self)

    # Check for the attribute in the result name set.
    if "result_indices" in dir(self) and name in self.result_indices:
      index = self.result_indices[name]
      value = self.opview.results[index]
      return OpOperand(self.opview.operation, index, value, self)

    # Forward "attributes" attribute from the operation.
    if name == "attributes":
      return self.opview.operation.attributes

    # If we fell through to here, the name isn't a result.
    raise AttributeError(f"unknown port name {name}")

  def create_default_value(self, index, data_type, arg_name):
    return BackedgeBuilder.create(data_type, arg_name, self)

  @property
  def operation(self):
    """Get the operation associated with this builder."""
    return self.opview.operation
