from __future__ import annotations

from typing import Dict, Type

from . import hw
from .. import support
from ..ir import *


def create_parameters(parameters: dict[str, _ir.Attribute], module: ModuleLike):
  # Compute mapping from parameter name to index, and initialize array.
  mod_param_decls = module.parameters
  mod_param_decls_idxs = {
      decl.name: idx for (idx, decl) in enumerate(mod_param_decls)
  }
  inst_param_array = [None] * len(module.parameters)

  # Fill in all the parameters specified.
  if isinstance(parameters, DictAttr):
    parameters = {i.name: i.attr for i in parameters}
  for (pname, pval) in parameters.items():
    if pname not in mod_param_decls_idxs:
      raise ValueError(
          f"Could not find parameter '{pname}' in module parameter decls")
    idx = mod_param_decls_idxs[pname]
    param_decl = mod_param_decls[idx]
    inst_param_array[idx] = hw.ParamDeclAttr.get(pname, param_decl.param_type,
                                                 pval)

  # Fill in the defaults from the module param decl.
  for (idx, pval) in enumerate(inst_param_array):
    if pval is not None:
      continue
    inst_param_array[idx] = mod_param_decls[idx]

  return inst_param_array


class InstanceBuilder(support.NamedValueOpView):
  """Helper class to incrementally construct an instance of a module."""

  def __init__(self,
               module,
               name,
               input_port_mapping,
               *,
               results=None,
               parameters={},
               sym_name=None,
               loc=None,
               ip=None):
    self.module = module
    instance_name = StringAttr.get(name)
    module_name = FlatSymbolRefAttr.get(StringAttr(module.name).value)
    inst_param_array = create_parameters(parameters, module)
    if sym_name:
      inner_sym = hw.InnerSymAttr.get(StringAttr.get(sym_name))
    else:
      inner_sym = None
    pre_args = [instance_name, module_name]
    post_args = [
        ArrayAttr.get([StringAttr.get(x) for x in self.operand_names()]),
        ArrayAttr.get([StringAttr.get(x) for x in self.result_names()]),
        ArrayAttr.get(inst_param_array)
    ]
    if results is None:
      results = module.type.results

    if not isinstance(module, hw.HWModuleExternOp):
      input_name_type_lookup = {
          name: support.type_to_pytype(ty)
          for name, ty in zip(self.operand_names(), module.type.inputs)
      }
      for input_name, input_value in input_port_mapping.items():
        if input_name not in input_name_type_lookup:
          continue  # This error gets caught and raised later.
        mod_input_type = input_name_type_lookup[input_name]
        if support.type_to_pytype(input_value.type) != mod_input_type:
          raise TypeError(f"Input '{input_name}' has type '{input_value.type}' "
                          f"but expected '{mod_input_type}'")

    super().__init__(hw.InstanceOp,
                     results,
                     input_port_mapping,
                     pre_args,
                     post_args,
                     needs_result_type=True,
                     inner_sym=inner_sym,
                     loc=loc,
                     ip=ip)

  def create_default_value(self, index, data_type, arg_name):
    type = self.module.type.inputs[index]
    return support.BackedgeBuilder.create(type,
                                          arg_name,
                                          self,
                                          instance_of=self.module)

  def operand_names(self):
    arg_names = ArrayAttr(self.module.attributes["argNames"])
    arg_name_attrs = map(StringAttr, arg_names)
    return list(map(lambda s: s.value, arg_name_attrs))

  def result_names(self):
    arg_names = ArrayAttr(self.module.attributes["resultNames"])
    arg_name_attrs = map(StringAttr, arg_names)
    return list(map(lambda s: s.value, arg_name_attrs))


class ModuleLike:
  """Custom Python base class for module-like operations."""

  def __init__(
      self,
      name,
      input_ports=[],
      output_ports=[],
      *,
      parameters=[],
      attributes={},
      body_builder=None,
      loc=None,
      ip=None,
  ):
    """
    Create a module-like with the provided `name`, `input_ports`, and
    `output_ports`.
    - `name` is a string representing the module name.
    - `input_ports` is a list of pairs of string names and mlir.ir types.
    - `output_ports` is a list of pairs of string names and mlir.ir types.
    - `body_builder` is an optional callback, when provided a new entry block
      is created and the callback is invoked with the new op as argument within
      an InsertionPoint context already set for the block. The callback is
      expected to insert a terminator in the block.
    """
    # Copy the mutable default arguments. 'Cause python.
    input_ports = list(input_ports)
    output_ports = list(output_ports)
    parameters = list(parameters)
    attributes = dict(attributes)

    operands = []
    results = []
    attributes["sym_name"] = StringAttr.get(str(name))

    input_types = []
    input_names = []
    input_locs = []
    unknownLoc = Location.unknown().attr
    for (i, (port_name, port_type)) in enumerate(input_ports):
      input_types.append(port_type)
      input_names.append(StringAttr.get(str(port_name)))
      input_locs.append(unknownLoc)
    attributes["argNames"] = ArrayAttr.get(input_names)
    attributes["argLocs"] = ArrayAttr.get(input_locs)

    output_types = []
    output_names = []
    output_locs = []
    for (i, (port_name, port_type)) in enumerate(output_ports):
      output_types.append(port_type)
      output_names.append(StringAttr.get(str(port_name)))
      output_locs.append(unknownLoc)
    attributes["resultNames"] = ArrayAttr.get(output_names)
    attributes["resultLocs"] = ArrayAttr.get(output_locs)

    if len(parameters) > 0 or "parameters" not in attributes:
      attributes["parameters"] = ArrayAttr.get(parameters)

    attributes["function_type"] = TypeAttr.get(
        FunctionType.get(inputs=input_types, results=output_types))

    super().__init__(
        self.build_generic(attributes=attributes,
                           results=results,
                           operands=operands,
                           loc=loc,
                           ip=ip))

    if body_builder:
      entry_block = self.add_entry_block()

      with InsertionPoint(entry_block):
        with support.BackedgeBuilder():
          outputs = body_builder(self)
          _create_output_op(name, output_ports, entry_block, outputs)

  @property
  def type(self):
    return FunctionType(TypeAttr(self.attributes["function_type"]).value)

  @property
  def name(self):
    return self.attributes["sym_name"]

  @property
  def is_external(self):
    return len(self.regions[0].blocks) == 0

  @property
  def parameters(self) -> list[ParamDeclAttr]:
    return [
        hw.ParamDeclAttr(a) for a in ArrayAttr(self.attributes["parameters"])
    ]

  def instantiate(self,
                  name: str,
                  parameters: Dict[str, object] = {},
                  results=None,
                  loc=None,
                  ip=None,
                  **kwargs):
    return InstanceBuilder(self,
                           name,
                           kwargs,
                           parameters=parameters,
                           results=results,
                           loc=loc,
                           ip=ip)


def _create_output_op(cls_name, output_ports, entry_block, bb_ret):
  """Create the hw.OutputOp from the body_builder return."""

  # Determine if the body already has an output op.
  block_len = len(entry_block.operations)
  if block_len > 0:
    last_op = entry_block.operations[block_len - 1]
    if isinstance(last_op, hw.OutputOp):
      # If it does, the return from body_builder must be None.
      if bb_ret is not None and bb_ret != last_op:
        raise support.ConnectionError(
            f"In {cls_name}, cannot return value from body_builder and "
            "create hw.OutputOp")
      return

  # If builder didn't create an output op and didn't return anything, this op
  # mustn't have any outputs.
  if bb_ret is None:
    if len(output_ports) == 0:
      hw.OutputOp([])
      return
    raise support.ConnectionError(
        f"In {cls_name}, must return module output values")

  # Now create the output op depending on the object type returned
  outputs: list[Value] = list()

  # Only acceptable return is a dict of port, value mappings.
  if not isinstance(bb_ret, dict):
    raise support.ConnectionError(
        f"In {cls_name}, can only return a dict of port, value mappings "
        "from body_builder.")

  # A dict of `OutputPortName` -> ValueLike must be converted to a list in port
  # order.
  unconnected_ports = []
  for (name, port_type) in output_ports:
    if name not in bb_ret:
      unconnected_ports.append(name)
      outputs.append(None)
    else:
      val = support.get_value(bb_ret[name])
      if val is None:
        field_type = type(bb_ret[name])
        raise TypeError(
            f"In {cls_name}, body_builder return doesn't support type "
            f"'{field_type}'")
      if val.type != port_type:
        if isinstance(port_type, hw.TypeAliasType) and \
           port_type.inner_type == val.type:
          val = hw.BitcastOp.create(port_type, val).result
        else:
          raise TypeError(
              f"In {cls_name}, output port '{name}' type ({val.type}) doesn't "
              f"match declared type ({port_type})")
      outputs.append(val)
      bb_ret.pop(name)
  if len(unconnected_ports) > 0:
    raise support.UnconnectedSignalError(cls_name, unconnected_ports)
  if len(bb_ret) > 0:
    raise support.ConnectionError(
        f"Could not map the following to output ports in {cls_name}: " +
        ",".join(bb_ret.keys()))

  hw.OutputOp(outputs)


class HWModuleOp(ModuleLike):
  """Specialization for the HW module op class."""

  def __init__(
      self,
      name,
      input_ports=[],
      output_ports=[],
      *,
      parameters=[],
      attributes={},
      body_builder=None,
      loc=None,
      ip=None,
  ):
    if "comment" not in attributes:
      attributes["comment"] = StringAttr.get("")
    super().__init__(name,
                     input_ports,
                     output_ports,
                     parameters=parameters,
                     attributes=attributes,
                     body_builder=body_builder,
                     loc=loc,
                     ip=ip)

  @property
  def body(self):
    return self.regions[0]

  @property
  def entry_block(self):
    return self.regions[0].blocks[0]

  @property
  def input_indices(self):
    indices: dict[int, str] = {}
    op_names = ArrayAttr(self.attributes["argNames"])
    for idx, name in enumerate(op_names):
      str_name = StringAttr(name).value
      indices[str_name] = idx
    return indices

  # Support attribute access to block arguments by name
  def __getattr__(self, name):
    if name in self.input_indices:
      index = self.input_indices[name]
      return self.entry_block.arguments[index]
    raise AttributeError(f"unknown input port name {name}")

  def inputs(self) -> dict[str:Value]:
    ret = {}
    for (name, idx) in self.input_indices.items():
      ret[name] = self.entry_block.arguments[idx]
    return ret

  def outputs(self) -> dict[str:Type]:
    result_names = [
        StringAttr(name).value
        for name in ArrayAttr(self.attributes["resultNames"])
    ]
    result_types = self.type.results
    return dict(zip(result_names, result_types))

  def add_entry_block(self):
    if not self.is_external:
      raise IndexError('The module already has an entry block')
    self.body.blocks.append(*self.type.inputs)
    return self.body.blocks[0]


class HWModuleExternOp(ModuleLike):
  """Specialization for the HW module op class."""

  def __init__(
      self,
      name,
      input_ports=[],
      output_ports=[],
      *,
      parameters=[],
      attributes={},
      body_builder=None,
      loc=None,
      ip=None,
  ):
    if "comment" not in attributes:
      attributes["comment"] = StringAttr.get("")
    super().__init__(name,
                     input_ports,
                     output_ports,
                     parameters=parameters,
                     attributes=attributes,
                     body_builder=body_builder,
                     loc=loc,
                     ip=ip)


class ConstantOp:

  @staticmethod
  def create(data_type, value):
    return hw.ConstantOp(IntegerAttr.get(data_type, value))


class BitcastOp:

  @staticmethod
  def create(data_type, value):
    value = support.get_value(value)
    return hw.BitcastOp(data_type, value)


class ArrayGetOp:

  @staticmethod
  def create(array_value, idx):
    array_value = support.get_value(array_value)
    array_type = support.get_self_or_inner(array_value.type)
    if isinstance(idx, int):
      idx_width = (array_type.size - 1).bit_length()
      idx_val = ConstantOp.create(IntegerType.get_signless(idx_width),
                                  idx).result
    else:
      idx_val = support.get_value(idx)
    return hw.ArrayGetOp(array_value, idx_val)


class ArraySliceOp:

  @staticmethod
  def create(array_value, low_index, ret_type):
    array_value = support.get_value(array_value)
    array_type = support.get_self_or_inner(array_value.type)
    if isinstance(low_index, int):
      idx_width = (array_type.size - 1).bit_length()
      idx_width = max(1, idx_width)  # hw.constant cannot produce i0.
      idx_val = ConstantOp.create(IntegerType.get_signless(idx_width),
                                  low_index).result
    else:
      idx_val = support.get_value(low_index)
    return hw.ArraySliceOp(ret_type, array_value, idx_val)


class ArrayCreateOp:

  @staticmethod
  def create(elements):
    if not elements:
      raise ValueError("Cannot 'create' an array of length zero")
    vals = []
    type = None
    for i, arg in enumerate(elements):
      arg_val = support.get_value(arg)
      vals.append(arg_val)
      if type is None:
        type = arg_val.type
      elif type != arg_val.type:
        raise TypeError(
            f"Argument {i} has a different element type ({arg_val.type}) than the element type of the array ({type})"
        )
    return hw.ArrayCreateOp(hw.ArrayType.get(type, len(vals)), vals)


class ArrayConcatOp:

  @staticmethod
  def create(*sub_arrays):
    vals = []
    types = []
    element_type = None
    for i, array in enumerate(sub_arrays):
      array_value = support.get_value(array)
      array_type = support.type_to_pytype(array_value.type)
      if array_value is None or not isinstance(array_type, hw.ArrayType):
        raise TypeError(f"Cannot concatenate {array_value}")
      if element_type is None:
        element_type = array_type.element_type
      elif element_type != array_type.element_type:
        raise TypeError(
            f"Argument {i} has a different element type ({element_type}) than the element type of the array ({array_type.element_type})"
        )

      vals.append(array_value)
      types.append(array_type)

    size = sum(t.size for t in types)
    combined_type = hw.ArrayType.get(element_type, size)
    return hw.ArrayConcatOp(combined_type, vals)


class StructCreateOp:

  @staticmethod
  def create(elements, result_type: Type = None):
    elem_name_values = [
        (name, support.get_value(value)) for (name, value) in elements
    ]
    struct_fields = [(name, value.type) for (name, value) in elem_name_values]
    struct_type = hw.StructType.get(struct_fields)

    if result_type is None:
      result_type = struct_type
    else:
      result_type_inner = support.get_self_or_inner(result_type)
      if result_type_inner != struct_type:
        raise TypeError(
            f"result type:\n\t{result_type_inner}\nmust match generated struct type:\n\t{struct_type}"
        )

    return hw.StructCreateOp(result_type,
                             [value for (_, value) in elem_name_values])


class StructExtractOp:

  @staticmethod
  def create(struct_value, field_name: str):
    struct_value = support.get_value(struct_value)
    struct_type = support.get_self_or_inner(struct_value.type)
    field_type = struct_type.get_field(field_name)
    return hw.StructExtractOp(field_type, struct_value,
                              StringAttr.get(field_name))


class TypedeclOp:

  @staticmethod
  def create(sym_name: str, type: Type, verilog_name: str = None):
    return hw.TypedeclOp(StringAttr.get(sym_name),
                         TypeAttr.get(type),
                         verilogName=verilog_name)


class TypeScopeOp:

  @staticmethod
  def create(sym_name: str):
    op = hw.TypeScopeOp(StringAttr.get(sym_name))
    op.regions[0].blocks.append()
    return op

  @property
  def body(self):
    return self.regions[0].blocks[0]
