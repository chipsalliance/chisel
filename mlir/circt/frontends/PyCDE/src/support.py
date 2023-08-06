from .circt import support
from .circt import ir

import os


# PyCDE needs a custom version of this to support python classes.
def _obj_to_attribute(obj) -> ir.Attribute:
  """Create an MLIR attribute from a Python object for a few common cases."""
  if obj is None:
    return ir.BoolAttr.get(False)
  if isinstance(obj, ir.Attribute):
    return obj
  if isinstance(obj, ir.Type):
    return ir.TypeAttr.get(obj)
  if isinstance(obj, bool):
    return ir.BoolAttr.get(obj)
  if isinstance(obj, int):
    attrTy = ir.IntegerType.get_signless(64)
    return ir.IntegerAttr.get(attrTy, obj)
  if isinstance(obj, str):
    return ir.StringAttr.get(obj)
  if isinstance(obj, list) or isinstance(obj, tuple):
    arr = [_obj_to_attribute(x) for x in obj]
    if all(arr):
      return ir.ArrayAttr.get(arr)
  if isinstance(obj, dict):
    attrs = {name: _obj_to_attribute(value) for name, value in obj.items()}
    return ir.DictAttr.get(attrs)
  if hasattr(obj, "__dict__"):
    attrs = {
        name: _obj_to_attribute(value) for name, value in obj.__dict__.items()
    }
    return ir.DictAttr.get(attrs)
  raise TypeError(f"Cannot convert type '{type(obj)}' to MLIR attribute. "
                  "This is required for parameters.")


__dir__ = os.path.dirname(__file__)
_local_files = set([os.path.join(__dir__, x) for x in os.listdir(__dir__)])
_hidden_filenames = set(["functools.py"])


def get_user_loc() -> ir.Location:
  import traceback
  stack = reversed(traceback.extract_stack())
  for frame in stack:
    fn = os.path.split(frame.filename)[1]
    if frame.filename in _local_files or fn in _hidden_filenames:
      continue
    return ir.Location.file(frame.filename, frame.lineno, 0)
  return ir.Location.unknown()


def create_const_zero(type):
  """Create a 'default' constant value of zero. Used for creating dummy values
  to connect to extern modules with input ports we want to ignore."""
  from .dialects import hw
  width = hw.get_bitwidth(type._type)

  with get_user_loc():
    zero = hw.ConstantOp(ir.IntegerType.get_signless(width), 0)
    return hw.BitcastOp(type, zero)


def _infer_type(x):
  """Infer the CIRCT type from a python object. Only works on lists."""
  from .types import Array
  from .signals import Signal
  if isinstance(x, Signal):
    return x.type

  if isinstance(x, (list, tuple)):
    list_types = [_infer_type(i) for i in x]
    list_type = list_types[0]
    if not all([i == list_type for i in list_types]):
      raise ValueError("CIRCT array must be homogenous, unlike object")
    return Array(list_type, len(x))
  if isinstance(x, int):
    raise ValueError(f"Cannot infer width of {x}")
  if isinstance(x, dict):
    raise ValueError(f"Cannot infer struct field order of {x}")
  return None


def _obj_to_value_infer_type(value) -> ir.Value:
  """Infer the CIRCT type, then convert the Python object to a CIRCT Value of
  that type."""
  cde_type = _infer_type(value)
  if cde_type is None:
    raise ValueError(f"Cannot infer CIRCT type from '{value}")
  return cde_type(value)


def create_type_string(ty):
  from .dialects import hw
  ty = support.type_to_pytype(ty)
  if isinstance(ty, hw.TypeAliasType):
    return ty.name
  if isinstance(ty, hw.ArrayType):
    return f"{ty.size}x" + create_type_string(ty.element_type)
  return str(ty)


def attributes_of_type(o, T):
  """Filter the attributes of an object 'o' to only those of type 'T'."""
  return {a: getattr(o, a) for a in dir(o) if isinstance(getattr(o, a), T)}
