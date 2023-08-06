#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import OrderedDict

from .support import get_user_loc

from .circt import ir, support
from .circt.dialects import esi, hw, sv
from .circt.dialects.esi import ChannelSignaling

import typing


class _Types:
  """Python syntactic sugar to get types"""

  def __init__(self):
    self.registered_aliases = OrderedDict()

  def __getattr__(self, name: str) -> ir.Type:
    return self.wrap(_FromCirctType(ir.Type.parse(name)))

  def int(self, width: int, name: str = None):
    return self.wrap(Bits(width), name)

  def array(self, inner: ir.Type, size: int, name: str = None) -> "Array":
    return self.wrap(Array(inner, size), name)

  def inout(self, inner: ir.Type):
    return self.wrap(InOut(inner))

  def channel(self, inner):
    return self.wrap(Channel(inner))

  def struct(self, members, name: str = None) -> "StructType":
    return self.wrap(StructType(members), name)

  @property
  def any(self):
    return self.wrap(Any())

  def wrap(self, type, name=None):
    if name is not None:
      type = TypeAlias(type, name)
    return type


types = _Types()


class Type:
  """PyCDE type hierarchy root class. Can wrap any MLIR/CIRCT type, but can only
  do anything useful with types for which subclasses exist."""

  # Global Type cache.
  _cache: typing.Dict[typing.Tuple[type, ir.Type], "Type"] = {}

  def __new__(cls, circt_type: ir.Type, incl_cls_in_key: bool = True) -> "Type":
    """Look up a type in the Type cache. If present, return it. If not, create
    it and put it in the cache."""
    assert isinstance(circt_type, ir.Type)
    if incl_cls_in_key:
      cache_key = (cls, circt_type)
    else:
      cache_key = circt_type

    if cache_key not in Type._cache:
      t = super(Type, cls).__new__(cls)
      t._type = circt_type
      Type._cache[cache_key] = t
    return Type._cache[cache_key]

  def __init__(self, *args, **kwargs) -> None:
    pass

  @property
  def strip(self):
    return self

  @property
  def bitwidth(self):
    return hw.get_bitwidth(self._type)

  def __call__(self, obj, name: str = None) -> "Signal":
    """Create a Value of this type from a python object."""
    assert not isinstance(obj, ir.Value)
    v = self._from_obj_or_sig(obj)
    if name is not None:
      v.name = name
    return v

  def _from_obj_or_sig(self,
                       obj,
                       alias: typing.Optional["TypeAlias"] = None) -> "Signal":
    """Implement the object-signal conversion wherein 'obj' can be a Signal. If
    'obj' is already a Signal, check its type and return it. Can be overriden by
    subclasses, though calls _from_obj() to do the type-specific const
    conversion so we recommend subclasses override that method."""

    from .signals import Signal
    if isinstance(obj, Signal):
      if obj.type != self:
        raise TypeError(f"Expected signal of type {self} but got {obj.type}")
      return obj
    return self._from_obj(obj, alias)

  def _from_obj(self,
                obj,
                alias: typing.Optional["TypeAlias"] = None) -> "Signal":
    """Do the type-specific object validity checks and return a Signal from the
    object. Can assume the 'obj' is NOT a Signal. Any subclass which wants to be
    created MUST override this method."""

    assert False, "Subclass must override this method"

  def _get_value_class(self):
    """Return the class which should be instantiated to create a Value."""
    from .signals import UntypedSignal
    return UntypedSignal

  def __mul__(self, len: int):
    """Create an array type"""
    return Array(self, len)

  def __repr__(self):
    return self._type.__repr__()


def _FromCirctType(type: typing.Union[ir.Type, Type]) -> Type:
  if isinstance(type, Type):
    return type
  type = support.type_to_pytype(type)
  if isinstance(type, hw.ArrayType):
    return Type.__new__(Array, type)
  if isinstance(type, hw.StructType):
    return Type.__new__(StructType, type)
  if isinstance(type, hw.TypeAliasType):
    return Type.__new__(TypeAlias, type, incl_cls_in_key=False)
  if isinstance(type, hw.InOutType):
    return Type.__new__(InOut, type)
  if isinstance(type, ir.IntegerType):
    if type.is_signed:
      return Type.__new__(SInt, type)
    elif type.is_unsigned:
      return Type.__new__(UInt, type)
    else:
      return Type.__new__(Bits, type)
  if isinstance(type, esi.AnyType):
    return Type.__new__(Any, type)
  if isinstance(type, esi.ChannelType):
    return Type.__new__(Channel, type)
  if isinstance(type, esi.ListType):
    return Type.__new__(List, type)
  return Type(type)


class InOut(Type):

  def __new__(cls, element_type: Type):
    return super(InOut, cls).__new__(cls, hw.InOutType.get(element_type._type))

  @property
  def element_type(self) -> Type:
    return _FromCirctType(self._type.element_type)

  def _get_value_class(self):
    from .signals import InOutSignal
    return InOutSignal

  def __repr__(self):
    return f"InOut<{repr(self.element_type)}"


class TypeAlias(Type):

  TYPE_SCOPE = "pycde"
  RegisteredAliases: typing.Optional[OrderedDict] = None

  def __new__(cls, inner_type: Type, name: str):
    if TypeAlias.RegisteredAliases is None:
      TypeAlias.RegisteredAliases = OrderedDict()

    if name in TypeAlias.RegisteredAliases:
      if inner_type._type != TypeAlias.RegisteredAliases[name].inner_type:
        raise RuntimeError(
            f"Re-defining type alias for {name}! "
            f"Given: {inner_type}, "
            f"existing: {TypeAlias.RegisteredAliases[name].inner_type}")
      alias = TypeAlias.RegisteredAliases[name]
    else:
      alias = hw.TypeAliasType.get(TypeAlias.TYPE_SCOPE, name, inner_type._type)
      TypeAlias.RegisteredAliases[name] = alias

    return super(TypeAlias, cls).__new__(cls, alias, incl_cls_in_key=False)

  @staticmethod
  def declare_aliases(mod):
    if TypeAlias.RegisteredAliases is None:
      return

    type_scopes = list()
    for op in mod.body.operations:
      if isinstance(op, hw.TypeScopeOp):
        type_scopes.append(op)
        continue
      if isinstance(op, sv.IfDefOp):
        if len(op.elseRegion.blocks) == 0:
          continue
        for ifdef_op in op.elseRegion.blocks[0]:
          if isinstance(ifdef_op, hw.TypeScopeOp):
            type_scopes.append(ifdef_op)

    assert len(type_scopes) <= 1
    if len(type_scopes) == 1:
      type_scope = type_scopes[0]
    else:
      with ir.InsertionPoint.at_block_begin(mod.body):
        guard_name = "__PYCDE_TYPES__"
        sv.VerbatimOp(ir.StringAttr.get("`ifndef " + guard_name), [],
                      symbols=ir.ArrayAttr.get([]))
        sv.VerbatimOp(ir.StringAttr.get("`define " + guard_name), [],
                      symbols=ir.ArrayAttr.get([]))
        type_scope = hw.TypeScopeOp.create(TypeAlias.TYPE_SCOPE)
        sv.VerbatimOp(ir.StringAttr.get("`endif // " + guard_name), [],
                      symbols=ir.ArrayAttr.get([]))

    with ir.InsertionPoint(type_scope.body):
      for (name, type) in TypeAlias.RegisteredAliases.items():
        declared_aliases = [
            op for op in type_scope.body.operations
            if isinstance(op, hw.TypedeclOp) and op.sym_name.value == name
        ]
        if len(declared_aliases) != 0:
          continue
        hw.TypedeclOp.create(name, type.inner_type)

  @property
  def name(self) -> str:
    return self._type.name

  @property
  def inner_type(self):
    return _FromCirctType(self._type.inner_type)

  def __repr__(self):
    return f"TypeAlias<'{self.name}', {repr(self.inner_type)}"

  def __str__(self):
    return self.name

  @property
  def strip(self):
    return _FromCirctType(self._type.inner_type)

  def _get_value_class(self):
    return self.strip._get_value_class()

  def wrap(self, value):
    return self(value)

  def _from_obj(self, obj, alias: typing.Optional["TypeAlias"] = None):
    return self.inner_type._from_obj_or_sig(obj, alias=self)


class Array(Type):

  def __new__(cls, element_type: Type, length: int):
    return super(Array,
                 cls).__new__(cls, hw.ArrayType.get(element_type._type, length))

  @property
  def inner_type(self):
    if isinstance(self.element_type, Array):
      return self.element_type.inner_type
    return self.element_type

  @property
  def element_type(self):
    return _FromCirctType(self._type.element_type)

  @property
  def size(self):
    return self._type.size

  @property
  def shape(self):
    _shape = [self.size]
    if isinstance(self.element_type, Array):
      _shape.extend(self.element_type.shape)
    return _shape

  def __len__(self):
    return self.size

  def _get_value_class(self):
    from .signals import ArraySignal
    return ArraySignal

  def __repr__(self) -> str:
    return f"Array({self.size}, {self.element_type})"

  def __str__(self) -> str:
    return f"{self.element_type}[{self.size}]"

  def _from_obj(self, obj, alias: typing.Optional[TypeAlias] = None):
    from .dialects import hw
    if not isinstance(obj, (list, tuple)):
      raise ValueError(
          f"Arrays can only be created from lists or tuples, not '{type(obj)}'")
    if len(obj) != self.size:
      raise ValueError("List must have same size as array "
                       f"{len(obj)} vs {self.size}")
    elemty = self.element_type
    list_of_vals = list(map(lambda x: elemty._from_obj_or_sig(x), obj))
    with get_user_loc():
      # CIRCT's ArrayCreate op takes the array in reverse order.
      return hw.ArrayCreateOp(reversed(list_of_vals))


class StructType(Type):

  def __new__(cls, fields: typing.Union[typing.List[typing.Tuple[str, Type]],
                                        typing.Dict[str, Type]]):
    if isinstance(fields, dict):
      fields = list(fields.items())
    if not isinstance(fields, list):
      raise TypeError("Expected either list or dict.")
    return super(StructType, cls).__new__(
        cls, hw.StructType.get([(n, t._type) for (n, t) in fields]))

  @property
  def fields(self):
    return [(n, _FromCirctType(t)) for n, t in self._type.get_fields()]

  def __getattr__(self, attrname: str):
    for field in self.fields:
      if field[0] == attrname:
        return _FromCirctType(self._type.get_field(attrname))
    return super().__getattribute__(attrname)

  def _get_value_class(self):
    from .signals import StructSignal
    return StructSignal

  def _from_obj(self, x, alias: typing.Optional[TypeAlias] = None):
    from .dialects import hw
    if not isinstance(x, dict):
      raise ValueError(
          f"Structs can only be created from dicts, not '{type(x)}'")
    elem_name_values = []
    for (fname, ftype) in self.fields:
      if fname not in x:
        raise ValueError(f"Could not find expected field: {fname}")
      v = ftype._from_obj_or_sig(x[fname])
      elem_name_values.append((fname, v))
      x.pop(fname)
    if len(x) > 0:
      raise ValueError(f"Extra fields specified: {x}")

    result_type = self if alias is None else alias
    with get_user_loc():
      return hw.StructCreateOp(elem_name_values, result_type=result_type._type)

  def __repr__(self) -> str:
    ret = "struct { "
    first = True
    for field in self.fields:
      if first:
        first = False
      else:
        ret += ", "
      ret += f"{field[0]}: {field[1]}"
    ret += "}"
    return ret


class RegisteredStruct(TypeAlias):
  """Represents a named struct with a custom signal class. Primarily used by
  `value.Struct`."""

  def __new__(cls, fields: typing.List[typing.Tuple[str, Type]], name: str,
              value_class):
    inner_type = StructType(fields)
    inst = super().__new__(cls, inner_type, name)
    inst._value_class = value_class
    return inst

  def __call__(self, **kwargs):
    return self._from_obj_or_sig(kwargs)

  def _get_value_class(self):
    return self._value_class


class BitVectorType(Type):

  @property
  def width(self):
    return self._type.width

  def _from_obj_check(self, x):
    """This functionality can be shared by all the int types."""
    if not isinstance(x, int):
      raise ValueError(f"{type(self).__name__} can only be created from ints, "
                       f"not {type(x).__name__}")
    signed_bit = 1 if isinstance(self, SInt) else 0
    if x.bit_length() + signed_bit > self.width:
      raise ValueError(f"{x} overflows type {self}")

  def __repr__(self) -> str:
    return f"{type(self).__name__}<{self.width}>"


class Bits(BitVectorType):

  def __new__(cls, width: int):
    return super(Bits, cls).__new__(
        cls,
        ir.IntegerType.get_signless(width),
    )

  def _get_value_class(self):
    from .signals import BitsSignal
    return BitsSignal

  def _from_obj(self, x: int, alias: typing.Optional[TypeAlias] = None):
    from .dialects import hw
    self._from_obj_check(x)
    circt_type = self if alias is None else alias
    return hw.ConstantOp(circt_type, x)


# A single bit is common enough to provide an alias to save 4 key strokes.
bit = Bits(1)


class SInt(BitVectorType):

  def __new__(cls, width: int):
    return super(SInt, cls).__new__(
        cls,
        ir.IntegerType.get_signed(width),
    )

  def _get_value_class(self):
    from .signals import SIntSignal
    return SIntSignal

  def _from_obj(self, x: int, alias: typing.Optional[TypeAlias] = None):
    from .dialects import hwarith
    self._from_obj_check(x)
    circt_type = self if alias is None else alias
    return hwarith.ConstantOp(circt_type, x)


class UInt(BitVectorType):

  def __new__(cls, width: int):
    return super(UInt, cls).__new__(
        cls,
        ir.IntegerType.get_unsigned(width),
    )

  def _get_value_class(self):
    from .signals import UIntSignal
    return UIntSignal

  def _from_obj(self, x: int, alias: typing.Optional[TypeAlias] = None):
    from .dialects import hwarith
    self._from_obj_check(x)
    if x < 0:
      raise ValueError(f"UInt can only store positive numbers, not {x}")
    circt_type = self if alias is None else alias
    return hwarith.ConstantOp(circt_type, x)


class ClockType(Bits):
  """A special single bit to represent a clock. Can't do any special operations
  on it, except enter it as a implicit clock block."""

  # TODO: the 'clock' type isn't represented in CIRCT IR. It may be useful to
  # have it there if for no other reason than being able to round trip this
  # type.

  def __new__(cls):
    return super(ClockType, cls).__new__(cls, 1)

  def _get_value_class(self):
    from .signals import ClockSignal
    return ClockSignal

  def __repr__(self):
    return "Clk"


class Any(Type):

  def __new__(cls):
    return super(Any, cls).__new__(cls, esi.AnyType.get())


class Channel(Type):
  """An ESI channel type."""

  SignalingNames = {
      ChannelSignaling.ValidReady: "ValidReady",
      ChannelSignaling.FIFO0: "FIFO0"
  }

  def __new__(cls,
              inner_type: Type,
              signaling: int = ChannelSignaling.ValidReady):
    return super(Channel,
                 cls).__new__(cls,
                              esi.ChannelType.get(inner_type._type, signaling))

  @property
  def inner_type(self):
    return _FromCirctType(self._type.inner)

  @property
  def signaling(self):
    return self._type.signaling

  def _get_value_class(self):
    from .signals import ChannelSignal
    return ChannelSignal

  def __repr__(self):
    signaling = Channel.SignalingNames[self.signaling]
    return f"Channel<{self.inner_type}, {signaling}>"

  @property
  def inner(self):
    return self.inner_type

  def wrap(self, value,
           valueOrEmpty) -> typing.Tuple["ChannelSignal", "BitsSignal"]:
    """Wrap a data signal and valid signal into a data channel signal and a
    ready signal."""

    # Instead of implementing __call__(), we require users to call this method
    # instead. In addition to being clearer, the type signature isn't the same
    # -- this returns a tuple of Signals (data, ready) -- rather than a single
    # one.

    from .dialects import esi
    signaling = self.signaling
    if signaling == ChannelSignaling.ValidReady:
      value = self.inner_type(value)
      valid = types.i1(valueOrEmpty)
      wrap_op = esi.WrapValidReadyOp(self._type, types.i1, value.value,
                                     valid.value)
      return wrap_op[0], wrap_op[1]
    elif signaling == ChannelSignaling.FIFO0:
      value = self.inner_type(value)
      empty = types.i1(valueOrEmpty)
      wrap_op = esi.WrapFIFOOp(self._type, types.i1, value.value, empty.value)
      return wrap_op[0], wrap_op[1]
    else:
      raise TypeError("Unknown signaling standard")


class List(Type):
  """An ESI list type represents variable length data. Just like a Python list."""

  def __new__(cls, element_type: Type):
    return super(List, cls).__new__(cls, esi.ListType.get(element_type._type))

  @property
  def element_type(self):
    return _FromCirctType(self._type.element_type)

  @property
  def _get_value_class(self):
    from .signals import ListSignal
    return ListSignal

  def __repr__(self):
    return f"List<{self.element_type}>"

  @property
  def inner(self):
    return self.inner_type


def dim(inner_type_or_bitwidth: typing.Union[Type, int],
        *size: typing.List[int],
        name: str = None) -> Array:
  """Creates a multidimensional array from innermost to outermost dimension."""
  if isinstance(inner_type_or_bitwidth, int):
    ret = _FromCirctType(ir.IntegerType.get_signless(inner_type_or_bitwidth))
  elif isinstance(inner_type_or_bitwidth, Type):
    ret = inner_type_or_bitwidth
  else:
    raise ValueError(f"Expected 'Type', not {inner_type_or_bitwidth}")
  for s in size:
    ret = Array(ret, s)
  if name is None:
    return ret
  return TypeAlias(ret, name)
