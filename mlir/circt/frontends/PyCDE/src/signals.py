#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .support import get_user_loc, _obj_to_value_infer_type
from .types import ChannelSignaling, Type

from .circt.dialects import sv
from .circt import support
from .circt import ir

from contextvars import ContextVar
from functools import singledispatchmethod
from typing import List, Optional, Tuple, Union
import re
import numpy as np


def _FromCirctValue(value: ir.Value, type: Type = None) -> Signal:
  from .types import _FromCirctType
  assert isinstance(value, ir.Value)
  if type is None:
    type = _FromCirctType(value.type)
  return type._get_value_class()(value, type)


class Signal:
  """Root of the PyCDE value (signal, in RTL terms) hierarchy."""

  def __init__(self, value: Union[Signal, ir.Value], type: Type):
    assert value is not None
    assert type is not None

    self.type = type
    if isinstance(value, ir.Value):
      self.value = value
    elif isinstance(value, Signal):
      self.value = value.value
    else:
      assert False, "'value' must be either ir.Value or Signal"

  _reg_name = re.compile(r"^(.*)__reg(\d+)$")

  @staticmethod
  def create(obj) -> Signal:
    """Create a Signal from any python object from which the hardware type can
    be inferred. For instance, a list of Signals is inferred as an `Array` of
    those signal types, assuming the types of all the `Signal` are the same."""
    return _obj_to_value_infer_type(obj)

  def reg(self,
          clk=None,
          rst=None,
          rst_value=None,
          ce=None,
          name=None,
          cycles=1,
          sv_attributes=None,
          appid=None):
    """Register this value, returning the delayed value.
    `clk`, `rst`: the clock and reset signals.
    `name`: name this register explicitly.
    `cycles`: number of registers to add."""

    if clk is None:
      clk = ClockSignal._get_current_clock_block()
      if clk is None:
        raise ValueError("If 'clk' not specified, must be in clock block")

    from .dialects import seq, hw
    from .types import types, Bits
    if name is None:
      basename = None
      if self.name is not None:
        m = Signal._reg_name.match(self.name)
        if m:
          basename = m.group(1)
          reg_num = m.group(2)
          if reg_num.isdigit():
            starting_reg = int(reg_num) + 1
          else:
            basename = self.name
        else:
          basename = self.name
          starting_reg = 1
    with get_user_loc():
      # If rst without reset value, provide a default '0'.
      if rst_value is None and rst is not None:
        rst_value = types.int(self.type.bitwidth)(0)
        if not isinstance(self.type, Bits):
          rst_value = hw.BitcastOp(self.type, rst_value)

      reg = self.value
      for i in range(cycles):
        give_name = name
        if give_name is None and basename is not None:
          give_name = f"{basename}__reg{i+starting_reg}"
        if ce is None:
          reg = seq.CompRegOp(self.value.type,
                              input=reg,
                              clk=clk,
                              reset=rst,
                              reset_value=rst_value,
                              name=give_name,
                              sym_name=give_name)
        else:
          reg = seq.CompRegClockEnabledOp(self.value.type,
                                          input=reg,
                                          clk=clk,
                                          clockEnable=ce,
                                          reset=rst,
                                          reset_value=rst_value,
                                          clock_enable=ce,
                                          name=give_name,
                                          sym_name=give_name)
      if sv_attributes is not None:
        reg.value.owner.attributes["sv.attributes"] = ir.ArrayAttr.get(
            [sv.SVAttributeAttr.get(attr) for attr in sv_attributes])
      if appid is not None:
        reg.appid = appid
      return reg

  @property
  def _namehint_attrname(self):
    if self.value.owner.name == "seq.compreg":
      return "name"
    return "sv.namehint"

  @property
  def name(self):
    owner = self.value.owner
    if hasattr(owner,
               "attributes") and self._namehint_attrname in owner.attributes:
      return ir.StringAttr(owner.attributes[self._namehint_attrname]).value
    from .circt.dialects import msft
    if isinstance(owner, ir.Block) and isinstance(owner.owner,
                                                  msft.MSFTModuleOp):
      block_arg = ir.BlockArgument(self.value)
      mod = owner.owner
      return ir.StringAttr(
          ir.ArrayAttr(mod.attributes["argNames"])[block_arg.arg_number]).value
    if hasattr(self, "_name"):
      return self._name

  @name.setter
  def name(self, new: str):
    owner = self.value.owner
    if hasattr(owner, "attributes"):
      owner.attributes[self._namehint_attrname] = ir.StringAttr.get(new)
    else:
      self._name = new

  @property
  def appid(self) -> Optional[object]:  # Optional AppID.
    from .module import AppID
    owner = self.value.owner
    if AppID.AttributeName in owner.attributes:
      return AppID(owner.attributes[AppID.AttributeName])
    return None

  @appid.setter
  def appid(self, appid) -> None:
    if "inner_sym" not in self.value.owner.attributes:
      raise ValueError("AppIDs can only be attached to ops with symbols")
    from .module import AppID
    self.value.owner.attributes[AppID.AttributeName] = appid._appid


class UntypedSignal(Signal):
  pass


_current_clock_context = ContextVar("current_clock_context")


class ClockSignal(Signal):
  """A clock signal."""

  __slots__ = ["_old_token"]

  def __enter__(self):
    self._old_token = _current_clock_context.set(self)

  def __exit__(self, exc_type, exc_value, traceback):
    if exc_value is not None:
      return
    _current_clock_context.reset(self._old_token)

  @staticmethod
  def _get_current_clock_block():
    return _current_clock_context.get(None)


class InOutSignal(Signal):
  # Maintain a caching of the read value.
  read_value = None

  @property
  def read(self):
    if self.read_value is None:
      self.read_value = _FromCirctValue(sv.ReadInOutOp.create(self).results[0])
    return self.read_value


def _validate_idx(size: int, idx: Union[int, BitVectorSignal]):
  """Validate that `idx` is a valid index into a bitvector or array."""
  if isinstance(idx, int):
    if idx >= size:
      raise ValueError("Subscript out-of-bounds")
  else:
    idx = support.get_value(idx)
    if idx is None or not isinstance(support.type_to_pytype(idx.type),
                                     ir.IntegerType):
      raise TypeError("Subscript on array must be either int or int signal"
                      f" not {type(idx)}.")


def get_slice_bounds(size, idxOrSlice: Union[int, slice]):
  if isinstance(idxOrSlice, int):
    s = slice(idxOrSlice, idxOrSlice + 1)
  elif isinstance(idxOrSlice, slice):
    if idxOrSlice.stop and idxOrSlice.stop > size:
      raise ValueError("Slice out-of-bounds")
    s = idxOrSlice
  else:
    raise TypeError("Expected int or slice")

  idxs = s.indices(size)
  if idxs[2] != 1:
    raise ValueError("Integer / bitvector slices do not support steps")
  return idxs[0], idxs[1]


class BitVectorSignal(Signal):

  def __len__(self):
    return self.type.width

  #  === Casting ===

  def _exec_cast(self, targetValueType, type_getter, width: int = None):

    from .dialects import hwarith
    if width is None:
      width = self.type.width

    if isinstance(self, targetValueType) and width == self.type.width:
      return self
    return hwarith.CastOp(self.value, type_getter(width))

  def as_bits(self, width: int = None):
    """
    Returns this value as a signless integer. If 'width' is provided, this value
    will be truncated to that width.
    """
    return self._exec_cast(BitsSignal, ir.IntegerType.get_signless, width)

  def as_sint(self, width: int = None):
    """
    Returns this value as a a signed integer. If 'width' is provided, this value
    will be truncated or sign-extended to that width.
    """
    return self._exec_cast(SIntSignal, ir.IntegerType.get_signed, width)

  def as_uint(self, width: int = None):
    """
    Returns this value as an unsigned integer. If 'width' is provided, this
    value will be truncated or zero-padded to that width.
    """
    return self._exec_cast(UIntSignal, ir.IntegerType.get_unsigned, width)


def And(*items: List[BitVectorSignal]):
  """Compute a bitwise 'and' of the arguments."""
  from .dialects import comb
  return comb.AndOp(*items)


def Or(*items: List[BitVectorSignal]):
  """Compute a bitwise 'or' of the arguments."""
  from .dialects import comb
  return comb.OrOp(*items)


class BitsSignal(BitVectorSignal):
  """Operations on signless ints (bits). These will all return signless values -
  a user is expected to reapply signedness semantics if needed."""

  @singledispatchmethod
  def __getitem__(self, idxOrSlice: Union[int, slice]) -> BitVectorSignal:
    lo, hi = get_slice_bounds(len(self), idxOrSlice)
    from .types import types
    from .dialects import comb
    ret_type = types.int(hi - lo)

    with get_user_loc():
      ret = comb.ExtractOp(lo, ret_type, self.value)
      if self.name is not None:
        ret.name = f"{self.name}_{lo}upto{hi}"
      return ret

  @__getitem__.register(Signal)
  def __get_item__value(self, idx: BitVectorSignal) -> BitVectorSignal:
    """Get the single bit at `idx`."""
    return self.slice(idx, 1)

  @staticmethod
  def concat(items: List[BitVectorSignal]):
    """Concatenate a list of bitvectors into one larger bitvector."""
    from .dialects import comb
    return comb.ConcatOp(*items)

  def slice(self, low_bit: BitVectorSignal, num_bits: int):
    """Get a constant-width slice starting at `low_bit` and ending at `low_bit +
    num_bits`."""
    _validate_idx(self.type.width, low_bit)

    from .dialects import comb
    # comb.extract only supports constant lowBits. Shift the bits right, then
    # extract the correct number from the 0th bit.
    with get_user_loc():
      # comb.shru's rhs and lhs must be the same width.
      low_bit = low_bit.pad_or_truncate(self.type.width)
      shifted = comb.ShrUOp(self.value, low_bit)
      ret = comb.ExtractOp(0, ir.IntegerType.get_signless(num_bits), shifted)
      return ret

  def pad_or_truncate(self, num_bits: int):
    """Make value exactly `num_bits` width by either adding zeros to or lopping
    off the MSB."""
    pad_width = num_bits - self.type.width

    from .dialects import comb, hw
    if pad_width < 0:
      return comb.ExtractOp(0, ir.IntegerType.get_signless(num_bits),
                            self.value)
    if pad_width == 0:
      return self
    pad = hw.ConstantOp(ir.IntegerType.get_signless(pad_width), 0)
    v: Signal = comb.ConcatOp(pad.value, self.value)
    if self.name is not None:
      v.name = f"{self.name}_padto_{num_bits}"
    return v

  def and_reduce(self):
    from .types import types
    bits = [self[i] for i in range(len(self))]
    assert bits[0].type == types.i1
    return And(*bits)

  def or_reduce(self):
    from .types import types
    bits = [self[i] for i in range(len(self))]
    assert bits[0].type == types.i1
    return Or(*bits)

  # === Infix operators ===

  def __exec_signless_binop_nocast__(self, other, op, op_symbol: str,
                                     op_name: str):
    from .dialects import comb
    if not isinstance(other, Signal):
      # Fall back to the default implementation in cases where we're not dealing
      # with PyCDE value comparison.
      if op == comb.EqOp:
        return super().__eq__(other)
      elif op == comb.NeOp:
        return super().__ne__(other)

    if not isinstance(other, BitsSignal):
      raise TypeError(
          f"Operator '{op_symbol}' requires RHS to be cast .as_bits().")
    if self.type.width != other.type.width:
      raise TypeError(
          f"Operator '{op_symbol}' requires both operands to be the same width."
      )

    ret = op(self, other)
    if self.name is not None and other.name is not None:
      ret.name = f"{self.name}_{op_name}_{other.name}"
    return ret

  def __eq__(self, other):
    from .dialects import comb
    return self.__exec_signless_binop_nocast__(other, comb.EqOp, "==", "eq")

  def __ne__(self, other):
    from .dialects import comb
    return self.__exec_signless_binop_nocast__(other, comb.NeOp, "!=", "neq")

  def __and__(self, other):
    from .dialects import comb
    return self.__exec_signless_binop_nocast__(other, comb.AndOp, "&", "and")

  def __or__(self, other):
    from .dialects import comb
    return self.__exec_signless_binop_nocast__(other, comb.OrOp, "|", "or")

  def __xor__(self, other):
    from .dialects import comb
    return self.__exec_signless_binop_nocast__(other, comb.XorOp, "^", "xor")

  def __invert__(self):
    from .types import types
    ret = self ^ types.int(self.type.width)(-1)
    if self.name is not None:
      ret.name = f"inv_{self.name}"
    return ret


class IntSignal(BitVectorSignal):

  #  === Infix operators ===

  # Generalized function for executing sign-aware binary operations. Performs
  # a check to ensure that the operands have signedness semantics, and then
  # calls the provided operator.
  def __exec_signedness_binop__(self, other, op, op_symbol: str, op_name: str):
    from .dialects import hwarith

    # If a python int, create a minimum-width constant.
    if isinstance(other, int):
      if other < 0:
        const_type = ir.IntegerType.get_signed(other.bit_length() + 1)
      else:
        const_type = ir.IntegerType.get_unsigned(other.bit_length())
      other = hwarith.ConstantOp(const_type, other)

    if not isinstance(other, IntSignal):
      raise TypeError(
          f"Operator '{op_symbol}' is not supported on non-int or signless "
          "signals. RHS operand should be cast .as_sint()/.as_uint() if "
          "possible.")

    ret = op(self, other)
    if self.name is not None and other.name is not None:
      ret.name = f"{self.name}_{op_name}_{other.name}"
    return ret

  def __add__(self, other):
    from .dialects import hwarith
    return self.__exec_signedness_binop__(other, hwarith.AddOp, "+", "plus")

  def __sub__(self, other):
    from .dialects import hwarith
    return self.__exec_signedness_binop__(other, hwarith.SubOp, "-", "minus")

  def __mul__(self, other):
    from .dialects import hwarith
    return self.__exec_signedness_binop__(other, hwarith.MulOp, "*", "mul")

  def __truediv__(self, other):
    from .dialects import hwarith
    return self.__exec_signedness_binop__(other, hwarith.DivOp, "/", "div")

  # Generalized function for executing sign-aware int comparisons.
  def __exec_icmp__(self, other, pred: int, op_name: str):
    from .dialects import hwarith

    # If a python int, create a minimum-width constant.
    if isinstance(other, int):
      if other < 0:
        const_type = ir.IntegerType.get_signed(other.bit_length() + 1)
      else:
        const_type = ir.IntegerType.get_unsigned(other.bit_length())
      other = hwarith.ConstantOp(const_type, other)

    if not isinstance(other, IntSignal):
      raise TypeError(
          f"Comparisons of signed/unsigned integers to {other.type} not "
          "supported. RHS operand should be cast .as_sint()/.as_uint() if "
          "possible.")

    ret = hwarith.ICmpOp(pred, self, other)
    if self.name is not None and other.name is not None:
      ret.name = f"{self.name}_{op_name}_{other.name}"
    return ret

  def __eq__(self, other):
    from .circt.dialects import hwarith
    return self.__exec_icmp__(other, hwarith.ICmpOp.PRED_EQ, "eq")

  def __ne__(self, other):
    from .circt.dialects import hwarith
    return self.__exec_icmp__(other, hwarith.ICmpOp.PRED_NE, "neq")

  def __lt__(self, other):
    from .circt.dialects import hwarith
    return self.__exec_icmp__(other, hwarith.ICmpOp.PRED_LT, "lt")

  def __gt__(self, other):
    from .circt.dialects import hwarith
    return self.__exec_icmp__(other, hwarith.ICmpOp.PRED_GT, "gt")

  def __le__(self, other):
    from .circt.dialects import hwarith
    return self.__exec_icmp__(other, hwarith.ICmpOp.PRED_LE, "le")

  def __ge__(self, other):
    from .circt.dialects import hwarith
    return self.__exec_icmp__(other, hwarith.ICmpOp.PRED_GE, "ge")


class UIntSignal(IntSignal):
  pass


class SIntSignal(IntSignal):

  def __neg__(self):
    from .types import types
    return self * types.int(self.type.width)(-1).as_sint()


class ArraySignal(Signal):

  @singledispatchmethod
  def __getitem__(self, idx: Union[int, BitVectorSignal]) -> Signal:
    _validate_idx(self.type.size, idx)
    from .dialects import hw
    with get_user_loc():
      v = hw.ArrayGetOp(self.value, idx)
      if self.name and isinstance(idx, int):
        v.name = self.name + f"__{idx}"
      return v

  @__getitem__.register(slice)
  def __get_item__slice(self, s: slice):
    idxs = s.indices(len(self))
    if idxs[2] != 1:
      raise ValueError("Array slices do not support steps")

    from .types import types
    from .dialects import hw
    ret_type = types.array(self.type.element_type, idxs[1] - idxs[0])

    with get_user_loc():
      ret = hw.ArraySliceOp(self.value, idxs[0], ret_type)
      if self.name is not None:
        ret.name = f"{self.name}_{idxs[0]}upto{idxs[1]}"
      return ret

  def slice(self, low_idx: Union[int, BitVectorSignal],
            num_elems: int) -> Signal:
    """Get an array slice starting at `low_idx` and ending at `low_idx +
    num_elems`."""
    _validate_idx(self.type.size, low_idx)
    if num_elems > self.type.size:
      raise ValueError(
          f"num_bits ({num_elems}) must be <= value width ({len(self)})")
    if isinstance(low_idx, BitVectorSignal):
      low_idx = low_idx.pad_or_truncate(self.type.size.bit_length())

    from .dialects import hw
    from .types import Array
    with get_user_loc():
      v = hw.ArraySliceOp(self.value, low_idx,
                          Array(self.type.element_type, num_elems))
      if self.name and isinstance(low_idx, int):
        v.name = self.name + f"__{low_idx}upto{low_idx+num_elems}"
      return v

  def and_reduce(self):
    from .types import types
    bits = [self[i] for i in range(len(self))]
    assert bits[0].type == types.i1
    return And(*bits)

  def or_reduce(self):
    from .types import types
    bits = [self[i] for i in range(len(self))]
    assert bits[0].type == types.i1
    return Or(*bits)

  def __len__(self):
    return self.type.strip.size

  """
  Add a curated set of Numpy functions through the Matrix class. This allows
  for directly manipulating the ArraySignals with numpy functionality.
  Power-users who use the Matrix directly have access to all numpy functions.
  In reality, it will only be a subset of the numpy array functions which are
  safe to be used in the PyCDE context. Curating access at the level of
  ArraySignals seems like a safe starting point.
  """

  def transpose(self, *args, **kwargs):
    from .ndarray import NDArray
    return NDArray(from_value=self).transpose(*args, **kwargs).to_circt()

  def reshape(self, *args, **kwargs):
    from .ndarray import NDArray
    return NDArray(from_value=self).reshape(*args, **kwargs).to_circt()

  def flatten(self, *args, **kwargs):
    from .ndarray import NDArray
    return NDArray(from_value=self).flatten(*args, **kwargs).to_circt()

  def moveaxis(self, *args, **kwargs):
    from .ndarray import NDArray
    return NDArray(from_value=self).moveaxis(*args, **kwargs).to_circt()

  def rollaxis(self, *args, **kwargs):
    from .ndarray import NDArray
    return NDArray(from_value=self).rollaxis(*args, **kwargs).to_circt()

  def swapaxes(self, *args, **kwargs):
    from .ndarray import NDArray
    return NDArray(from_value=self).swapaxes(*args, **kwargs).to_circt()

  def concatenate(self, arrays, axis=0):
    from .ndarray import NDArray
    return NDArray(from_value=np.concatenate(
        NDArray.to_ndarrays([self] + list(arrays)), axis=axis)).to_circt()

  def roll(self, shift, axis=None):
    from .ndarray import NDArray
    return np.roll(NDArray(from_value=self), shift=shift, axis=axis).to_circt()


class StructSignal(Signal):

  def __getitem__(self, sub):
    if sub not in [name for name, _ in self.type.strip.fields]:
      raise ValueError(f"Struct field '{sub}' not found in {self.type}")
    from .dialects import hw
    with get_user_loc():
      return hw.StructExtractOp(self.value, sub)

  def __getattr__(self, attr):
    ty = self.type.strip
    if attr in [name for name, _ in ty.fields]:
      from .dialects import hw
      with get_user_loc():
        v = hw.StructExtractOp(self.value, attr)
        if self.name:
          v.name = f"{self.name}__{attr}"
        return v
    raise AttributeError(f"{type(self)} object has no attribute '{attr}'")


class StructMetaType(type):

  def __new__(self, name, bases, dct):
    """Scans the class being created for type hints, creates a CIRCT struct
    object and returns the CIRCT struct object instead of the class. Use the
    class when a `Signal` of the struct type is instantiated."""

    cls = super().__new__(self, name, bases, dct)
    from .types import RegisteredStruct, Type
    if "__annotations__" not in dct:
      return cls
    fields: List[Tuple[str, Type]] = []
    for attr_name, attr in dct["__annotations__"].items():
      if isinstance(attr, Type):
        fields.append((attr_name, attr))

    return RegisteredStruct(fields, name, cls)


class Struct(StructSignal, metaclass=StructMetaType):
  """Subclassing this class creates a hardware struct which can be used in port
  definitions and will be instantiated in generators:

  ```
  class ExStruct(Struct):
    a: Bits(4)
    b: UInt(32)

    def get_b(self):
      return self.b

  class TestStruct(Module):
    inp1 = Input(ExStruct)

    @generator
    def build(self):
      ... = self.inp1.get_b()
  ```
  """
  # All the work is done in the metaclass.


class ChannelSignal(Signal):

  def reg(self, clk, rst=None, name=None):
    raise TypeError("Cannot register a channel")

  def unwrap(self, readyOrRden):
    from .dialects import esi
    from .types import types
    signaling = self.type.signaling
    if signaling == ChannelSignaling.ValidReady:
      ready = types.i1(readyOrRden)
      unwrap_op = esi.UnwrapValidReadyOp(self.type.inner_type, types.i1,
                                         self.value, ready.value)
      return unwrap_op[0], unwrap_op[1]
    elif signaling == ChannelSignaling.FIFO0:
      rden = types.i1(readyOrRden)
      wrap_op = esi.UnwrapFIFOOp(self.value, rden.value)
      return wrap_op[0], wrap_op[1]
    else:
      raise TypeError("Unknown signaling standard")


class ListSignal(Signal):
  pass


def wrap_opviews_with_values(dialect, module_name, excluded=[]):
  """Wraps all of a dialect's OpView classes to have their create method return
     a Signal instead of an OpView. The wrapped classes are inserted into
     the provided module."""
  import sys
  from .types import Type
  module = sys.modules[module_name]

  for attr in dir(dialect):
    cls = getattr(dialect, attr)

    if attr not in excluded and isinstance(cls, type) and issubclass(
        cls, ir.OpView):

      def specialize_create(cls):

        def create(*args, **kwargs):
          # If any of the arguments are Signal or Type (which are both PyCDE
          # classes) objects, we need to convert them.
          def to_circt(arg):
            if isinstance(arg, Signal):
              return arg.value
            elif isinstance(arg, Type):
              return arg._type
            return arg

          args = [to_circt(arg) for arg in args]
          kwargs = {k: to_circt(v) for k, v in kwargs.items()}
          # Create the OpView.
          with get_user_loc():
            if hasattr(cls, "create"):
              created = cls.create(*args, **kwargs)
            else:
              created = cls(*args, **kwargs)
            if isinstance(created, support.NamedValueOpView):
              created = created.opview
            if hasattr(created, "twoState"):
              created.twoState = True

          # Return the wrapped values, if any.
          converted_results = tuple(
              _FromCirctValue(res) for res in created.results)
          return converted_results[0] if len(
              converted_results) == 1 else converted_results

        return create

      wrapped_class = specialize_create(cls)
      setattr(module, attr, wrapped_class)
    else:
      setattr(module, attr, cls)
