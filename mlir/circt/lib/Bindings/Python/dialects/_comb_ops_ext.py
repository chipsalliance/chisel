from . import comb
from ..support import NamedValueOpView, get_value
from ..ir import IntegerAttr, IntegerType


# Builder base classes for non-variadic unary and binary ops.
class UnaryOpBuilder(NamedValueOpView):

  def operand_names(self):
    return ["input"]

  def result_names(self):
    return ["result"]


def UnaryOp(base):

  class _Class(base):

    @classmethod
    def create(cls, input=None, result_type=None):
      mapping = {"input": input} if input else {}
      return UnaryOpBuilder(cls, result_type, mapping)

  return _Class


class ExtractOpBuilder(UnaryOpBuilder):

  def __init__(self, low_bit, data_type, input_port_mapping={}, **kwargs):
    low_bit = IntegerAttr.get(IntegerType.get_signless(32), low_bit)
    super().__init__(comb.ExtractOp, data_type, input_port_mapping, [],
                     [low_bit], **kwargs)


class BinaryOpBuilder(NamedValueOpView):

  def operand_names(self):
    return ["lhs", "rhs"]

  def result_names(self):
    return ["result"]


def BinaryOp(base):

  class _Class(base):

    @classmethod
    def create(cls, lhs=None, rhs=None, result_type=None):
      mapping = {}
      if lhs:
        mapping["lhs"] = lhs
      if rhs:
        mapping["rhs"] = rhs
      return BinaryOpBuilder(cls, result_type, mapping)

  return _Class


# Base classes for the variadic ops.
def VariadicOp(base):

  class _Class(base):

    @classmethod
    def create(cls, *args):
      return cls([get_value(a) for a in args])

  return _Class


# Base class for miscellaneous ops that can't be abstracted but should provide a
# create method for uniformity.
def CreatableOp(base):

  class _Class(base):

    @classmethod
    def create(cls, *args, **kwargs):
      return cls(*args, **kwargs)

  return _Class


# Sugar classes for the various non-variadic unary ops.
class ExtractOp:

  @staticmethod
  def create(low_bit, result_type, input=None):
    mapping = {"input": input} if input else {}
    return ExtractOpBuilder(low_bit,
                            result_type,
                            mapping,
                            needs_result_type=True)


@UnaryOp
class ParityOp:
  pass


# Sugar classes for the various non-variadic binary ops.
@BinaryOp
class DivSOp:
  pass


@BinaryOp
class DivUOp:
  pass


@BinaryOp
class ModSOp:
  pass


@BinaryOp
class ModUOp:
  pass


@BinaryOp
class ShlOp:
  pass


@BinaryOp
class ShrSOp:
  pass


@BinaryOp
class ShrUOp:
  pass


@BinaryOp
class SubOp:
  pass


# Sugar classes for the variadic ops.
@VariadicOp
class AddOp:
  pass


@VariadicOp
class MulOp:
  pass


@VariadicOp
class AndOp:
  pass


@VariadicOp
class OrOp:
  pass


@VariadicOp
class XorOp:
  pass


@VariadicOp
class ConcatOp:
  pass


# Sugar classes for miscellaneous ops.
@CreatableOp
class MuxOp:
  pass
