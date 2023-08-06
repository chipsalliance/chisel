from ..support import NamedValueOpView, get_value
from ..ir import IntegerAttr, IntegerType


class BinaryOpBuilder(NamedValueOpView):

  def operand_names(self):
    return ["lhs", "rhs"]

  def result_names(self):
    return ["result"]


def BinaryOp(base):

  class _Class(base):

    @classmethod
    def create(cls, lhs=None, rhs=None, result_type=None):
      return cls([get_value(lhs), get_value(rhs)])

  return _Class


@BinaryOp
class DivOp:
  pass


@BinaryOp
class SubOp:
  pass


@BinaryOp
class AddOp:
  pass


@BinaryOp
class MulOp:
  pass


class CastOp:

  @classmethod
  def create(cls, value, result_type):
    return cls(result_type, value)


class ICmpOp:
  # Predicate constants.

  # `==` and `!=`
  PRED_EQ = 0b000
  PRED_NE = 0b001
  # `<` and `>=`
  PRED_LT = 0b010
  PRED_GE = 0b011
  # `<=` and `>`
  PRED_LE = 0b100
  PRED_GT = 0b101

  @classmethod
  def create(cls, pred, a, b):
    if isinstance(pred, int):
      pred = IntegerAttr.get(IntegerType.get_signless(64), pred)
    return cls(pred, a, b)


class ConstantOp:

  @classmethod
  def create(cls, data_type, value):
    return cls(IntegerAttr.get(data_type, value))
