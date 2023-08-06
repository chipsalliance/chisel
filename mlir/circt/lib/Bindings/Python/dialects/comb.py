#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ._comb_ops_gen import *

from ..support import NamedValueOpView

from ..ir import IntegerAttr, IntegerType, OpView


# Sugar classes for the various possible verions of ICmpOp.
class ICmpOpBuilder(NamedValueOpView):

  def operand_names(self):
    return ["lhs", "rhs"]

  def result_names(self):
    return ["result"]

  def __init__(self, predicate, data_type, input_port_mapping={}, **kwargs):
    predicate = IntegerAttr.get(IntegerType.get_signless(64), predicate)
    super().__init__(ICmpOp, data_type, input_port_mapping, [predicate],
                     **kwargs)


def CompareOp(predicate):

  def decorated(cls):

    class _Class(cls, OpView):

      @staticmethod
      def create(lhs=None, rhs=None):
        mapping = {}
        if lhs:
          mapping["lhs"] = lhs
        if rhs:
          mapping["rhs"] = rhs
        if len(mapping) == 0:
          result_type = IntegerType.get_signless(1)
        else:
          result_type = None
        return ICmpOpBuilder(predicate, result_type, mapping)

    return _Class

  return decorated


@CompareOp(0)
class EqOp:
  pass


@CompareOp(1)
class NeOp:
  pass


@CompareOp(2)
class LtSOp:
  pass


@CompareOp(3)
class LeSOp:
  pass


@CompareOp(4)
class GtSOp:
  pass


@CompareOp(5)
class GeSOp:
  pass


@CompareOp(6)
class LtUOp:
  pass


@CompareOp(7)
class LeUOp:
  pass


@CompareOp(8)
class GtUOp:
  pass


@CompareOp(9)
class GeUOp:
  pass
