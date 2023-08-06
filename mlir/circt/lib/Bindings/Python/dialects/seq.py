#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ._seq_ops_gen import *

from .seq import CompRegOp


# Create a computational register whose input is the given value, and is clocked
# by the given clock. If a reset is provided, the register will be reset by that
# signal. If a reset value is provided, the register will reset to that,
# otherwise it will reset to zero. If name is provided, the register will be
# named.
def reg(value, clock, reset=None, reset_value=None, name=None, sym_name=None):
  from . import hw
  from ..ir import IntegerAttr
  value_type = value.type
  if reset:
    if not reset_value:
      zero = IntegerAttr.get(value_type, 0)
      reset_value = hw.ConstantOp(zero).result
    return CompRegOp.create(value_type,
                            input=value,
                            clk=clock,
                            reset=reset,
                            reset_value=reset_value,
                            name=name,
                            sym_name=sym_name).data.value
  else:
    return CompRegOp.create(value_type,
                            input=value,
                            clk=clock,
                            name=name,
                            sym_name=sym_name).data.value
