from ..support import BackedgeBuilder, NamedValueOpView
from ..ir import IntegerType, OpView, StringAttr
from . import hw


class CompRegLikeBuilder(NamedValueOpView):

  def result_names(self):
    return ["data"]

  def create_initial_value(self, index, data_type, arg_name):
    if arg_name == "input":
      operand_type = data_type
    else:
      operand_type = IntegerType.get_signless(1)
    return BackedgeBuilder.create(operand_type, arg_name, self)


class CompRegLike:

  def __init__(self,
               data_type,
               input,
               clk,
               clockEnable=None,
               *,
               reset=None,
               reset_value=None,
               name=None,
               sym_name=None,
               loc=None,
               ip=None):
    operands = []
    results = []
    attributes = {}
    results.append(data_type)
    operands.append(input)
    operands.append(clk)
    if isinstance(self, CompRegOp):
      if clockEnable is not None:
        raise Exception("Clock enable not supported on compreg")
    elif isinstance(self, CompRegClockEnabledOp):
      if clockEnable is None:
        raise Exception("Clock enable required on compreg.ce")
      operands.append(clockEnable)
    else:
      assert False, "Class not recognized"
    if reset is not None:
      operands.append(reset)
    if reset_value is not None:
      operands.append(reset_value)
    if name is None:
      attributes["name"] = StringAttr.get("")
    else:
      attributes["name"] = StringAttr.get(name)
    if sym_name is not None:
      attributes["inner_sym"] = hw.InnerSymAttr.get(StringAttr.get(sym_name))

    OpView.__init__(
        self,
        self.build_generic(
            attributes=attributes,
            results=results,
            operands=operands,
            loc=loc,
            ip=ip,
        ),
    )


class CompRegBuilder(CompRegLikeBuilder):

  def operand_names(self):
    return ["input", "clk"]


class CompRegOp(CompRegLike):

  @classmethod
  def create(cls,
             result_type,
             reset=None,
             reset_value=None,
             name=None,
             sym_name=None,
             **kwargs):
    return CompRegBuilder(cls,
                          result_type,
                          kwargs,
                          reset=reset,
                          reset_value=reset_value,
                          name=name,
                          sym_name=sym_name,
                          needs_result_type=True)


class CompRegClockEnabledBuilder(CompRegLikeBuilder):

  def operand_names(self):
    return ["input", "clk", "clockEnable"]


class CompRegClockEnabledOp(CompRegLike):

  @classmethod
  def create(cls,
             result_type,
             reset=None,
             reset_value=None,
             name=None,
             sym_name=None,
             **kwargs):
    return CompRegClockEnabledBuilder(cls,
                                      result_type,
                                      kwargs,
                                      reset=reset,
                                      reset_value=reset_value,
                                      name=name,
                                      sym_name=sym_name,
                                      needs_result_type=True)
