# RUN: rm -rf %t
# RUN: %PYTHON% %s %t 2>&1 | FileCheck %s
# RUN: FileCheck %s --input-file %t/hw/PolynomialCompute.sv --check-prefix=OUTPUT

from __future__ import annotations

import pycde
from pycde import (AppID, Input, Output, generator, types)
from pycde.module import Module, modparams
from pycde.dialects import comb, hw
from pycde.constructs import Wire

import sys


@modparams
def PolynomialCompute(coefficients: Coefficients):

  class PolynomialCompute(Module):
    """Module to compute ax^3 + bx^2 + cx + d for design-time coefficients"""
    module_name = f"PolyComputeForCoeff_{coefficients.coeff}"

    # Evaluate polynomial for 'x'.
    x = Input(types.i32)
    y = Output(types.int(8 * 4))

    def __init__(self, name: str, **kwargs):
      """coefficients is in 'd' -> 'a' order."""
      super().__init__(instance_name=name, **kwargs)

    @generator
    def construct(mod):
      """Implement this module for input 'x'."""

      x = mod.x
      taps = list()
      for power, coeff in enumerate(coefficients.coeff):
        coeffVal = hw.ConstantOp(types.i32, coeff)
        if power == 0:
          newPartialSum = coeffVal
        else:
          partialSum = taps[-1]
          if power == 1:
            currPow = x
          else:
            x_power = [x for i in range(power)]
            currPow = comb.MulOp(*x_power)
          newPartialSum = comb.AddOp(partialSum, comb.MulOp(coeffVal, currPow))

        taps.append(newPartialSum)

      # Final output
      mod.y = taps[-1]

  return PolynomialCompute


class CoolPolynomialCompute(Module):
  module_name = "supercooldevice"
  x = Input(types.i32)
  y = Output(types.i32)

  def __init__(self, coefficients, **inputs):
    super().__init__(**inputs)
    self.coefficients = coefficients


@modparams
def ExternWithParams(A: str, B: int):

  typedef1 = types.struct({"a": types.i1}, "exTypedef")

  class M(Module):
    module_name = "parameterized_extern"
    ignored_input = Input(types.i1)
    used_input = Input(types.int(B))

    @property
    def instance_name(self):
      return "singleton"

  return M


class Coefficients:

  def __init__(self, coeff):
    self.coeff = coeff


class PolynomialSystem(Module):
  y = Output(types.i32)

  @generator
  def construct(self):
    i32 = types.i32
    x = hw.ConstantOp(i32, 23)
    poly = PolynomialCompute(Coefficients([62, 42, 6]))("example",
                                                        appid=AppID("poly", 0),
                                                        x=x)
    PolynomialCompute(coefficients=Coefficients([62, 42, 6]))("example2",
                                                              x=poly.y)
    PolynomialCompute(Coefficients([1, 2, 3, 4, 5]))("example2", x=poly.y)

    CoolPolynomialCompute([4, 42], x=23)

    w1 = Wire(types.i4)
    m = ExternWithParams("foo", 4)(ignored_input=None, used_input=w1)
    m.name = "pexternInst"
    w1.assign(0)

    self._set_outputs(poly._outputs())


poly = pycde.System([PolynomialSystem],
                    name="PolynomialSystem",
                    output_directory=sys.argv[1])
poly.print()

print("Generating 1...")
poly.generate(iters=1)

print("Printing...")
poly.print()
# CHECK-LABEL: msft.module @PolynomialSystem {} () -> (y: i32) attributes {fileName = "PolynomialSystem.sv"} {
# CHECK:         %example.y = msft.instance @example @PolyComputeForCoeff__62__42__6_(%c23_i32) {msft.appid = #msft.appid<"poly"[0]>} : (i32) -> i32
# CHECK:         %example2.y = msft.instance @example2 @PolyComputeForCoeff__62__42__6_(%example.y) : (i32) -> i32
# CHECK:         %example2_1.y = msft.instance @example2_1 @PolyComputeForCoeff__1__2__3__4__5_(%example.y) : (i32) -> i32
# CHECK:         %CoolPolynomialCompute.y = msft.instance @CoolPolynomialCompute @supercooldevice(%{{.+}}) : (i32) -> i32
# CHECK:         [[R0:%.+]] = hw.bitcast %false : (i1) -> i1
# CHECK:         msft.instance @singleton @parameterized_extern(%0, %c0_i4) <A: none = "foo", B: i64 = 4> : (i1, i4) -> ()
# CHECK:         %c0_i4 = hw.constant 0 : i4
# CHECK:         msft.output %example.y : i32
# CHECK:       }
# CHECK:       msft.module @PolyComputeForCoeff__62__42__6_ {coefficients = {coeff = [62, 42, 6]}} (%x: i32) -> (y: i32)
# CHECK:       msft.module @PolyComputeForCoeff__1__2__3__4__5_ {coefficients = {coeff = [1, 2, 3, 4, 5]}} (%x: i32) -> (y: i32)
# CHECK:       msft.module.extern @supercooldevice(%x: i32) -> (y: i32) attributes {verilogName = "supercooldevice"}
# CHECK:       msft.module.extern @parameterized_extern<A: none, B: i64>(%ignored_input: i1, %used_input: i4) attributes {verilogName = "parameterized_extern"}

print("Generating rest...")
poly.generate()
poly.print()

print("=== Post-generate IR...")
poly.run_passes()
poly.print()
# CHECK-LABEL: === Post-generate IR...
# CHECK: hw.module @PolynomialSystem
# CHECK: %[[EXAMPLE_Y:.+]] = hw.instance "example" sym @example @PolyComputeForCoeff__62__42__6_<__INST_HIER: none = #hw.param.expr.str.concat<#hw.param.decl.ref<"__INST_HIER">, ".example">>(x: %c23_i32: i32) -> (y: i32)
# CHECK: %example2.y = hw.instance "example2" sym @example2 @PolyComputeForCoeff__62__42__6_<__INST_HIER: none = #hw.param.expr.str.concat<#hw.param.decl.ref<"__INST_HIER">, ".example2">>(x: %[[EXAMPLE_Y]]: i32) -> (y: i32)
# CHECK: hw.instance "example2_1" sym @example2_1 @PolyComputeForCoeff__1__2__3__4__5_<__INST_HIER: none = #hw.param.expr.str.concat<#hw.param.decl.ref<"__INST_HIER">, ".example2_1">>(x: %[[EXAMPLE_Y]]: i32)
# CHECK: %CoolPolynomialCompute.y = hw.instance "CoolPolynomialCompute" sym @CoolPolynomialCompute @supercooldevice(x: %c23_i32{{.*}}: i32) -> (y: i32)
# CHECK-LABEL: hw.module @PolyComputeForCoeff__62__42__6_<__INST_HIER: none = "INSTANTIATE_WITH_INSTANCE_PATH">(%x: i32) -> (y: i32)
# CHECK-LABEL: hw.module @PolyComputeForCoeff__1__2__3__4__5_<__INST_HIER: none = "INSTANTIATE_WITH_INSTANCE_PATH">(%x: i32)
# CHECK-NOT: hw.module @pycde.PolynomialCompute

poly.emit_outputs()

# OUTPUT-LABEL: `ifndef __PYCDE_TYPES__
# OUTPUT: `define __PYCDE_TYPES__
# OUTPUT: typedef struct packed {logic a; } exTypedef;
# OUTPUT: `endif // __PYCDE_TYPES__

# OUTPUT-LABEL:   module PolyComputeForCoeff__62__42__6_
# OUTPUT:    input  [31:0] x,
# OUTPUT:    output [31:0] y
# OUTPUT:    );
