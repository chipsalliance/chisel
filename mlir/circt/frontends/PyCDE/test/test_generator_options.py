# RUN: %PYTHON% %s | FileCheck %s
# XFAIL: *

from pycde import System, module, generator, types

from pycde.dialects import hw


@module
class GeneratorOptions:

  @generator
  def generator_a(mod):
    hw.ConstantOp(types.i32, 1)

  @generator
  def generator_b(mod):
    hw.ConstantOp(types.i32, 2)


# CHECK: hw.constant 1
top1 = System([GeneratorOptions])
top1.generate(["generator_a"])
top1.print()

# CHECK: hw.constant 2
top2 = System([GeneratorOptions])
top2.generate(["generator_b"])
top2.print()

# CHECK: generator exception
top3 = System([GeneratorOptions])
try:
  top3.generate()
except RuntimeError:
  print("generator exception")
  pass

# CHECK: generator exception
top4 = System([GeneratorOptions])
try:
  top4.generate(["generator_a", "generator_b"])
except RuntimeError:
  print("generator exception")
  pass

# CHECK: generator exception
top5 = System([GeneratorOptions])
try:
  top5.generate(["nonexistant"])
except RuntimeError:
  print("generator exception")
  pass
