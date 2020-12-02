// SPDX-License-Identifier: Apache-2.0
// Author: Kevin Laeufer <laeufer@cs.berkeley.edu>

package firrtl.backends.experimental.smt

import firrtl.annotations._
import firrtl.ir
import firrtl.passes.PassException

/** ExtModules annotated as UninterpretedModule will be modelled as
  * UninterpretedFunction (SMTLib) or constant arrays (btor2).
  * This can be useful when trying to abstract over a function that the
  * SMT solver or model checker is struggling with.
  *
  * E.g., one could declare an abstract 64bit multiplier like this:
  * ```
  * extmodule Mul64 :
  *   input a : UInt<64>
  *   input b : UInt<64>
  *   output r : UInt<64>
  * ```
  * Now instead of using Chisel to actually implement a multiplication circuit
  * we can instantiate this Mul64 module twice: Once in our implementation
  * and once for our correctness property that might specify how the
  * multiply instruction is supposed to be executed on our CPU.
  * Now instead of having to prove equivalence of multiplication circuits, the
  * solver only has to make sure that the connections to the multiplier are correct,
  * since if `a` and `b` are the same on both instances of `Mul64`, then the `r` output
  * will also be the same. This is a much easier problem and will result in much faster
  * solving due to manual abstraction.
  *
  * When [[stateBits]] is 0, we model the module as purely combinatorial circuit and
  * thus expect there to be no clock wire going into the module.
  * Every output is thus a function of all inputs of the module.
  *
  * When [[stateBits]] is an N greater than zero, we will model the module as having an abstract state of width N.
  * Thus on every clock transition the abstract state is updated and all outputs will take the state
  * as well as the current inputs as arguments.
  * TODO: Support for stateful circuits is work in progress.
  *
  * All output functions well be prefixed with [[prefix]] and end in the name of the output pin.
  * It is the users responsibility to ensure that all function names will be unique by choosing apropriate
  * prefixes.
  *
  * The annotation is consumed by the [[FirrtlToTransitionSystem]] pass.
  */
case class UninterpretedModuleAnnotation(target: ModuleTarget, prefix: String, stateBits: Int = 0)
    extends SingleTargetAnnotation[ModuleTarget] {
  require(stateBits >= 0, "negative number of bits is forbidden")
  if (stateBits > 0) throw new NotImplementedError("TODO: support for stateful circuits is not implemented yet!")
  override def duplicate(n: ModuleTarget) = copy(n)
}

object UninterpretedModuleAnnotation {

  /** checks to see whether the annotation module can actually be abstracted. Use *after* LowerTypes! */
  def checkModule(m: ir.DefModule, anno: UninterpretedModuleAnnotation): Unit = m match {
    case _: ir.Module =>
      throw new UninterpretedModuleException(s"UninterpretedModuleAnnotation can only be used with extmodule! $anno")
    case m: ir.ExtModule =>
      val clockInputs = m.ports.collect { case p @ ir.Port(_, _, ir.Input, ir.ClockType) => p.name }
      val clockOutput = m.ports.collect { case p @ ir.Port(_, _, ir.Output, ir.ClockType) => p.name }
      val asyncResets = m.ports.collect { case p @ ir.Port(_, _, _, ir.AsyncResetType) => p.name }
      if (clockOutput.nonEmpty) {
        throw new UninterpretedModuleException(
          s"We do not support clock outputs for uninterpreted modules! $clockOutput"
        )
      }
      if (asyncResets.nonEmpty) {
        throw new UninterpretedModuleException(
          s"We do not support async reset I/O for uninterpreted modules! $asyncResets"
        )
      }
      if (anno.stateBits == 0) {
        if (clockInputs.nonEmpty) {
          throw new UninterpretedModuleException(s"A combinatorial module may not have any clock inputs! $clockInputs")
        }
      } else {
        if (clockInputs.size != 1) {
          throw new UninterpretedModuleException(s"A stateful module must have exactly one clock input! $clockInputs")
        }
      }
  }
}

private class UninterpretedModuleException(s: String) extends PassException(s)
