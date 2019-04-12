// See LICENSE for license details.

package chisel3

import chisel3.experimental._
import chisel3.internal.Builder._
import chisel3.internal._
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo.{InstTransform, SourceInfo}

import scala.language.experimental.macros

object Module extends SourceInfoDoc {
  /** A wrapper method that all Module instantiations must be wrapped in
    * (necessary to help Chisel track internal state).
    *
    * @param bc the Module being created
    *
    * @return the input module `m` with Chisel metadata properly set
    */
  def apply[T <: BaseModule](bc: => T): T = macro InstTransform.apply[T]

  /** @group SourceInfoTransformMacro */
  def do_apply[T <: BaseModule](bc: => T)
                               (implicit sourceInfo: SourceInfo,
                                         compileOptions: CompileOptions): T = {
    if (Builder.readyForModuleConstr) {
      throwException("Error: Called Module() twice without instantiating a Module." +
                     sourceInfo.makeMessage(" See " + _))
    }
    Builder.readyForModuleConstr = true

    val parent = Builder.currentModule
    val whenDepth: Int = Builder.whenDepth

    // Save then clear clock and reset to prevent leaking scope, must be set again in the Module
    val clockAndReset: Option[ClockAndReset] = Builder.currentClockAndReset
    Builder.currentClockAndReset = None

    // Execute the module, this has the following side effects:
    //   - set currentModule
    //   - unset readyForModuleConstr
    //   - reset whenDepth to 0
    //   - set currentClockAndReset
    val module: T = bc  // bc is actually evaluated here

    if (Builder.whenDepth != 0) {
      throwException("Internal Error! when() scope depth is != 0, this should have been caught!")
    }
    if (Builder.readyForModuleConstr) {
      throwException("Error: attempted to instantiate a Module, but nothing happened. " +
                     "This is probably due to rewrapping a Module instance with Module()." +
                     sourceInfo.makeMessage(" See " + _))
    }
    Builder.currentModule = parent // Back to parent!
    Builder.whenDepth = whenDepth
    Builder.currentClockAndReset = clockAndReset // Back to clock and reset scope

    val component = module.generateComponent()
    Builder.components += component

    // Handle connections at enclosing scope
    if(!Builder.currentModule.isEmpty) {
      pushCommand(DefInstance(sourceInfo, module, component.ports))
      module.initializeInParent(compileOptions)
    }
    module
  }

  /** Returns the implicit Clock */
  def clock: Clock = Builder.forcedClock
  /** Returns the implicit Reset */
  def reset: Reset = Builder.forcedReset
  /** Returns the current Module */
  def currentModule: Option[BaseModule] = Builder.currentModule
}
