// See LICENSE for license details.

package chisel3.plugin

import chisel3.{CompileOptions, Data}
import chisel3.experimental.BaseModule
import chisel3.internal.Builder
import chisel3.internal.sourceinfo.SourceInfo

/** These are public APIs because the compiler plugin must generate code which calls these
  * However, they are not for users to use, so unless you know what you are doing, DO NOT USE THESE FUNCTIONS!!
  */
object APIs {

  /** Returns null if we are building an Instance, rather than a Module
    * @param thing A line of code in a Module's constructor, that optionally will not be executed
    * @param sourceInfo source code info
    * @param compileOptions compile options info
    * @tparam X Return type of code to optionally execute
    * @return Returns either thing (executed) or null
    */
  def nullifyIfInstance[X](thing: => X)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): X = {
    val back = Builder.getBackingModule() orElse Builder.currentModule.flatMap(_.backingModule)
    if (back.isDefined) {
      null.asInstanceOf[X]
    } else {
      val x = thing
      x
    }
  }

  /** If its an instance, return the backing module. Otherwise, returns provided module.
    * @param moduleOrInstance
    * @tparam X
    * @return Module
    */
  def resolveBackingModule[X <: BaseModule](moduleOrInstance: X): X = {
    moduleOrInstance.backingModule.map(_._1).getOrElse(moduleOrInstance).asInstanceOf[X]
  }

  /** Resolves a module access, given the context that the original access was on either a module or an instance
    * @param moduleOrInstance Original module/instance the access was on
    * @param moduleAccess The new access on the module/backingmodule
    * @tparam X
    * @return
    */
  def resolveModuleAccess[X](moduleOrInstance: BaseModule, moduleAccess: X): X = {
    moduleOrInstance.backingModule match {
      case Some((module, instance)) =>
        val portMap = module.getModulePorts.zip(instance.io.elements.values).toMap
        moduleAccess match {
          case d: Data if portMap.contains(d) => portMap(d).asInstanceOf[X]
          case _ => moduleAccess
        }
      case _ => moduleAccess
    }
  }
}
