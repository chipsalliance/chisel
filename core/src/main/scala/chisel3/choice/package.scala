// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.{BaseModule, SourceInfo}
import chisel3.internal.Builder
import chisel3.util.simpleClassName

import scala.collection.mutable
import scala.reflect.runtime.universe._
import scala.reflect.runtime.{currentMirror => cm}

/** This package contains Chisel language definitions for describing configuration options and their accepted values.
  */
package object choice {

  /** An option group declaration. Specifies a container grouping values for some design configuration parameter.
    *
    * @example
    * {{{
    * import chisel3.choice.{Group, Case}
    * object Platform extends Group {
    *   object FPGA extends Case
    *   object ASIC extends Case
    * }
    * }}}
    */
  abstract class Group(implicit _sourceInfo: SourceInfo) {
    self: Singleton =>

    private[choice] def registerCases(): Unit = {
      // Grab a symbol for the derived class (a concrete Group)
      val instanceMirror = cm.reflect(this)
      val symbol = instanceMirror.symbol

      symbol.typeSignature.members.collect {
        // Look only for inner objects in the Group. Note, this is not recursive.
        case m: ModuleSymbol if m.isStatic =>
          val instance = cm.reflectModule(m.asModule).instance
          // Confirms the instance is a subtype of Case
          if (cm.classSymbol(instance.getClass).toType <:< typeOf[Case]) {
            Builder.options += instance.asInstanceOf[Case]
          }
      }
    }

    private[chisel3] def sourceInfo: SourceInfo = _sourceInfo

    private[chisel3] def name: String = simpleClassName(this.getClass())

    final implicit def group: Group = this
  }

  /** An option case declaration.
    */
  abstract class Case(implicit val group: Group, _sourceInfo: SourceInfo) {
    self: Singleton =>

    private[chisel3] def sourceInfo: SourceInfo = _sourceInfo

    private[chisel3] def name: String = simpleClassName(this.getClass())

    /** A helper method to allow ModuleChoice to use the `->` syntax to specify case-module mappings.
      *
      * It captures a lazy reference to the module and produces a generator to avoid instantiating it.
      *
      * @param module Module to map to the current case.
      */
    def ->[T](module: => T): (Case, () => T) = (this, () => module)
  }

  /** Registers all options in a group with the Builder.
    * This lets Chisel know  that this layer should be emitted into FIRRTL text.
    *
    * This API can be used to guarantee that a design will always have certain
    * group defined.  This is analagous in spirit to [[layer.addLayer]].
    */
  def addGroup(group: Group): Unit = {
    group.registerCases()
  }
}
