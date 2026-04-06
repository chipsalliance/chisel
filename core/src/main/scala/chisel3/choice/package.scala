// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.{BaseModule, SourceInfo}
import chisel3.util.simpleClassName

/** This package contains Chisel language definitions for describing configuration options and their accepted values.
  */
package object choice {

  /** An option group declaration. Specifies a container grouping values for some design configuration parameter.
    *
    * @example
    * {{{
    * import chisel3.option.{Group, Case}
    * object Platform extends Group {
    *   object FPGA extends Case
    *   object ASIC extends Case
    * }
    * }}}
    */
  abstract class Group(implicit _sourceInfo: SourceInfo) {
    self: Singleton =>

    private[chisel3] def sourceInfo: SourceInfo = _sourceInfo

    private[chisel3] def name: String = simpleClassName(this.getClass())

    final implicit def group: Group = this
  }

  /** Dynamic option group with runtime-customizable name.
    *
    * Unlike static [[Group]] objects, DynamicGroup allows the group name to be specified at instantiation time.
    * This is useful for parameterized designs where the same group structure is reused with different names.
    *
    * @param customName The runtime name for this group
    *
    * @example
    * {{{
    * class Opt(name: String)(implicit sourceInfo: SourceInfo) extends DynamicGroup(name) {
    *   object Fast extends DynamicCase
    *   object Slow extends DynamicCase
    * }
    *
    * // Use with ModuleChoice
    * class MyModule extends Module {
    *   val opt = new Opt("OptMyModule")
    *   val impl = ModuleChoice(new DefaultImpl)(
    *     Seq(
    *       opt.Fast -> new FastImpl,
    *       opt.Slow -> new SlowImpl
    *     )
    *   )
    * }
    * }}}
    */
  abstract class DynamicGroup(customName: String)(implicit _sourceInfo: SourceInfo) {
    private[chisel3] def sourceInfo: SourceInfo = _sourceInfo

    private[chisel3] val groupName: String = customName

    private val _group: Group = {
      object DynamicGroupSingleton extends Group()(sourceInfo) {
        override private[chisel3] def name = groupName
      }
      DynamicGroupSingleton
    }

    final def group: Group = _group

    // Provide an implicit DynamicGroup for DynamicCase objects
    implicit protected def implicitGroup: DynamicGroup = this
  }

  /** An option case declaration for [[DynamicGroup]].
    *
    * DynamicCase objects must be defined inside a DynamicGroup class.
    * They use implicit parameters to automatically associate with their parent DynamicGroup.
    */
  abstract class DynamicCase(implicit val dynamicGroup: DynamicGroup, _sourceInfo: SourceInfo)
      extends Case()(using dynamicGroup.group, _sourceInfo) {
    self: Singleton =>
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
}
