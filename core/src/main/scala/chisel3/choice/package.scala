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

  /** Dynamic option group that accepts a name and case names as runtime parameters.
    * @example {{{ val platform = new DynamicGroup("Platform", Seq("FPGA", "ASIC")) }}}
    */
  class DynamicGroup(val groupName: String, caseNames: Seq[String])(implicit _sourceInfo: SourceInfo) {
    import chisel3.internal.Builder

    private[chisel3] def sourceInfo: SourceInfo = _sourceInfo
    private[chisel3] def name: String = groupName

    private val _group: Group = Builder.getOrCreateDynamicGroup(groupName, caseNames, () => {
      object DynamicGroupSingleton extends Group()(_sourceInfo) {
        override private[chisel3] def name = groupName
      }
      DynamicGroupSingleton
    })

    final implicit def group: Group = _group

    private val _cases: Map[String, Case] = caseNames.map { caseName =>
      caseName -> Builder.getOrCreateDynamicCase(_group, caseName, () => {
        object DynamicCaseSingleton extends Case()(_group, _sourceInfo) {
          override private[chisel3] def name = caseName
        }
        DynamicCaseSingleton
      })
    }.toMap

    def cases: Map[String, Case] = _cases

    def apply(caseName: String): Case = _cases.getOrElse(
      caseName,
      throw new NoSuchElementException(s"Case '$caseName' not found in group '$groupName'. Available cases: ${_cases.keys.mkString(", ")}")
    )
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
