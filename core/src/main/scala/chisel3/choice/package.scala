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
