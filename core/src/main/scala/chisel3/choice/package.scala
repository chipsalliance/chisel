// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.{BaseModule, SourceInfo}
import chisel3.util.simpleClassName
import chisel3.internal.Builder

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
  class DynamicGroup(val groupName: String, caseNames: Seq[String])(implicit sourceInfo: SourceInfo) {

    private val normalizedCaseNames = caseNames.toVector
    private val cachedGroup = Builder.getOrCreateDynamicGroup(
      groupName,
      normalizedCaseNames,
      DynamicGroup.groupFactory(groupName)
    )

    final implicit def group: Group = cachedGroup

    val cases: Map[String, Case] = normalizedCaseNames.iterator.map { caseName =>
      caseName -> Builder.getOrCreateDynamicCase(cachedGroup, caseName, DynamicGroup.caseFactory(cachedGroup, caseName))
    }.toMap

    def apply(caseName: String): Case = cases.getOrElse(
      caseName,
      throw new NoSuchElementException(
        s"Case '$caseName' not found in group '$groupName'. Available cases: ${cases.keys.mkString(", ")}"
      )
    )
  }

  object DynamicGroup {
    private def groupFactory(groupName: String)(implicit sourceInfo: SourceInfo): () => Group = () => {
      object DynamicGroupSingleton extends Group()(sourceInfo) {
        override private[chisel3] def name = groupName
      }
      DynamicGroupSingleton
    }

    private def caseFactory(group: Group, caseName: String)(implicit sourceInfo: SourceInfo): () => Case = () => {
      object DynamicCaseSingleton extends Case()(group, sourceInfo) {
        override private[chisel3] def name = caseName
      }
      DynamicCaseSingleton
    }

    /** Create a DynamicGroup with the given name and case names.
      * If a group with this name already exists in the elaboration context,
      * the returned DynamicGroup will share the same underlying Group singleton.
      *
      * @param name The name of the group
      * @param caseNames List of case names for this group
      * @param sourceInfo Source location information
      * @return A DynamicGroup with the given name
      */
    def apply(name: String, caseNames: Seq[String])(implicit sourceInfo: SourceInfo): DynamicGroup =
      new DynamicGroup(name, caseNames)

    /** Create a DynamicGroup with a builder function that provides a trait-like interface.
      * This allows you to define a type-safe interface for accessing cases.
      *
      * @tparam T The trait type that defines the case structure
      * @param name The name of the group
      * @param caseNames List of case names for this group (must match trait method names)
      * @param builder A function that takes the DynamicGroup and returns an instance of T
      * @param sourceInfo Source location information
      * @return An instance of T that provides access to the cases
      *
      * @example
      * {{{
      * trait PlatformType {
      *   def FPGA: Case
      *   def ASIC: Case
      * }
      * val platform = DynamicGroup[PlatformType]("Platform", Seq("FPGA", "ASIC")) { group =>
      *   new PlatformType {
      *     def FPGA = group("FPGA")
      *     def ASIC = group("ASIC")
      *   }
      * }
      * platform.FPGA // Type-safe access
      * }}}
      */
    def apply[T](name: String, caseNames: Seq[String])(builder: DynamicGroup => T)(implicit sourceInfo: SourceInfo): T =
      builder(apply(name, caseNames))
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
