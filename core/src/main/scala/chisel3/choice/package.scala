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

  /** Schema-style dynamic option group.
    *
    * This allows trait-based declarations of cases without requiring a singleton [[Group]] object.
    *
    * @example
    * {{{
    * trait PlatformType extends DynamicGroup {
    *   object FPGA extends Case
    *   object ASIC extends Case
    * }
    * }}}
    */
  trait DynamicGroup {
    private var initializedName:       Option[String] = None
    private var initializedCaseNames:  Option[Seq[String]] = None
    private var initializedSourceInfo: Option[SourceInfo] = None

    private def groupName: String = initializedName.getOrElse(
      throw new IllegalStateException("DynamicGroup was used before it was initialized")
    )

    private def caseNames: Seq[String] = initializedCaseNames.getOrElse(
      throw new IllegalStateException("DynamicGroup was used before it was initialized")
    )

    private implicit def sourceInfo: SourceInfo = initializedSourceInfo.getOrElse(
      throw new IllegalStateException("DynamicGroup was used before it was initialized")
    )

    private[chisel3] def initialize(name: String, caseNames: Seq[String], sourceInfo: SourceInfo): this.type = {
      initializedName = Some(name)
      initializedCaseNames = Some(caseNames.toVector)
      initializedSourceInfo = Some(sourceInfo)
      this
    }

    private lazy val cachedGroup = Builder.getOrCreateDynamicGroup(
      groupName,
      caseNames.toVector,
      DynamicGroup.groupFactory(groupName)
    )

    final implicit def group: Group = cachedGroup
  }
  object DynamicGroup {
    private[chisel3] def groupFactory(groupName: String)(implicit sourceInfo: SourceInfo): () => Group = () => {
      object DynamicGroupSingleton extends Group()(sourceInfo) {
        override private[chisel3] def name = groupName
      }
      DynamicGroupSingleton
    }

    /** Typeclass for constructing trait-based [[DynamicGroup]] instances.
      */
    trait Factory[T <: DynamicGroup] {
      def caseNames:                                 Seq[String]
      def create()(implicit sourceInfo: SourceInfo): T
    }
    object Factory extends DynamicGroupFactoryIntf

    /** Create a trait-based [[DynamicGroup]] using an implicit factory.
      *
      * @tparam T The trait type that defines the case structure
      * @param groupName The name of the group
      * @param sourceInfo Source location information
      * @return An instance of T that provides type-safe access to the cases
      *
      * @example
      * {{{
      * trait PlatformType extends DynamicGroup {
      *   object FPGA extends Case
      *   object ASIC extends Case
      * }
      * val platform = DynamicGroup[PlatformType]("Platform")
      * }}}
      */
    def apply[T <: DynamicGroup](groupName: String)(implicit factory: Factory[T], sourceInfo: SourceInfo): T = {
      val instance = factory.create().initialize(groupName, factory.caseNames, sourceInfo)
      instance.group
      instance
    }
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
