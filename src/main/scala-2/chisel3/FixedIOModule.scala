// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.dataview.DataProduct
import chisel3.experimental.hierarchy.{Definition, Instance, IsInstantiable, Lookupable}
import chisel3.experimental.{BaseModule, UnlocatableSourceInfo}
import chisel3.internal.MacroGenerated

/** A module or external module whose IO is generated from a specific generator.
  * This module may have no additional IO created other than what is specified
  * by its `ioGenerator` abstract member.
  *
  * `A` may be any type that has a [[chisel3.experimental.dataview.DataProduct]]
  * implementation, including a single [[Data]], a tuple of `Data`-containing
  * types, or a `Seq` of `Data`-containing types.  Each contained [[Data]] is
  * turned into a port; for the single-`Data` case [[FlatIO]] is used so that
  * existing modules are unaffected.
  */
sealed trait FixedIOBaseModule[A] extends BaseModule with IsInstantiable {

  /** A generator of IO */
  protected def ioGenerator: A

  /** Evidence that `A` contains [[Data]] elements that can be turned into ports. */
  protected implicit def ioDataProduct: DataProduct[A]

  /** The IO of this module, of shape `A` whose contained [[Data]] are ports. */
  def io: A

}

/** A Chisel module whose IO is determined by an IO generator.  This module
  * cannot have additional IO created by modules that extend it.
  *
  * @param ioGenerator
  */
class FixedIORawModule[A](final val ioGenerator: A)(implicit val ioDataProduct: DataProduct[A])
    extends RawModule
    with FixedIOBaseModule[A] {
  final val io: A = FixedIO.bindPorts(ioGenerator, "io")(UnlocatableSourceInfo).asInstanceOf[A]
  endIOCreation()
}

object FixedIORawModule {
  // Manually-written equivalent of what `@public` on `io` would produce, but
  // with a `Lookupable[A]` evidence param so we can support arbitrary
  // `DataProduct[A]`-shaped IOs in addition to the historical `A <: Data` case.
  implicit class FixedIORawModuleDefinitionExtensions[A](
    private val ___module: Definition[FixedIORawModule[A]]
  ) extends AnyVal {
    def io(implicit lookup: Lookupable[A]): lookup.C = {
      implicit val mg: MacroGenerated = new MacroGenerated {}
      ___module._lookup(_.io)
    }
  }
  implicit class FixedIORawModuleInstanceExtensions[A](
    private val ___module: Instance[FixedIORawModule[A]]
  ) extends AnyVal {
    def io(implicit lookup: Lookupable[A]): lookup.C = {
      implicit val mg: MacroGenerated = new MacroGenerated {}
      ___module._lookup(_.io)
    }
  }
}

/** A Chisel module whose IO (in addition to [[clock]] and [[reset]]) is determined
 *  by an IO generator. This module cannot have additional IO created by modules that
 *  extend it.
  *
  * @param ioGenerator
  */
class FixedIOModule[A](final val ioGenerator: A)(implicit val ioDataProduct: DataProduct[A])
    extends Module
    with FixedIOBaseModule[A] {
  final val io: A = FixedIO.bindPorts(ioGenerator, "io")(UnlocatableSourceInfo).asInstanceOf[A]
  endIOCreation()
}

object FixedIOModule {
  implicit class FixedIOModuleDefinitionExtensions[A](
    private val ___module: Definition[FixedIOModule[A]]
  ) extends AnyVal {
    def io(implicit lookup: Lookupable[A]): lookup.C = {
      implicit val mg: MacroGenerated = new MacroGenerated {}
      ___module._lookup(_.io)
    }
  }
  implicit class FixedIOModuleInstanceExtensions[A](
    private val ___module: Instance[FixedIOModule[A]]
  ) extends AnyVal {
    def io(implicit lookup: Lookupable[A]): lookup.C = {
      implicit val mg: MacroGenerated = new MacroGenerated {}
      ___module._lookup(_.io)
    }
  }
}

/** A Chisel blackbox whose IO is determined by an IO generator.  This module
  * cannot have additional IO created by modules that extend it.
  *
  * @param ioGenerator
  * @param params
  */
class FixedIOExtModule[A](
  final val ioGenerator: A,
  params:                Map[String, Param] = Map.empty[String, Param]
)(
  implicit val ioDataProduct: DataProduct[A])
    extends ExtModule(params)
    with FixedIOBaseModule[A] {
  final val io: A = FixedIO.bindPorts(ioGenerator, "io")(UnlocatableSourceInfo).asInstanceOf[A]
  endIOCreation()
}

object FixedIOExtModule {
  implicit class FixedIOExtModuleDefinitionExtensions[A](
    private val ___module: Definition[FixedIOExtModule[A]]
  ) extends AnyVal {
    def io(implicit lookup: Lookupable[A]): lookup.C = {
      implicit val mg: MacroGenerated = new MacroGenerated {}
      ___module._lookup(_.io)
    }
  }
  implicit class FixedIOExtModuleInstanceExtensions[A](
    private val ___module: Instance[FixedIOExtModule[A]]
  ) extends AnyVal {
    def io(implicit lookup: Lookupable[A]): lookup.C = {
      implicit val mg: MacroGenerated = new MacroGenerated {}
      ___module._lookup(_.io)
    }
  }
}
