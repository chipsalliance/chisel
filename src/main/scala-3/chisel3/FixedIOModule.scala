// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.{BaseModule, ExtModule, Param}
import chisel3.experimental.dataview.DataProduct
import chisel3.experimental.hierarchy.{instantiable, public}

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
sealed trait FixedIOBaseModule[A] extends BaseModule {

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
@instantiable
class FixedIORawModule[A](final val ioGenerator: A)(using val ioDataProduct: DataProduct[A])
    extends RawModule
    with FixedIOBaseModule[A] {
  @public
  final val io: A = FixedIO.bindPorts(ioGenerator, "io").asInstanceOf[A]
  endIOCreation()
}

/** A Chisel module whose IO (in addition to [[clock]] and [[reset]]) is determined
 *  by an IO generator. This module cannot have additional IO created by modules that
 *  extend it.
  *
  * @param ioGenerator
  */
@instantiable
class FixedIOModule[A](final val ioGenerator: A)(using val ioDataProduct: DataProduct[A])
    extends Module
    with FixedIOBaseModule[A] {
  @public
  final val io: A = FixedIO.bindPorts(ioGenerator, "io").asInstanceOf[A]
  endIOCreation()
}

/** A Chisel blackbox whose IO is determined by an IO generator.  This module
  * cannot have additional IO created by modules that extend it.
  *
  * @param ioGenerator
  * @param params
  */
@instantiable
class FixedIOExtModule[A](
  final val ioGenerator: A,
  params:                Map[String, Param] = Map.empty[String, Param]
)(using val ioDataProduct: DataProduct[A])
    extends ExtModule(params)
    with FixedIOBaseModule[A] {
  @public
  final val io: A = FixedIO.bindPorts(ioGenerator, "io").asInstanceOf[A]
  endIOCreation()
}
