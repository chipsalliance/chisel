// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.hierarchy.{instantiable, public}
import chisel3.experimental.hierarchy.core.{Definition, Instance, Lookupable}
import chisel3.experimental.{BaseModule, UnlocatableSourceInfo}
import chisel3.experimental.dataview.DataProduct

/** A module or external module whose IO is generated from a specific generator.
  * This module may have no additional IO created other than what is specified
  * by its `ioGenerator` abstract member.
  */
@instantiable
sealed trait FixedIOBaseModule[A] extends BaseModule {

  /** A generator of IO */
  protected def ioGenerator: A

  // Concrete subclasses implement this via an implicit constructor parameter.
  protected implicit def lookupable: Lookupable[A]

  protected implicit def dataProduct: DataProduct[A]

  private[chisel3] def _lookupable: Lookupable[A] = lookupable

  // No @public, lookups implemented manually in companion object.
  final val io: A = {
    val a = ioGenerator
    val dataElems = lookupable.in(a)
    // TODO add check for name collisions
    val nameLookup = dataProduct.dataIterator(a, "").toMap
    val io = dataElems.map { d =>
      val p = FlatIO(d)(UnlocatableSourceInfo)
      val name = {
        val n = nameLookup(d)
        if (n == "") "io" else n
      }
      p.suggestName(name)
      p
    }
    lookupable.out(ioGenerator, io.iterator)
  }
  endIOCreation()

}

object FixedIOBaseModule {

  // @public val io: A on the trait would generate these, but the @instantiable macro produces an
  // external implicit class with no access to the trait's `lookupable` member.  We replicate the
  // generated lookup by passing proto.lookupable explicitly instead.
  implicit class DefinitionOps[A](___module: Definition[FixedIOBaseModule[A]]) {
    def io: A = {
      implicit val _mg: chisel3.internal.MacroGenerated = new chisel3.internal.MacroGenerated {}
      implicit val _l: Lookupable.Aux[A, A] =
        ___module.proto.lookupable.asInstanceOf[Lookupable.Aux[A, A]]
      ___module._lookup(_.io)
    }
  }

  implicit class InstanceOps[A](___module: Instance[FixedIOBaseModule[A]]) {
    def io: A = {
      implicit val _mg: chisel3.internal.MacroGenerated = new chisel3.internal.MacroGenerated {}
      implicit val _l: Lookupable.Aux[A, A] =
        ___module.proto.lookupable.asInstanceOf[Lookupable.Aux[A, A]]
      ___module._lookup(_.io)
    }
  }
}

/** A Chisel module whose IO is determined by an IO generator.  This module
  * cannot have additional IO created by modules that extend it.
  *
  * @param ioGenerator
  */
class FixedIORawModule[A](final val ioGenerator: A)(
  implicit val lookupable: Lookupable[A],
  val dataProduct:         DataProduct[A]
) extends RawModule
    with FixedIOBaseModule[A]

/** A Chisel module whose IO (in addition to [[clock]] and [[reset]]) is determined
 *  by an IO generator. This module cannot have additional IO created by modules that
 *  extend it.
  *
  * @param ioGenerator
  */
class FixedIOModule[A](final val ioGenerator: A)(
  implicit val lookupable: Lookupable[A],
  val dataProduct:         DataProduct[A]
) extends Module
    with FixedIOBaseModule[A]

/** A Chisel blackbox whose IO is determined by an IO generator.  This module
  * cannot have additional IO created by modules that extend it.
  *
  * @param ioGenerator
  * @param params
  */
class FixedIOExtModule[A](final val ioGenerator: A, params: Map[String, Param] = Map.empty[String, Param])(
  implicit val lookupable: Lookupable[A],
  val dataProduct:         DataProduct[A]
) extends ExtModule(params)
    with FixedIOBaseModule[A]
