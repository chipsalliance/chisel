// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.{BaseModule, ExtModule, Param}
import chisel3.experimental.hierarchy.core.Lookupable

/** A module or external module whose IO is generated from a specific generator.
  * This module may have no additional IO created other than what is specified
  * by its `ioGenerator` abstract member.
  */
sealed trait FixedIOBaseModule[A](using lookupable: Lookupable[A]) extends BaseModule {

  /** A generator of IO */
  protected def ioGenerator: A

  final val io: A = {
    val dataElems = lookupable.in(ioGenerator)
    val names = LazyList.from(0).map(i => ('a' + i).toChar.toString)
    val ports = dataElems.zip(names).map { case (d, name) =>
      val p = IO(d)
      p.suggestName(name)
      p
    }
    lookupable.out(ioGenerator, ports.iterator)
  }
  endIOCreation()

}

/** A Chisel module whose IO is determined by an IO generator.  This module
  * cannot have additional IO created by modules that extend it.
  *
  * @param ioGenerator
  */
class FixedIORawModule[A](final val ioGenerator: A)(using Lookupable[A]) extends RawModule with FixedIOBaseModule[A]

/** A Chisel module whose IO (in addition to [[clock]] and [[reset]]) is determined
 *  by an IO generator. This module cannot have additional IO created by modules that
 *  extend it.
  *
  * @param ioGenerator
  */
class FixedIOModule[A](final val ioGenerator: A)(using Lookupable[A]) extends Module with FixedIOBaseModule[A]

/** A Chisel blackbox whose IO is determined by an IO generator.  This module
  * cannot have additional IO created by modules that extend it.
  *
  * @param ioGenerator
  * @param params
  */
class FixedIOExtModule[A](final val ioGenerator: A, params: Map[String, Param] = Map.empty[String, Param])(
  using Lookupable[A]
) extends ExtModule(params)
    with FixedIOBaseModule[A]
