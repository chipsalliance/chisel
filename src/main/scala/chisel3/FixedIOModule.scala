// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.hierarchy.{instantiable, public}
import chisel3.experimental.{BaseModule, UnlocatableSourceInfo}

/** A module or external module whose IO is generated from a specific generator.
  * This module may have no additional IO created other than what is specified
  * by its `ioGenerator` abstract member.
  */
@instantiable
sealed trait FixedIOBaseModule[A <: Data] extends BaseModule {

  /** A generator of IO */
  protected def ioGenerator: A

  @public
  final val io = FlatIO(ioGenerator)(UnlocatableSourceInfo)
  endIOCreation()

}

/** A Chisel module whose IO is determined by an IO generator.  This module
  * cannot have additional IO created by modules that extend it.
  *
  * @param ioGenerator
  */
class FixedIORawModule[A <: Data](final val ioGenerator: A) extends RawModule with FixedIOBaseModule[A]

/** A Chisel module whose IO (in addition to [[clock]] and [[reset]]) is determined
 *  by an IO generator. This module cannot have additional IO created by modules that
 *  extend it.
  *
  * @param ioGenerator
  */
class FixedIOModule[A <: Data](final val ioGenerator: A) extends Module with FixedIOBaseModule[A]

/** A Chisel blackbox whose IO is determined by an IO generator.  This module
  * cannot have additional IO created by modules that extend it.
  *
  * @param ioGenerator
  * @param params
  */
class FixedIOExtModule[A <: Data](final val ioGenerator: A, params: Map[String, Param] = Map.empty[String, Param])
    extends ExtModule(params)
    with FixedIOBaseModule[A]
