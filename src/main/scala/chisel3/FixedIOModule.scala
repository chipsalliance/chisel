// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.{BaseModule, ExtModule, FlatIO, Param}

/** A module or external module whose IO is generated from a specific generator.
  * This module may have no additional IO created other than what is specified
  * by its `ioGenerator` abstract member.
  */
sealed trait FixedIOBaseModule[A <: Data] extends BaseModule {

  /** A generator of IO */
  protected def ioGenerator: A

  final val io = FlatIO(ioGenerator)
  endIOCreation()

}

/** A Chisel module whose IO is determined by an IO generator.  This module
  * cannot have additional IO created by modules that extend it.
  *
  * @param ioGenerator
  */
class FixedIORawModule[A <: Data](final val ioGenerator: A) extends RawModule with FixedIOBaseModule[A]

/** A Chisel blackbox whose IO is determined by an IO generator.  This module
  * cannot have additional IO created by modules that extend it.
  *
  * @param ioGenerator
  * @param params
  */
class FixedIOExtModule[A <: Data](final val ioGenerator: A, params: Map[String, Param] = Map.empty[String, Param])
    extends ExtModule(params)
    with FixedIOBaseModule[A]
