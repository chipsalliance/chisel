// SPDX-License-Identifier: Apache-2.0

package chiseltest

import chisel3._
import scala.annotation.compileTimeOnly

/**
 * Formal compatibility API placeholders.
 *
 * Formal verification is currently unsupported in this compatibility layer.
 * Any usage should fail at compile time to avoid vacuously passing tests.
 */
package object formal {

  /** Annotation placeholder for source compatibility only. */
  case class BoundedCheck(depth: Int)

  @compileTimeOnly("chiseltest.formal.Formal is unsupported in this compatibility layer")
  trait Formal {
    @compileTimeOnly("chiseltest.formal.verify is unsupported in this compatibility layer")
    def verify[T <: Module](dut: => T, annotations: Seq[Any]): Unit =
      throw new UnsupportedOperationException("chiseltest.formal.verify is unsupported")
  }

  @compileTimeOnly("chiseltest.formal.past is unsupported in this compatibility layer")
  def past[T <: Data](x: T, delay: Int = 1): T =
    throw new UnsupportedOperationException("chiseltest.formal.past is unsupported")

  @compileTimeOnly("chiseltest.formal.past is unsupported in this compatibility layer")
  def past[T <: Data](x: T): T =
    throw new UnsupportedOperationException("chiseltest.formal.past is unsupported")
}
