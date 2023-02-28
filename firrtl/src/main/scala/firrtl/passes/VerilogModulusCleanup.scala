// SPDX-License-Identifier: Apache-2.0

package firrtl
package passes

import firrtl.ir._
import firrtl.options.Dependency

import scala.collection.mutable

/**
  * Verilog has the width of (a % b) = Max(W(a), W(b))
  * FIRRTL has the width of (a % b) = Min(W(a), W(b)), which makes more sense,
  * but nevertheless is a problem when emitting verilog
  *
  * This pass finds every instance of (a % b) and:
  *   1) adds a temporary node equal to (a % b) with width Max(W(a), W(b))
  *   2) replaces the reference to (a % b) with a bitslice of the temporary node
  *      to get back down to width Min(W(a), W(b))
  *
  *  This is technically incorrect firrtl, but allows the verilog emitter
  *  to emit correct verilog without needing to add temporary nodes
  */
@deprecated("This pass's functionality has been moved to LegalizeVerilog", "FIRRTL 1.5.2")
object VerilogModulusCleanup extends Pass {

  override def prerequisites = firrtl.stage.Forms.LowFormMinimumOptimized ++
    Seq(
      Dependency[firrtl.transforms.BlackBoxSourceHelper],
      Dependency[firrtl.transforms.FixAddingNegativeLiterals],
      Dependency[firrtl.transforms.ReplaceTruncatingArithmetic],
      Dependency[firrtl.transforms.InlineBitExtractionsTransform],
      Dependency[firrtl.transforms.InlineAcrossCastsTransform],
      Dependency[firrtl.transforms.LegalizeClocksAndAsyncResetsTransform],
      Dependency[firrtl.transforms.FlattenRegUpdate]
    )

  override def optionalPrerequisites = firrtl.stage.Forms.LowFormOptimized

  override def optionalPrerequisiteOf = Seq.empty

  override def invalidates(a: Transform) = false

  def run(c: Circuit): Circuit = c
}
