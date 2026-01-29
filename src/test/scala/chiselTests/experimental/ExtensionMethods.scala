// SPDX-License-Identifier: Apache-2.0

package chiselTests.experimental

import chisel3.RawModule
import circt.stage.ChiselStage

/** Object that contains various extension methods for all the [[experimental]] package. */
private[experimental] object ExtensionMethods {

  /** Extension methods for [[ChiselStage]]. */
  implicit class ChiselStageHelpers(obj: ChiselStage.type) {

    /** Construct a module and return it.  This has a _very_ narrow use case and
      * should not be generally used for writing tests of Chisel.
      */
    def getModule[A <: RawModule](gen: => A): A = {
      var res: Any = null
      obj.elaborate {
        res = gen
        res.asInstanceOf[A]
      }
      res.asInstanceOf[A]
    }

  }

}
