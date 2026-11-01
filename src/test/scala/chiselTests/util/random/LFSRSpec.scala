// SPDX-License-Identifier: Apache-2.0

package chiselTests.util.random

import chisel3.util.random.LFSR
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class LFSRSpec extends AnyFlatSpec with Matchers {

  behavior.of("LFSR.tapsMaxPeriod")

  it should "only contain taps in the range 1 to width" in {
    for {
      (width, sets) <- LFSR.tapsMaxPeriod
      set <- sets
    } {
      withClue(s"width $width, taps $set: ") {
        set.foreach(tap => tap should ((be >= 1).and(be <= width)))
      }
    }
  }
}
