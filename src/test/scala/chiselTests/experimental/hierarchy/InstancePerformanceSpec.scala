// SPDX-License-Identifier: Apache-2.0

package chiselTests.experimental.hierarchy

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, Definition, Instance}
import circt.stage.ChiselStage
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers

class InstancePerformanceSpec extends AnyFunSpec with Matchers {
  it("should look up 20000 module definitions without quadratic equality comparisons") {
    val count = 20000
    var comparisons = 0L
    var hashes = 0L

    @instantiable
    class CountedModule(index: Int) extends RawModule {
      override def desiredName = s"CountedModule_$index"
      override def equals(that: Any): Boolean = {
        comparisons += 1
        super.equals(that)
      }
      override def hashCode: Int = {
        hashes += 1
        super.hashCode
      }
    }

    class Top extends RawModule {
      val definitions = Vector.tabulate(count)(i => Definition(new CountedModule(i)))
      // Re-registering the shared registry after each Definition would also be quadratic.
      hashes should be <= (count.toLong * 20)
      comparisons = 0
      // Instantiate every definition twice, including definitions near the end of the registry.
      val instances = definitions.map(Instance(_))
      val repeated = definitions.map(Instance(_))
    }

    val chirrtl = ChiselStage.emitCHIRRTL(new Top)
    // Count work instead of time so this remains reliable on slow or busy CI machines.
    // The old linear search performs approximately count * count comparisons here.
    comparisons should be <= (count.toLong * 10)
    // Also check for non-quadratic number of hashes
    hashes should be <= (count.toLong * 10)
    "(?m)^  module CountedModule_".r.findAllMatchIn(chirrtl).length should be(count)
    "(?m)^    inst .* of CountedModule_".r.findAllMatchIn(chirrtl).length should be(count * 2)
  }
}
