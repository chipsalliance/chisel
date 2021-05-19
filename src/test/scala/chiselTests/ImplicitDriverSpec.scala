// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3.util.experimental.ImplicitDriver
import org.scalatest.flatspec.AnyFlatSpec

class ImplicitDriverSpec extends AnyFlatSpec {
  "implicit driver" should "emit verilog without error" in {
    (new GCD).emitVerilog
  }
  "implicit driver" should "emit firrtl without error" in {
    (new GCD).emitFirrtl
  }
  "implicit driver" should "emit chirrtl without error" in {
    (new GCD).emitChirrtl
  }
  "implicit driver" should "emit system verilog without error" in {
    (new GCD).emitSystemVerilog
  }
}
