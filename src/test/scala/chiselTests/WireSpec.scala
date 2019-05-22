// See LICENSE for license details.

package chiselTests

import chisel3._

class WireSpec extends ChiselFlatSpec {
  "WireDefault.apply" should "work" in {
    assertCompiles("WireDefault(UInt(4.W), 2.U)")
  }
  it should "allow DontCare" in {
    assertCompiles("WireDefault(UInt(4.W), DontCare)")
  }
  it should "not allow DontCare to affect type inference" in {
    assertCompiles("val x: UInt = WireDefault(UInt(4.W), DontCare)")
  }
  it should "not allow init argument to affect type inference" in {
    assertDoesNotCompile("val x: UInt = WireDefault(UInt(4.W), 2.S)")
  }
}
