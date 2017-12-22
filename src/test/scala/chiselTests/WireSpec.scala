// See LICENSE for license details.

package chiselTests

import chisel3._

class WireSpec extends ChiselFlatSpec {
  "WireInit.apply" should "work" in {
    assertCompiles("WireInit(UInt(4.W), 2.U)")
  }
  it should "allow DontCare" in {
    assertCompiles("WireInit(UInt(4.W), DontCare)")
  }
  it should "not allow DontCare to affect type inference" in {
    assertCompiles("val x: UInt = WireInit(UInt(4.W), DontCare)")
  }
  it should "not allow init argument to affect type inference" in {
    assertDoesNotCompile("val x: UInt = WireInit(UInt(4.W), 2.S)")
  }
}
