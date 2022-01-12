// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._

class ImplicitConversionsSpec extends ChiselFlatSpec {
  ".data on arbitrary Data objects" should "not work" in {
    assertTypeError("UInt(8.W).data")
    assertTypeError("8.S.data")
    assertTypeError("(new Bundle {}).data")
    assertTypeError("VecInit(1.U).data")
  }

  ".target on arbitrary Data objects" should "not work" in {
    assertTypeError("UInt(8.W).target")
    assertTypeError("8.S.target")
    assertTypeError("(new Bundle {}).target")
    assertTypeError("VecInit(1.U).target")
  }

  ".x on Strings and Numerical values" should "not work" in {
    assertTypeError("3.x")
    assertTypeError("3L.x")
    assertTypeError("BigInt(-4).x")
    assertTypeError("false.x")
    assertTypeError(""""a".x""")
  }

  ".bigint on Strings and Numerical values" should "not work" in {
    assertTypeError("3.bigint")
    assertTypeError("3L.bigint")
    assertTypeError("BigInt(-4).bigint")
    assertTypeError("false.bigint")
    assertTypeError(""""a".bigint""")
  }

  ".target on DecoupledIO" should "not work" in {
    import chisel3.util._
    assertTypeError("Decoupled(UInt(8.W)).target")
  }

  "X.B for X not in [0,1]" should "throw an exception, even outside hardware context" in {
    a[ChiselException] should be thrownBy {
      65.B
    }
  }
}
