// See LICENSE for license details.

package cookbook

import chisel3._

/* ### How do I create a Vec of Bools from a UInt?
 *
 * Use the builtin function [[chisel3.core.Bits.toBools]] to create a Scala Seq of Bool,
 * then wrap the resulting Seq in Vec(...)
 */
class UInt2VecOfBool extends CookbookTester(1) {
  // Example
  val uint = 0xc.U
  val vec = VecInit(uint.toBools)
  printf(p"$vec") // Vec(0, 0, 1, 1)

  // Test
  assert(vec(0) === false.B)
  assert(vec(1) === false.B)
  assert(vec(2) === true.B)
  assert(vec(3) === true.B)
}

class UInt2VecOfBoolSpec extends CookbookSpec {
  "UInt2VecOfBool" should "work" in {
    assertTesterPasses { new UInt2VecOfBool }
  }
}
