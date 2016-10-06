// See LICENSE for license details.

package cookbook

import chisel3._

/* ### How do I create a Vec of Bools from a UInt?
 *
 * Use the builtin function [[chisel3.core.Bits.toBools]] to create a Scala Seq of Bool,
 * then wrap the resulting Seq in Vec(...)
 */
class UInt2VecOfBool extends CookbookTester(0) {
  // Example
  val uint = UInt(0xc)
  val vec = Vec(uint.toBools)
  printf(p"$vec") // Vec(0, 0, 1, 1)

  // Test
  assert(vec(0) === Bool(false))
  assert(vec(1) === Bool(false))
  assert(vec(2) === Bool(true))
  assert(vec(3) === Bool(true))
}

class UInt2VecOfBoolSpec extends CookbookSpec {
  "UInt2VecOfBool" should "work" in {
    assertTesterPasses { new UInt2VecOfBool }
  }
}
