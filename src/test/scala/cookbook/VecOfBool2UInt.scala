// See LICENSE for license details.

package cookbook

import chisel3._

/* ### How do I create a UInt from a Vec of Bool?
 *
 * Use the builtin function asUInt
 */
class VecOfBool2UInt extends CookbookTester(1) {
  // Example
  val vec = VecInit(true.B, false.B, true.B, true.B)
  val uint = vec.asUInt
  printf(p"$uint") // 13

  /* Test
   *
   * (remember leftmost Bool in Vec is low order bit)
   */
  assert(0xd.U === uint)
}

class VecOfBool2UIntSpec  extends CookbookSpec {
  "VecOfBool2UInt" should "work" in {
    assertTesterPasses { new VecOfBool2UInt }
  }
}
