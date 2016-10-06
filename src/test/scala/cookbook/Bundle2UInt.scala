// See LICENSE for license details.

package cookbook

import chisel3._

/* ### How do I create a UInt from an instance of a Bundle?
 *
 * Call asUInt on the Bundle instance
 */
class Bundle2UInt extends CookbookTester(0) {
  // Example
  class MyBundle extends Bundle {
    val foo = UInt(width = 4)
    val bar = UInt(width = 4)
  }
  val bundle = Wire(new MyBundle)
  bundle.foo := UInt(0xc)
  bundle.bar := UInt(0x3)
  val uint = bundle.asUInt
  printf(p"$uint") // 195

  // Test
  assert(uint === UInt(0xc3))
}

class Bundle2UIntSpec  extends CookbookSpec {
  "Bundle2UInt" should "work" in {
    assertTesterPasses { new Bundle2UInt }
  }
}
