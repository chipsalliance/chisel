// See LICENSE for license details.

package cookbook

import chisel3._

/* ### How do I create a Bundle from a UInt?
 *
 * On an instance of the Bundle, call the method fromBits with the UInt as the argument
 */
class UInt2Bundle extends CookbookTester(1) {
  // Example
  class MyBundle extends Bundle {
    val foo = UInt(4.W)
    val bar = UInt(4.W)
  }
  val uint = 0xb4.U
  val bundle = uint.asTypeOf(new MyBundle)
  printf(p"$bundle") // Bundle(foo -> 11, bar -> 4)

  // Test
  assert(bundle.foo === 0xb.U)
  assert(bundle.bar === 0x4.U)
}

class UInt2BundleSpec extends CookbookSpec {
  "UInt2Bundle" should "work" in {
    assertTesterPasses { new UInt2Bundle }
  }
}
