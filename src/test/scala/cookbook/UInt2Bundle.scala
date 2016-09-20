// See LICENSE for license details.

package cookbook

import chisel3._

/* ### How do I create a Bundle from a UInt?
 *
 * On an instance of the Bundle, call the method fromBits with the UInt as the argument
 */
class UInt2Bundle extends CookbookTester(0) {
  // Example
  class MyBundle extends Bundle {
    val foo = UInt(width = 4)
    val bar = UInt(width = 4)
  }
  val uint = UInt(0xb4)
  val bundle = (new MyBundle).fromBits(uint)
  printf(p"$bundle") // Bundle(foo -> 11, bar -> 4)

  // Test
  assert(bundle.foo === UInt(0xb))
  assert(bundle.bar === UInt(0x4))
}

class UInt2BundleSpec extends CookbookSpec {
  "UInt2Bundle" should "work" in {
    assertTesterPasses { new UInt2Bundle }
  }
}
