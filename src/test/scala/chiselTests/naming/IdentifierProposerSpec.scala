// SPDX-License-Identifier: Apache-2.0

package chiselTests
package naming

import chisel3._
import chisel3.naming.IdentifierProposer

case class Blah(i: Int)
class Foo(val i: Int)
object Foo {
  implicit val fooProposer = new IdentifierProposer[Foo] {
    def propose(foo: Foo): String = s"Foo(${foo.i})"
  }
}
class Baz(val i: Int)

class IdentifierProposerSpec extends ChiselFunSpec with Utils {
  it("(0): getProposal works on a variety of object types") {
    assert(IdentifierProposer.getProposal("Hi") == "Hi")
    assert(IdentifierProposer.getProposal(1) == "1")
    assert(IdentifierProposer.getProposal(Blah(1)) == "Blah_1")
    assert(IdentifierProposer.getProposal(new Bundle {}) == "AnonymousBundle")
    assert(IdentifierProposer.getProposal(new Foo(1)) == "Foo_1")
    assert(IdentifierProposer.getProposal(new Baz(2)) == "chiselTests_naming_Baz")
    assert(IdentifierProposer.getProposal(Seq(1, 2, 3)) == "1_2_3")
    assert(IdentifierProposer.getProposal(List(1, 2, 3)) == "1_2_3")
    assert(IdentifierProposer.getProposal((1, 2)) == "1_2")
  }
}
