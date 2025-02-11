// SPDX-License-Identifier: Apache-2.0

package chiselTests.naming

import chisel3._
import chisel3.naming.{HasCustomIdentifier, IdentifierProposer}
import chiselTests.ChiselFunSpec

case class Blah(i: Int)

class Foo(val i: Int) extends HasCustomIdentifier {
  protected val customDefinitionIdentifierProposal = s"A_Different_Name_for_Foo"
}

class Baz(val i: Int)

class IdentifierProposerSpec extends ChiselFunSpec {
  it("(0): getProposal works on a variety of object types") {
    assert(IdentifierProposer.getProposal("Hi") == "Hi")
    assert(IdentifierProposer.getProposal(1) == "1")
    assert(IdentifierProposer.getProposal(Blah(1)) == "Blah_1")
    assert(IdentifierProposer.getProposal(new Bundle {}) == "AnonymousBundle")
    assert(IdentifierProposer.getProposal(new Foo(1)) == "A_Different_Name_for_Foo")
    assert(IdentifierProposer.getProposal(new Baz(2)) == "chiselTests_naming_Baz")
    assert(IdentifierProposer.getProposal(Seq(1, 2, 3)) == "1_2_3")
    assert(IdentifierProposer.getProposal(List(1, 2, 3)) == "1_2_3")
    assert(IdentifierProposer.getProposal((1, 2)) == "1_2")
    assert(IdentifierProposer.getProposal(()) == "")
  }
}
