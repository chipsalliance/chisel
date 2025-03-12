// SPDX-License-Identifier: Apache-2.0

package chiselTests.naming

import chisel3._
import chisel3.naming.{HasCustomIdentifier, IdentifierProposer}
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers

case class Blah(i: Int)

class Foo(val i: Int) extends HasCustomIdentifier {
  protected val customDefinitionIdentifierProposal = s"A_Different_Name_for_Foo"
}

class Baz(val i: Int)

class IdentifierProposerSpec extends AnyFunSpec with Matchers {
  it("(0): getProposal works on a variety of object types") {
    IdentifierProposer.getProposal("Hi") should be("Hi")
    IdentifierProposer.getProposal(1) should be("1")
    IdentifierProposer.getProposal(Blah(1)) should be("Blah_1")
    IdentifierProposer.getProposal(new Bundle {}) should be("AnonymousBundle")
    IdentifierProposer.getProposal(new Foo(1)) should be("A_Different_Name_for_Foo")
    IdentifierProposer.getProposal(new Baz(2)) should be("chiselTests_naming_Baz")
    IdentifierProposer.getProposal(Seq(1, 2, 3)) should be("1_2_3")
    IdentifierProposer.getProposal(List(1, 2, 3)) should be("1_2_3")
    IdentifierProposer.getProposal((1, 2)) should be("1_2")
    IdentifierProposer.getProposal(()) should be("")
  }
}
