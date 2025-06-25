// SPDX-License-Identifier: Apache-2.0

package chiselTests.experimental

import chisel3._
import chisel3.experimental.firrtlComment
import chisel3.testing.scalatest.FileCheck
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec

class FirrtlCommentSpec extends AnyFlatSpec with FileCheck {

  behavior.of("firrtlComment")

  it should "enable leaving comments in the firrtl text" in {
    class Foo extends Module {
      val in = IO(Input(UInt(8.W)))
      val out = IO(Output(UInt(8.W)))
      firrtlComment("This is a comment")
      out := in
      firrtlComment("This is another comment")
    }
    ChiselStage
      .emitCHIRRTL(new Foo)
      .fileCheck()(
        s"""|CHECK:      ; This is a comment
            |CHECK-NEXT: connect out, in
            |CHECK-NEXT: ; This is another comment
            |""".stripMargin
      )
  }

  it should "support empty comments" in {
    class Foo extends Module {
      val in = IO(Input(UInt(8.W)))
      val out = IO(Output(UInt(8.W)))
      firrtlComment("")
      out := in
    }
    ChiselStage
      .emitCHIRRTL(new Foo)
      .fileCheck()(
        s"""|CHECK:      ;
            |CHECK-NEXT: connect out, in
            |""".stripMargin
      )
  }

  it should "support multi-line comments" in {
    class Foo extends Module {
      val in = IO(Input(UInt(8.W)))
      val out = IO(Output(UInt(8.W)))
      firrtlComment("This is a comment\n  This is another comment with spacing.")
      out := in
    }
    ChiselStage
      .emitCHIRRTL(new Foo)
      .fileCheck()(
        s"""|CHECK:      ; This is a comment
            |CHECK-NEXT: ;   This is another comment with spacing.
            |CHECK-NEXT: connect out, in
            |""".stripMargin
      )
  }
}
