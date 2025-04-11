// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.experimental.{inlineInstance, inlineInstanceAllowDedup}
import chisel3.testing.scalatest.FileCheck
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class AnnotationInlineSpec extends AnyFlatSpec with Matchers with FileCheck {
  class FooBundle extends Bundle {
    val in = Input(UInt(8.W))
    val out = Output(UInt(8.W))
  }

  class Foo extends RawModule {
    val io = IO(new FooBundle)
    io.out :<= io.in + 1.U
  }

  class Bar(inlineOneInstance: Boolean, allowDedup: Boolean) extends RawModule {
    val io = IO(new FooBundle)

    val inst0 = Module(new Foo)
    val inst1 = Module(new Foo)
    val inst2 = Module(new Foo)

    inst0.io.in :<= io.in
    inst1.io.in :<= inst0.io.out
    inst2.io.in :<= inst1.io.out
    io.out :<= inst2.io.out

    if (inlineOneInstance) {
      if (allowDedup) {
        inlineInstanceAllowDedup(inst1)
      } else {
        inlineInstance(inst1)
      }
    }
  }

  behavior of ("inlineInstance object")

  it should "not inline anything when disabled" in {
    ChiselStage
      .emitSystemVerilog(new Bar(false, false))
      .fileCheck()(
        """|CHECK:      module Foo(
           |
           |CHECK:      module Bar(
           |CHECK:        Foo inst0 (
           |CHECK-NEXT:     .io_in
           |CHECK-NEXT:     .io_out
           |CHECK-NEXT:   );
           |CHECK-NEXT:   Foo inst1 (
           |CHECK-NEXT:     .io_in
           |CHECK-NEXT:     .io_out
           |CHECK-NEXT:   );
           |CHECK-NEXT:   Foo inst2 (
           |CHECK-NEXT:     .io_in
           |CHECK-NEXT:     .io_out
           |CHECK-NEXT:   );
           |""".stripMargin
      )
  }

  it should "inline one Foo instance when enabled without dedup, and not block dedup of non-inlined Foo modules" in {
    ChiselStage
      .emitSystemVerilog(new Bar(true, false))
      .fileCheck()(
        """|CHECK:      module Foo(
           |
           |CHECK:      module Bar(
           |CHECK:        Foo inst0 (
           |CHECK-NEXT:     .io_in
           |CHECK-NEXT:     .io_out
           |CHECK-NEXT:   );
           |CHECK-NEXT:   Foo inst2 (
           |CHECK-NEXT:     .io_in (_inst0_io_out + 8'h1),
           |""".stripMargin
      )
  }

  it should "inline all Foo instances when enabled with dedup" in {
    ChiselStage
      .emitSystemVerilog(new Bar(true, true))
      .fileCheck()(
        """|CHECK-NOT:  module Foo(
           |
           |CHECK:      module Bar(
           |CHECK-NEXT:   input [7:0] io_in,
           |CHECK-NEXT:   output [7:0] io_out
           |CHECK-NEXT: );
           |CHECK-EMPTY:
           |CHECK-NEXT:   assign io_out = io_in + 8'h3;
           |CHECK-NEXT: endmodule
           |""".stripMargin
      )
  }
}
