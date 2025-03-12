// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.testing.scalatest.FileCheck
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class PlaceholderSpec extends AnyFlatSpec with Matchers with FileCheck {

  "Placeholders" should "allow insertion of commands" in {

    class Foo extends RawModule {

      val placeholder = new Placeholder()

      val a = Wire(UInt(1.W))

      val b = placeholder.append {
        Wire(UInt(2.W))
      }

    }

    ChiselStage.emitCHIRRTL(new Foo).fileCheck() {
      s"""|CHECK:      wire b : UInt<2>
          |CHECK-NEXT: wire a : UInt<1>
          |""".stripMargin
    }

  }

  they should "be capable of being nested" in {

    class Foo extends RawModule {

      val placeholder_1 = new Placeholder()
      val placeholder_2 = placeholder_1.append(new Placeholder())

      val a = Wire(UInt(1.W))

      val b = placeholder_1.append {
        Wire(UInt(2.W))
      }

      val c = placeholder_2.append {
        Wire(UInt(3.W))
      }

      val d = placeholder_1.append {
        Wire(UInt(4.W))
      }

    }

    ChiselStage.emitCHIRRTL(new Foo).fileCheck() {
      s"""|CHECK:      wire c : UInt<3>
          |CHECK-NEXT: wire b : UInt<2>
          |CHECK-NEXT: wire d : UInt<4>
          |CHECK-NEXT: wire a : UInt<1>
          |""".stripMargin
    }

  }

  they should "emit no statements if empty" in {

    class Foo extends RawModule {
      val a = new Placeholder()
    }

    ChiselStage.emitCHIRRTL(new Foo).fileCheck() {
      """|CHECK: public module Foo :
         |CHECK:   skip
         |""".stripMargin
    }
  }

  they should "allow constructing hardware in a parent" in {

    class Bar(placeholder: Placeholder, a: Bool) extends RawModule {

      placeholder.append {
        val b = Wire(Bool())
        b :<= a
      }

    }

    class Foo extends RawModule {

      val a = Wire(Bool())
      val placeholder = new Placeholder

      val bar = Module(new Bar(placeholder, a))

    }

    ChiselStage.emitCHIRRTL(new Foo).fileCheck() {
      """|CHECK:      module Bar :
         |CHECK-NOT:    {{wire|connect}}
         |
         |CHECK:      module Foo :
         |CHECK:        wire a : UInt<1>
         |CHECK-NEXT:   wire b : UInt<1>
         |CHECK-NEXT:   connect b, a
         |CHECK-NEXT:   inst bar of Bar
         |""".stripMargin
    }

  }

  // TODO: This test can be changed to pass in the future in support of advanced
  // APIs like Providers or an improved BoringUtils.
  they should "error if constructing hardware in a child" in {

    class Bar extends RawModule {

      val a = Wire(Bool())
      val placeholder = new Placeholder()

    }

    class Foo extends RawModule {

      val bar = Module(new Bar)

      bar.placeholder.append {
        val b = Wire(Bool())
        b :<= bar.a
      }

    }

    intercept[IllegalArgumentException] { ChiselStage.emitCHIRRTL(new Foo) }.getMessage() should include(
      "Can't write to module after module close"
    )

  }

}
