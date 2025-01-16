// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chiselTests.{ChiselFlatSpec, FileCheck}

class PlaceholderSpec extends ChiselFlatSpec with FileCheck {

  "Placeholders" should "allow insertion of commands" in {

    class Foo extends RawModule {

      val placeholder = new Placeholder()

      val a = Wire(UInt(1.W))

      val b = placeholder.append {
        Wire(UInt(2.W))
      }

    }

    generateFirrtlAndFileCheck(new Foo) {
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

    generateFirrtlAndFileCheck(new Foo) {
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

    generateFirrtlAndFileCheck(new Foo) {
      """|CHECK: public module Foo :
         |CHECK:   skip
         |""".stripMargin
    }
  }

}
