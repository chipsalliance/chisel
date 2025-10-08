// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.domain.Domain
import chisel3.domains.ClockDomain
import chisel3.testing.FileCheck
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class DomainSpec extends AnyFlatSpec with Matchers with FileCheck {

  behavior of "Domains"

  they should "emit FIRRTL for internal and user-defined domains" in {

    object UserDefined extends Domain

    class Foo extends RawModule {
      val A = IO(Input(domain.Type(ClockDomain)))
      val B = IO(Input(domain.Type(UserDefined)))
      val a = IO(Input(Bool()))
      val b = IO(Input(Bool()))

      associate(a, A)
      associate(b, B)
    }

    val chirrtl = ChiselStage.emitCHIRRTL(new Foo, args = Array("--full-stacktrace"))
    println(chirrtl)

    chirrtl.fileCheck() {
      """|CHECK:      circuit Foo :
         |CHECK:        domain ClockDomain :
         |CHECK-NEXT:   domain UserDefined :
         |
         |CHECK:        public module Foo :
         |CHECK-NEXT:     input A : Domain of ClockDomain
         |CHECK-NEXT:     input B : Domain of UserDefined
         |CHECK-NEXT:     input a : UInt<1> domains [A]
         |CHECK-NEXT:     input b : UInt<1> domains [B]
         |""".stripMargin
    }

  }

}
