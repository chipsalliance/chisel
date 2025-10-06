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
      val a = IO(Input(domain.Type(ClockDomain)))
      val b = IO(Input(domain.Type(UserDefined)))
    }

    ChiselStage.emitCHIRRTL(new Foo).fileCheck() {
      """|CHECK:      circuit Foo :
         |CHECK:        domain ClockDomain :
         |CHECK-NEXT:   domain UserDefined :
         |
         |CHECK:        public module Foo :
         |CHECK-NEXT:     input a : Domain of ClockDomain
         |CHECK-NEXT:     input b : Domain of UserDefined
         |""".stripMargin
    }

  }

}
