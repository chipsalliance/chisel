// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.domain.{Domain, Field}
import chisel3.domains.ClockDomain
import chisel3.experimental.ExtModule
import chisel3.testing.FileCheck
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class DomainSpec extends AnyFlatSpec with Matchers with FileCheck {

  behavior of "Domains"

  they should "emit FIRRTL for internal and user-defined domains" in {

    object PowerDomain extends Domain {

      override def fields = Seq(
        ("name", Field.String),
        ("voltage", Field.Integer),
        ("alwaysOn", Field.Boolean)
      )

    }

    class Foo extends RawModule {
      val A = IO(Input(ClockDomain.Type()))
      val B = IO(Input(PowerDomain.Type()))
      val a = IO(Input(Bool()))
      val b = IO(Output(Bool()))

      associate(a, A)
      associate(b, B)

      b :<= domain.unsafeCast(a, B)
    }

    ChiselStage.emitCHIRRTL(new Foo).fileCheck() {
      """|CHECK:      circuit Foo :
         |CHECK:        domain ClockDomain :
         |CHECK-NEXT:     name : String
         |CHECK-NEXT:     period : Integer
         |
         |CHECK:        domain PowerDomain :
         |CHECK-NEXT:     name : String
         |CHECK-NEXT:     voltage : Integer
         |CHECK-NEXT:     alwaysOn : Bool
         |
         |CHECK:        public module Foo :
         |CHECK-NEXT:     input A : Domain of ClockDomain
         |CHECK-NEXT:     input B : Domain of PowerDomain
         |CHECK-NEXT:     input a : UInt<1> domains [A]
         |CHECK-NEXT:     output b : UInt<1> domains [B]
         |
         |CHECK:          node [[cast:.*]] = unsafe_domain_cast(a, B)
         |CHECK-NEXT:     connect b, [[cast]]
         |""".stripMargin
    }

  }

  they should "fail to work with blackboxes" in {

    class Bar extends BlackBox {
      val io = IO {
        new Bundle {
          val A = Input(ClockDomain.Type())
          val a = Input(UInt(1.W))
        }
      }
      associate(io.a, io.A)
    }

    class Foo extends Module {
      private val bar = Module(new Bar)
    }

    intercept[ChiselException] {
      ChiselStage.elaborate(new Foo, Array("--throw-on-first-error"))
    }.getMessage should include("Unable to associate port")

  }

  they should "work for extmodules" in {

    class Bar extends ExtModule {
      val A = IO(Input(ClockDomain.Type()))
      val a = IO(Input(UInt(1.W)))
      associate(a, A)
    }

    class Foo extends RawModule {
      private val bar = Module(new Bar)
    }

    ChiselStage.emitCHIRRTL(new Foo).fileCheck() {
      """|CHECK:      extmodule Bar :
         |CHECK-NEXT:   input A : Domain of ClockDomain
         |CHECK-NEXT:   input a : UInt<1> domains [A]
         |""".stripMargin
    }

  }

  they should "be capable of being forwarded with the domain define operation" in {

    class Foo extends RawModule {
      val a = IO(Input(ClockDomain.Type()))
      val b = IO(Output(ClockDomain.Type()))

      domain.define(b, a)
    }

    ChiselStage.emitCHIRRTL(new Foo).fileCheck() {
      """|CHECK: module Foo :
         |CHECK:   domain_define b = a
         |""".stripMargin
    }

  }

  behavior of "The associate method"

  it should "error if given zero arguments" in {

    class Foo extends RawModule {
      val a = IO(Input(Bool()))
      associate(a)
    }

    intercept[IllegalArgumentException] {
      ChiselStage.elaborate(new Foo)
    }.getMessage should include("cannot associate a port with zero domains")

  }

  behavior of "unsafe_domain_cast"

  it should "work for zero, one, and two casts" in {

    object DomainA extends Domain
    object DomainB extends Domain

    class Foo extends RawModule {
      val A = IO(Input(DomainA.Type()))
      val B = IO(Input(DomainB.Type()))
      val in = IO(Input(Bool()))

      val out = IO(Output(Bool()))

      out :<= domain.unsafeCast(in)
      out :<= domain.unsafeCast(in, A)
      out :<= domain.unsafeCast(in, B)
      out :<= domain.unsafeCast(in, A, B)
    }

    ChiselStage.emitCHIRRTL(new Foo).fileCheck() {
      """|CHECK: unsafe_domain_cast(in)
         |CHECK: unsafe_domain_cast(in, A)
         |CHECK: unsafe_domain_cast(in, B)
         |CHECK: unsafe_domain_cast(in, A, B)
         |""".stripMargin
    }

  }

}
