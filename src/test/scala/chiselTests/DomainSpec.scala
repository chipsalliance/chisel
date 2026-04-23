// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.domain.{Domain, Field}
import chisel3.domains.ClockDomain
import chisel3.experimental.dataview._
import chisel3.properties.Property
import chisel3.testing.{FileCheck, HasTestingDirectory}
import chisel3.testing.scalatest.TestingDirectory
import chisel3.util.experimental.InlineInstance
import circt.stage.ChiselStage
import java.io.ByteArrayOutputStream
import java.nio.file.Files
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import scala.annotation.nowarn
import scala.sys.process._

class DomainSpec extends AnyFlatSpec with Matchers with FileCheck with TestingDirectory {

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
         |CHECK-NEXT:     source : String
         |CHECK-NEXT:     relationship : String
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

    @nowarn("cat=deprecation")
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
    }.getMessage should include("Unable to associate data")

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
    }.getMessage should include("cannot associate a port or wire with zero domains")

  }

  it should "work on views" in {
    class Foo extends RawModule {
      val A = IO(Input(ClockDomain.Type()))
      val a = IO(Input(Bool()))
      associate(a.viewAs[Bool], A)
    }

    ChiselStage.emitCHIRRTL(new Foo).fileCheck() {
      """|CHECK: module Foo :
         |CHECK:   input a : UInt<1> domains [A]
         |""".stripMargin
    }
  }

  it should "work with FlatIO" in {
    class Foo extends RawModule {
      val A = IO(Input(ClockDomain.Type()))
      val io = FlatIO(new Bundle {
        val a = Input(Bool())
      })
      associate(io.a, A)
    }

    ChiselStage.emitCHIRRTL(new Foo).fileCheck() {
      """|CHECK: module Foo :
         |CHECK:   input a : UInt<1> domains [A]
         |""".stripMargin
    }
  }

  it should "allow for multiple ports" in {
    class Foo extends RawModule {
      val A = IO(Input(ClockDomain.Type()))
      val a, b = IO(Input(Bool()))
      associate(Seq(a, b), A)
    }

    ChiselStage.emitCHIRRTL(new Foo).fileCheck() {
      """|CHECK:   input a : UInt<1> domains [A]
         |CHECK:   input b : UInt<1> domains [A]
         |""".stripMargin
    }
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

  behavior of "domain subfield access"

  it should "allow accessing fields of a domain port" in {

    class Foo extends RawModule {
      val A = IO(Input(ClockDomain.Type()))
      val nameOut = IO(Output(Property[String]()))
      val sourceOut = IO(Output(Property[String]()))

      nameOut := A.field.name
      sourceOut := A.field.source
    }

    ChiselStage.emitCHIRRTL(new Foo).fileCheck() {
      """|CHECK: module Foo :
         |CHECK:   input A : Domain of ClockDomain
         |CHECK:   output nameOut : String
         |CHECK:   output sourceOut : String
         |CHECK:   propassign nameOut, A.name
         |CHECK:   propassign sourceOut, A.source
         |""".stripMargin
    }

  }

  behavior of "domain instantiation"

  it should "allow instantiating a synchronous domain" in {

    class Foo extends RawModule {
      val A = IO(Input(ClockDomain.Type()))
      val B = IO(Output(ClockDomain.Type()))

      val b = ClockDomain.synchronous(A, "_1to4")
      domain.define(B, b)
    }

    ChiselStage.emitCHIRRTL(new Foo).fileCheck() {
      """|CHECK: module Foo :
         |CHECK:   input A : Domain of ClockDomain
         |CHECK:   output B : Domain of ClockDomain
         |CHECK:   wire [[name:.+]] : String
         |CHECK:   propassign [[name]], string_concat(A.name, String("_1to4"))
         |CHECK:   domain [[b:.*]] of ClockDomain([[name]], A.name, String("synchronous"))
         |CHECK:   domain_define B = [[b]]
         |""".stripMargin
    }

  }

  it should "allow instantiating a rational domain" in {

    class Foo extends RawModule {
      val A = IO(Input(ClockDomain.Type()))
      val B = IO(Output(ClockDomain.Type()))

      val b = ClockDomain.rational(A, "_2to3")
      domain.define(B, b)
    }

    ChiselStage.emitCHIRRTL(new Foo).fileCheck() {
      """|CHECK: module Foo :
         |CHECK:   input A : Domain of ClockDomain
         |CHECK:   output B : Domain of ClockDomain
         |CHECK:   wire [[name:.+]] : String
         |CHECK:   propassign [[name]], string_concat(A.name, String("_2to3"))
         |CHECK:   domain [[b:.*]] of ClockDomain([[name]], A.name, String("rational"))
         |CHECK:   domain_define B = [[b]]
         |""".stripMargin
    }

  }

  it should "error when property type doesn't match field type" in {

    object TestDomain extends Domain {
      override def fields = Seq(
        ("name", Field.String),
        ("value", Field.Integer),
        ("flag", Field.Boolean)
      )
      def apply(): chisel3.domain.Type = TestDomain(Property("test"), Property("wrong"), Property(true))
    }

    class WrongTypeModule extends RawModule {
      // Wrong type for second field: String instead of Integer
      val bad = TestDomain()
    }

    val exception = intercept[ChiselException] {
      ChiselStage.elaborate(new WrongTypeModule, Array("--throw-on-first-error"))
    }
    exception.getMessage should include("field 'value' expects Property[Int] but got Property[String]")

  }

  it should "error when property count doesn't match field count" in {

    object TestDomain extends Domain {
      override def fields = Seq(
        ("name", Field.String),
        ("value", Field.Integer),
        ("flag", Field.Boolean)
      )
      def apply(): chisel3.domain.Type = TestDomain(Property("test"))
    }

    class WrongCountModule extends RawModule {
      // Only one property when three are expected
      val bad = TestDomain()
    }

    val exception = intercept[ChiselException] {
      ChiselStage.elaborate(new WrongCountModule, Array("--throw-on-first-error"))
    }
    exception.getMessage should include("requires 3 properties but got 1")

  }

  behavior of "Wire domain associations"

  it should "allow associating a wire with a domain" in {
    class Foo extends RawModule {
      val A = IO(Input(ClockDomain.Type()))
      val w = Wire(UInt(8.W))
      associate(w, A)
    }

    ChiselStage.emitCHIRRTL(new Foo).fileCheck() {
      """|CHECK: module Foo :
         |CHECK:   input A : Domain of ClockDomain
         |CHECK:   wire w : UInt<8> domains [A]
         |""".stripMargin
    }
  }

  it should "allow associating a wire with multiple domains" in {
    object PowerDomain extends Domain {
      override def fields = Seq(
        ("name", Field.String),
        ("voltage", Field.Integer)
      )
    }

    class Foo extends RawModule {
      val A = IO(Input(ClockDomain.Type()))
      val B = IO(Input(PowerDomain.Type()))
      val w = Wire(UInt(16.W))
      associate(w, A, B)
    }

    ChiselStage.emitCHIRRTL(new Foo).fileCheck() {
      """|CHECK: module Foo :
         |CHECK:   wire w : UInt<16> domains [A, B]
         |""".stripMargin
    }
  }

  it should "error if associating a wire from another module" in {
    class Bar extends RawModule {
      val A = IO(Input(ClockDomain.Type()))
      val w = Wire(UInt(8.W))
    }

    class Foo extends RawModule {
      val bar = Module(new Bar)
      val A = IO(Input(ClockDomain.Type()))
      // Try to associate a wire from another module
      associate(bar.w, A)
    }

    intercept[ChiselException] {
      ChiselStage.elaborate(new Foo, Array("--throw-on-first-error"))
    }.getMessage should include("Unable to associate")
  }

  it should "work with multiple wires" in {
    class Foo extends RawModule {
      val A = IO(Input(ClockDomain.Type()))
      val w1 = Wire(UInt(8.W))
      val w2 = Wire(UInt(8.W))
      associate(Seq(w1, w2), A)
    }

    ChiselStage.emitCHIRRTL(new Foo).fileCheck() {
      """|CHECK:   wire w1 : UInt<8> domains [A]
         |CHECK:   wire w2 : UInt<8> domains [A]
         |""".stripMargin
    }
  }

  it should "error if given zero domains" in {
    class Foo extends RawModule {
      val w = Wire(UInt(8.W))
      associate(w)
    }

    intercept[IllegalArgumentException] {
      ChiselStage.elaborate(new Foo)
    }.getMessage should include("cannot associate a port or wire with zero domains")
  }

  behavior of ("Domain assertions")

  trait Domains {
    def A: domain.Type
    def B: domain.Type
  }

  object Domains {

    /** Return two asynchronous (unrelated) domains. */
    def asynchronous() = new Domains {
      override val A = ClockDomain("A")
      override val B = ClockDomain("B")
    }

    /** Return two domains, the second of which is synchronous to the first */
    def synchronous() = new Domains {
      override val A = ClockDomain("A")
      override val B = ClockDomain.synchronous(A, "_1to2")
    }

    /** Return two domains, the second of which is rationally related to the first. */
    def rational() = new Domains {
      override val A = ClockDomain("A")
      override val B = ClockDomain.rational(A, "_2to3")
    }
  }

  abstract class Crossing extends RawModule {
    val A, B = IO(Input(ClockDomain.Type()))
  }

  object Crossing {

    /** Cross between any two domains. */
    class Asynchronous extends Crossing

    /** Cross between synchronous domains. */
    class Synchronous extends Crossing {
      (A.field.source.asInstanceOf[Property[String]] === B.field.source.asInstanceOf[Property[String]])
        .assert(s"clock domains 'A' and 'B' must have the same source")
      (A.field.relationship.asInstanceOf[Property[String]] === ClockDomain.Relationship.Synchronous.toProperty())
        .assert(s"clock domain 'A' must have a synchronous relationship")
      (B.field.relationship.asInstanceOf[Property[String]] === ClockDomain.Relationship.Synchronous.toProperty())
        .assert(s"clock domain 'B' must have a synchronous relationship")
    }

    /** Cross between synchronous or rational domains.
      *
      * @note This needs to be updated once boolean or is a supported property
      * expression.
      */
    class Rational extends Crossing {
      (A.field.source.asInstanceOf[Property[String]] === B.field.source.asInstanceOf[Property[String]])
        .assert(s"clock domains 'A' and 'B' must have the same source")
      (A.field.relationship.asInstanceOf[Property[String]] === ClockDomain.Relationship.Synchronous.toProperty())
        .assert(s"clock domain 'A' must have a rational relationship")
      (B.field.relationship.asInstanceOf[Property[String]] === ClockDomain.Relationship.Rational.toProperty())
        .assert(s"clock domain 'B' must have a rational relationship")
    }

  }

  class CrossingTestHarness(
    crossingGen: () => Crossing,
    domainsGen:  () => Domains
  ) extends RawModule {
    private val crossing = Module(crossingGen())
    private val domains = domainsGen()
    domain.define(crossing.A, domains.A)
    domain.define(crossing.B, domains.B)
  }

  case class CrossingTest(name: String, genHarness: () => CrossingTestHarness, result: Either[String, Unit]) {

    private val description = {
      (result match {
        case Right(_) => "pass"
        case Left(_)  => "error"
      }) ++ s" for $name"
    }

    def test() = they should description in {
      val dir = implicitly[HasTestingDirectory].getDirectory
      Files.createDirectories(dir)

      val finalMlir = dir.resolve("final.mlir").toString
      ChiselStage.emitSystemVerilog(
        genHarness(),
        firtoolOpts = Array("-domain-mode=infer-all", "-output-final-mlir", finalMlir)
      )

      val stderrStream = new ByteArrayOutputStream
      val logger = ProcessLogger(_ => (), line => { stderrStream.write(line.getBytes); stderrStream.write('\n') })
      try {
        val exitCode: Int = Seq("domaintool", "--module", "CrossingTestHarness", finalMlir).!(logger)
        result match {
          case Right(_) =>
            exitCode should be(0)
          case Left(error) =>
            exitCode should not be (0)
            stderrStream.toString should include(error)
        }

      } catch {
        case a: java.io.IOException if a.getMessage().startsWith("Cannot run program") =>
          info("skipped as 'domaintool' is not available")
      }
    }

  }

  Seq(
    // ---------------------------------- Synchronous crossing
    CrossingTest(
      "a synchronous crossing with synchronous clocks",
      () => new CrossingTestHarness(() => new Crossing.Synchronous, Domains.synchronous),
      Right(())
    ),
    CrossingTest(
      "a synchronous crossing with rational clocks",
      () => new CrossingTestHarness(() => new Crossing.Synchronous, Domains.rational),
      Left("clock domain 'B' must have a synchronous relationship")
    ),
    CrossingTest(
      "a synchronous crossing with asynchronous clocks",
      () => new CrossingTestHarness(() => new Crossing.Synchronous, Domains.asynchronous),
      Left("clock domains 'A' and 'B' must have the same source")
    ),
    // ---------------------------------- Rational crossing
    // TODO: This currently fails as we need a property boolean or to express
    // that a rational crossing can handle both synchronous and rational
    // relationships.
    // CrossingTest(
    //   "a rationl crossing with synchronous clocks",
    //   () => new CrossingTestHarness(() => new Crossing.Rational, Domains.synchronous),
    //   Right(())
    // ),
    CrossingTest(
      "a rationl crossing with rational clocks",
      () => new CrossingTestHarness(() => new Crossing.Rational, Domains.rational),
      Right(())
    ),
    CrossingTest(
      "a rational crossing with asynchronous clocks",
      () => new CrossingTestHarness(() => new Crossing.Rational, Domains.asynchronous),
      Left("clock domains 'A' and 'B' must have the same source")
    ),
    // ---------------------------------- Asynchronous crossing
    CrossingTest(
      "an aasynchronous crossing with synchronous clocks",
      () => new CrossingTestHarness(() => new Crossing.Asynchronous, Domains.synchronous),
      Right(())
    ),
    CrossingTest(
      "an asynchronous crossing with rational clocks",
      () => new CrossingTestHarness(() => new Crossing.Asynchronous, Domains.rational),
      Right(())
    ),
    CrossingTest(
      "an asynchronous crossing with asynchronous clocks",
      () => new CrossingTestHarness(() => new Crossing.Asynchronous, Domains.asynchronous),
      Right(())
    )
  ).foreach { _.test() }

}
