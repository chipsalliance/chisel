// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.testing.scalatest.FileCheck
import chisel3.experimental.hierarchy.{instantiable, Definition, Instance}
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class PublicModuleSpec extends AnyFlatSpec with Matchers with FileCheck {

  @instantiable
  class Grault extends RawModule with Public {
    override def isPublic = false
  }

  @instantiable
  class Corge extends RawModule with Public

  @instantiable
  class Quz extends RawModule

  class Qux extends RawModule

  class Baz extends RawModule with Public {
    val qux = Module(new Qux)
    override def isPublic = false
  }

  class Bar extends RawModule with Public {
    val baz = Module(new Baz)
  }

  class Foo extends RawModule {
    val bar = Module(new Bar)
    val quz = Instance(Definition(new Quz))
    val corge = Instance(Definition(new Corge))
    val grault = Instance(Definition(new Grault))
  }

  val chirrtl = ChiselStage.emitCHIRRTL(new Foo)

  "the main module" should "be implicitly public" in {
    chirrtl should include("public module Foo")
  }

  "non-main modules" should "be implicitly private" in {
    chirrtl.fileCheck()(
      """|CHECK-NOT: public module Qux
         |CHECK:     module Qux
         |"""".stripMargin
    )
  }

  "definitions" should "be implicitly private" in {
    chirrtl.fileCheck()(
      """|CHECK-NOT: public module Quz
         |CHECK:     module Quz
         |"""".stripMargin
    )
  }

  behavior.of("the Public trait")

  it should "cause a module that mixes it in to be public" in {
    chirrtl.fileCheck()(
      """|CHECK-NOT: public module Bar
         |CHECK:     module Bar
         |"""".stripMargin
    )
  }

  it should "allow making a module that mixes it in private via an override" in {
    chirrtl.fileCheck()(
      """|CHECK-NOT: public module Baz
         |CHECK:     module Baz
         |"""".stripMargin
    )
  }

  it should "cause a Definition that mixes it in to be public" in {
    chirrtl should include("public module Corge")
  }

  it should "allow making a Definition that mixes it in private via an override" in {
    chirrtl.fileCheck()(
      """|CHECK-NOT: public module Grault
         |CHECK:     module Grault
         |"""".stripMargin
    )
  }

}
