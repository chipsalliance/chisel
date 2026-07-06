// SPDX-License-Identifier: Apache-2.0

package chiselTests.properties

import chisel3._
import chisel3.properties.{AnyClassType, Class, Property}
import chisel3.testing.scalatest.FileCheck
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class AnyClassTypeSpec extends AnyFlatSpec with Matchers with FileCheck {
  behavior.of("AnyClassType")

  it should "build a Property[Seq[AnyClassType]] from object references via asAnyClassType" in {
    ChiselStage
      .emitCHIRRTL(new RawModule {
        val myClassA = Class.unsafeGetDynamicObject("MyClassA")
        val myClassB = Class.unsafeGetDynamicObject("MyClassB")
        val out = IO(Output(Property[Seq[AnyClassType]]()))
        out :#= Property(Seq(myClassA.getReference.asAnyClassType, myClassB.getReference.asAnyClassType))
      })
      .fileCheck()(
        """|CHECK: output out : List<AnyRef>
           |CHECK: propassign out, List<AnyRef>(myClassA, myClassB)
           |""".stripMargin
      )
  }

  it should "build a single-element Property[Seq[AnyClassType]] via asAnyClassType" in {
    ChiselStage
      .emitCHIRRTL(new RawModule {
        val myClass = Class.unsafeGetDynamicObject("MyClass")
        val out = IO(Output(Property[Seq[AnyClassType]]()))
        out :#= Property(Seq(myClass.getReference.asAnyClassType))
      })
      .fileCheck()(
        """|CHECK: output out : List<AnyRef>
           |CHECK: propassign out, List<AnyRef>(myClass)
           |""".stripMargin
      )
  }
}
