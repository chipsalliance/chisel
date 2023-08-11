// SPDX-License-Identifier: Apache-2.0

package chiselTests.properties

import chisel3._
import chisel3.properties.Class
import chiselTests.{ChiselFlatSpec, MatchesAndOmits}
import circt.stage.ChiselStage

class ClassSpec extends ChiselFlatSpec with MatchesAndOmits {
  behavior.of("Class")

  it should "serialize to FIRRTL with anonymous names" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      Module(new Class)
      Module(new Class)
      Module(new Class)
    })

    matchesAndOmits(chirrtl)(
      "class Class",
      "class Class_1",
      "class Class_2"
    )()
  }

  it should "serialize to FIRRTL with a desiredName" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      Module(new Class {
        override def desiredName = "Foo"
      })
    })

    matchesAndOmits(chirrtl)(
      "class Foo"
    )()
  }

  it should "serialize to FIRRTL with the Scala class name" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      class MyClass extends Class {}

      Module(new MyClass)
    })

    matchesAndOmits(chirrtl)(
      "class MyClass"
    )()
  }
}
