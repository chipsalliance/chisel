// SPDX-License-Identifier: Apache-2.0

package chiselTests.properties

import chisel3._
import chisel3.properties.Property
import chiselTests.{ChiselFlatSpec, MatchesAndOmits}
import circt.stage.ChiselStage

class PropertySpec extends ChiselFlatSpec with MatchesAndOmits {
  behavior.of("Property")

  it should "fail to compile with unsupported Property types" in {
    assertTypeError("""
      class MyThing
      val badProp = Property[MyThing]()
    """)
  }

  it should "support Int as a Property type" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val intProp = IO(Input(Property[Int]()))
    })

    matchesAndOmits(chirrtl)(
      "input intProp : Integer"
    )()
  }

  it should "support Long as a Property type" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val longProp = IO(Input(Property[Long]()))
    })

    matchesAndOmits(chirrtl)(
      "input longProp : Integer"
    )()
  }

  it should "support BigInt as a Property type" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val bigIntProp = IO(Input(Property[BigInt]()))
    })

    matchesAndOmits(chirrtl)(
      "input bigIntProp : Integer"
    )()
  }
}
