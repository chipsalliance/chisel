// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.stage.ChiselStage

trait VecToTargetSpecUtils {
  class Foo extends RawModule {
    val vec = IO(Input(Vec(4, Bool())))

    // Index a Vec with a Scala literal.
    val scalaLitIdx = 0
    val vecScalaLitIdx = vec(scalaLitIdx)

    // Index a Vec with a Chisel literal.
    val chiselLitIdx = 0.U
    val vecChiselLitIdx = vec(chiselLitIdx)

    // Index a Vec with a node.
    val nodeIdx = IO(Input(UInt(2.W)))
    val vecNodeIdx = vec(nodeIdx)

    // Put an otherwise un-targetable Vec subaccess into a temp.
    val vecWithTmp = WireInit(vecNodeIdx)
  }

  var foo: Foo = null

  ChiselStage.elaborate { foo = new Foo; foo }

  val expectedError = "You cannot target Vec subaccess:"
}

class VecToTargetSpec extends ChiselFlatSpec with VecToTargetSpecUtils with Utils {
  "Vec subaccess with Scala literal" should "convert to ReferenceTarget" in {
    foo.vecScalaLitIdx.toTarget
  }

  "Vec subaccess with Scala literal" should "convert to ComponentName" in {
    foo.vecScalaLitIdx.toNamed
  }

  "Vec subaccess with Chisel literal" should "fail to convert to ReferenceTarget" in {
    (the[ChiselException] thrownBy extractCause[ChiselException] {
      foo.vecChiselLitIdx.toTarget
    }).getMessage should include(expectedError)
  }

  "Vec subaccess with Chisel literal" should "fail to convert to ComponentName" in {
    (the[ChiselException] thrownBy extractCause[ChiselException] {
      foo.vecChiselLitIdx.toNamed
    }).getMessage should include(expectedError)
  }

  "Vec subaccess with node" should "fail to convert to ReferenceTarget" in {
    (the[ChiselException] thrownBy extractCause[ChiselException] {
      foo.vecNodeIdx.toTarget
    }).getMessage should include(expectedError)
  }

  "Vec subaccess with node" should "fail to convert to ComponentName" in {
    (the[ChiselException] thrownBy extractCause[ChiselException] {
      foo.vecNodeIdx.toNamed
    }).getMessage should include(expectedError)
  }

  "Vec subaccess with un-targetable construct" should "convert to ReferenceTarget if assigned to a temporary" in {
    foo.vecWithTmp.toTarget
  }

  "Vec subaccess with un-targetable construct" should "convert to ComponentName if assigned to a temporary" in {
    foo.vecWithTmp.toNamed
  }
}
