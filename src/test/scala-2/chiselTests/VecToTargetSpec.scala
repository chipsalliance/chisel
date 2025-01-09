// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import circt.stage.ChiselStage

trait VecToTargetSpecUtils extends Utils {
  this: ChiselFunSpec =>

  class Foo extends RawModule {
    val vec = IO(Input(Vec(4, Bool())))

    // Index a Vec with a Scala literal.
    val scalaLit = 0
    val vecSubaccessScalaLit = vec(scalaLit)

    // Index a Vec with a Chisel literal.
    val chiselLit = 0.U
    val vecSubaccessChiselLit = vec(chiselLit)

    // Index a Vec with a node.
    val node = IO(Input(UInt(2.W)))
    val vecSubaccessNode = vec(node)

    // Put an otherwise un-targetable Vec subaccess into a temp.
    val vecSubaccessTmp = WireInit(vecSubaccessNode)
  }

  val expectedError = "You cannot target Vec subaccess:"

  def conversionSucceeds(data: InstanceId) = {
    describe(".toTarget") {
      it("should convert successfully") {
        data.toTarget
      }
    }

    describe(".toNamed") {
      it("should convert successfully") {
        data.toNamed
      }
    }
  }

  def conversionFails(data: InstanceId) = {
    describe(".toTarget") {
      it("should fail to convert with a useful error message") {
        (the[ChiselException] thrownBy extractCause[ChiselException] {
          data.toTarget
        }).getMessage should include(expectedError)
      }
    }

    describe(".toNamed") {
      it("should fail to convert with a useful error message") {
        (the[ChiselException] thrownBy extractCause[ChiselException] {
          data.toNamed
        }).getMessage should include(expectedError)
      }
    }
  }
}

class VecToTargetSpec extends ChiselFunSpec with VecToTargetSpecUtils {
  describe("Vec subaccess") {
    var foo: Foo = null
    ChiselStage.emitCHIRRTL { foo = new Foo; foo }

    describe("with a Scala literal") {
      (it should behave).like(conversionSucceeds(foo.vecSubaccessScalaLit))
    }

    describe("with a Chisel literal") {
      (it should behave).like(conversionSucceeds(foo.vecSubaccessChiselLit))
    }

    describe("with a Node") {
      (it should behave).like(conversionFails(foo.vecSubaccessNode))
    }

    describe("with an un-targetable construct that is assigned to a temporary") {
      (it should behave).like(conversionSucceeds(foo.vecSubaccessTmp))
    }
  }
}
