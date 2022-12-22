// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.stage.ChiselStage

class VecToTargetSpec extends ChiselFlatSpec with Utils {
  "Vec subaccess with Scala literal" should "convert to ReferenceTarget" in {
    ChiselStage.convert {
      new Module {
        val vec = IO(Input(Vec(4,Bool())))
        val idx = 0
        dontTouch(vec(idx))
      }
    }
  }

  "Vec subaccess with Chisel literal" should "fail to convert to ReferenceTarget" in {
    (the[ChiselException] thrownBy extractCause[ChiselException] {
      ChiselStage.convert {
        new Module {
          val vec = IO(Input(Vec(4,Bool())))
          val idx = 0.U
          dontTouch(vec(idx))
        }
      }
    }).getMessage should include("Cannot target a Vec subaccess")
  }

  "Vec subaccess with node" should "fail to convert to ReferenceTarget" in {
    (the[ChiselException] thrownBy extractCause[ChiselException] {
      ChiselStage.convert {
        new Module {
          val vec = IO(Input(Vec(4,Bool())))
          val idx = IO(Input(UInt(4.W)))
          dontTouch(vec(idx))
        }
      }
    }).getMessage should include("Cannot target a Vec subaccess")
  }

  "Vec subaccess with illegal construct" should "convert to ReferenceTarget if assigned to a temporary" in {
    ChiselStage.convert {
      new Module {
        val vec = IO(Input(Vec(4,Bool())))
        val idx = 0.U
        val tmp = WireInit(vec(idx))
        dontTouch(tmp)
      }
    }
  }
}
