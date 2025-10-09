// SPDX-License-Identifier: Apache-2.0

package chiselTests
package experimental

import chisel3._
import chisel3.util.Valid
import chisel3.experimental.OpaqueType
import chisel3.reflect.DataMirror
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import scala.collection.immutable.SeqMap

object LOI {
  def apply(width: Width): UInt = UInt(width)
}

object LOITypeSpec {

  class Instruction extends Bundle {
    val pc = UInt(32.W)
    val globalID = LOI(100.W)
  }
  class Passthrough extends Module {
    val in  = IO(Input(new Instruction))
    val out = IO(Output(new Instruction))
    val reg = RegNext(in)
    out := reg
  }
  class Top extends Module {
    val in  = IO(Input(new Instruction))
    val out = IO(Output(new Instruction))
    val pipelineView = IO(Output(probe.Probe(Vec(3, LOI(100.W)))))

    val inst = Module(new Passthrough)
    val inst2 = Module(new Passthrough)

    inst.in := in
    inst2.in := inst.out
    out := inst2.out

    pipelineView := VecInit(Seq(in.globalID, inst.out.globalID, inst2.out.globalID))

  }

}
class LOITypeSpec extends AnyFlatSpec with Matchers {
  import LOITypeSpec._

  behavior.of("LOITypes")

  they should "support LOIType for maps with single unnamed elements" in {
    val singleElementChirrtl = ChiselStage.emitCHIRRTL { new Top }
    println(singleElementChirrtl)
  }
}