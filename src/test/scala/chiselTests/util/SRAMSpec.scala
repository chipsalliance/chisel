// SPDX-License-Identifier: Apache-2.0

package chiselTests.util

import chisel3._
import chisel3.util.SRAM
import chisel3.experimental.annotate
import chiselTests.ChiselFlatSpec
import _root_.circt.stage.ChiselStage.emitCHIRRTL
import firrtl.annotations.{Annotation, ReferenceTarget, SingleTargetAnnotation}

class SRAMSpec extends ChiselFlatSpec {
  case class DummyAnno(target: ReferenceTarget) extends SingleTargetAnnotation[ReferenceTarget] {
    override def duplicate(n: ReferenceTarget) = this.copy(target = n)
  }
  behavior.of("SRAMInterface")

  it should "Provide target information about its instantiating SRAM" in {

    class Top extends Module {
      val sram = SRAM(
        size = 32,
        tpe = UInt(8.W),
        numReadPorts = 0,
        numWritePorts = 0,
        numReadwritePorts = 1
      )
      require(sram.underlying.nonEmpty)
      annotate(sram.underlying.get)(Seq(DummyAnno(sram.underlying.get.toTarget)))
    }
    val (chirrtlCircuit, annos) = getFirrtlAndAnnos(new Top)
    val chirrtl = chirrtlCircuit.serialize
    chirrtl should include("module Top :")
    chirrtl should include("smem sram_mem : UInt<8> [32]")
    chirrtl should include(
      "wire sram : { readPorts : { flip address : UInt<5>, flip enable : UInt<1>, data : UInt<8>}[0], writePorts : { flip address : UInt<5>, flip enable : UInt<1>, flip data : UInt<8>}[0], readwritePorts : { flip address : UInt<5>, flip enable : UInt<1>, flip isWrite : UInt<1>, readData : UInt<8>, flip writeData : UInt<8>}[1]}"
    )

    val dummyAnno = annos.collectFirst { case DummyAnno(t) => (t.toString) }
    dummyAnno should be(Some("~Top|Top>sram_mem"))
  }

  it should "Get emitted with a custom name when one is suggested" in {

    class Top extends Module {
      val sramInterface = SRAM(
        size = 32,
        tpe = UInt(8.W),
        numReadPorts = 0,
        numWritePorts = 0,
        numReadwritePorts = 1
      )
      require(sramInterface.underlying.nonEmpty)
      sramInterface.underlying.get.suggestName("carrot")
      annotate(sramInterface.underlying.get)(Seq(DummyAnno(sramInterface.underlying.get.toTarget)))
    }
    val (chirrtlCircuit, annos) = getFirrtlAndAnnos(new Top)
    val chirrtl = chirrtlCircuit.serialize
    chirrtl should include("module Top :")
    chirrtl should include("smem carrot : UInt<8> [32]")
    chirrtl should include(
      "wire sramInterface : { readPorts : { flip address : UInt<5>, flip enable : UInt<1>, data : UInt<8>}[0], writePorts : { flip address : UInt<5>, flip enable : UInt<1>, flip data : UInt<8>}[0], readwritePorts : { flip address : UInt<5>, flip enable : UInt<1>, flip isWrite : UInt<1>, readData : UInt<8>, flip writeData : UInt<8>}[1]}"
    )

    val dummyAnno = annos.collectFirst { case DummyAnno(t) => (t.toString) }
    dummyAnno should be(Some("~Top|Top>carrot"))
  }

}
