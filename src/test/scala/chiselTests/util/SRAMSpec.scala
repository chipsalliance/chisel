// SPDX-License-Identifier: Apache-2.0

package chiselTests.util

import chisel3._
import chisel3.properties.{Path, Property}
import chisel3.util.{MemoryReadWritePort, SRAM, SRAMInterface}
import chisel3.experimental.{annotate, ChiselAnnotation, OpaqueType}
import chiselTests.ChiselFlatSpec
import _root_.circt.stage.ChiselStage.{emitCHIRRTL, emitSystemVerilog}
import firrtl.annotations.{Annotation, ReferenceTarget, SingleTargetAnnotation}

import scala.collection.immutable.SeqMap

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
      annotate(new ChiselAnnotation {
        override def toFirrtl: Annotation = DummyAnno(sram.underlying.get.toTarget)
      })
    }
    val (chirrtlCircuit, annos) = getFirrtlAndAnnos(new Top)
    val chirrtl = chirrtlCircuit.serialize
    chirrtl should include("module Top :")
    chirrtl should include(
      "wire sram : { readPorts : { flip address : UInt<5>, flip enable : UInt<1>, data : UInt<8>}[0], writePorts : { flip address : UInt<5>, flip enable : UInt<1>, flip data : UInt<8>}[0], readwritePorts : { flip address : UInt<5>, flip enable : UInt<1>, flip isWrite : UInt<1>, readData : UInt<8>, flip writeData : UInt<8>}[1], description : { depth : Integer, dataWidth : Integer, masked : Bool, read : Integer, write : Integer, readwrite : Integer, maskGranularity : Integer}"
    )
    chirrtl should include("mem sram_sram")
    chirrtl should include("data-type => UInt<8>")
    chirrtl should include("depth => 32")
    chirrtl should include("read-latency => 1")
    chirrtl should include("write-latency => 1")
    chirrtl should include("readwriter => RW0")
    chirrtl should include("read-under-write => undefined")
    chirrtl should include("connect sram_sram.RW0.addr, sram.readwritePorts[0].address")
    chirrtl should include("connect sram_sram.RW0.clk, clock")
    chirrtl should include("connect sram_sram.RW0.en, sram.readwritePorts[0].enable")
    chirrtl should include("connect sram.readwritePorts[0].readData, sram_sram.RW0.rdata")
    chirrtl should include("connect sram_sram.RW0.wdata, sram.readwritePorts[0].writeData")
    chirrtl should include("connect sram_sram.RW0.wmode, sram.readwritePorts[0].isWrite")

    val dummyAnno = annos.collectFirst { case DummyAnno(t) => (t.toString) }
    dummyAnno should be(Some("~Top|Top>sram_sram"))
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
      annotate(new ChiselAnnotation {
        override def toFirrtl: Annotation = DummyAnno(sramInterface.underlying.get.toTarget)
      })
    }
    val (chirrtlCircuit, annos) = getFirrtlAndAnnos(new Top)
    val chirrtl = chirrtlCircuit.serialize
    chirrtl should include("module Top :")
    chirrtl should include("mem carrot :")
    chirrtl should include(
      "wire sramInterface : { readPorts : { flip address : UInt<5>, flip enable : UInt<1>, data : UInt<8>}[0], writePorts : { flip address : UInt<5>, flip enable : UInt<1>, flip data : UInt<8>}[0], readwritePorts : { flip address : UInt<5>, flip enable : UInt<1>, flip isWrite : UInt<1>, readData : UInt<8>, flip writeData : UInt<8>}[1], description : { depth : Integer, dataWidth : Integer, masked : Bool, read : Integer, write : Integer, readwrite : Integer, maskGranularity : Integer}"
    )

    val dummyAnno = annos.collectFirst { case DummyAnno(t) => (t.toString) }
    dummyAnno should be(Some("~Top|Top>carrot"))
  }

  it should "emit proper masks for non-Aggregate memories" in {
    class Top extends Module {
      val foo = SRAM(
        size = 32,
        tpe = UInt(8.W),
        numReadPorts = 1,
        numWritePorts = 1,
        numReadwritePorts = 0
      )
      val fooIo = IO(foo.cloneType)
      fooIo :<>= foo
    }
    val chirrtl = emitCHIRRTL(new Top)
    chirrtl should include("connect foo_sram.W0.mask, UInt<1>(0h1)")

    // check CIRCT can compile the output
    val sv = emitSystemVerilog(new Top)
  }

  it should "emit proper masks for Vec memories" in {
    class Top extends Module {
      val maskedVecMem = SRAM.masked(
        size = 32,
        tpe = Vec(2, UInt(8.W)),
        numReadPorts = 0,
        numWritePorts = 0,
        numReadwritePorts = 1
      )
      val maskedVecMemIo = IO(maskedVecMem.cloneType)
      maskedVecMemIo :<>= maskedVecMem

      val unmaskedVecMem = SRAM(
        size = 64,
        tpe = Vec(4, UInt(8.W)),
        numReadPorts = 1,
        numWritePorts = 1,
        numReadwritePorts = 0
      )
      val unmaskedVecMemIo = IO(unmaskedVecMem.cloneType)
      unmaskedVecMemIo :<>= unmaskedVecMem

      val maskedVecRecordMem = SRAM.masked(
        size = 64,
        tpe = Vec(
          3,
          new Bundle {
            val x = UInt(3.W)
            val y = Vec(4, Bool())
          }
        ),
        numReadPorts = 1,
        numWritePorts = 1,
        numReadwritePorts = 0
      )
      val maskedVecRecordMemIo = IO(maskedVecRecordMem.cloneType)
      maskedVecRecordMemIo :<>= maskedVecRecordMem
    }
    val chirrtl = emitCHIRRTL(new Top)
    _root_.circt.stage.ChiselStage.emitCHIRRTLFile(new Top)
    chirrtl should include("connect maskedVecMem_sram.RW0.wmask[0], maskedVecMem.readwritePorts[0].mask[0]")
    chirrtl should include("connect maskedVecMem_sram.RW0.wmask[1], maskedVecMem.readwritePorts[0].mask[1]")
    chirrtl should include("connect unmaskedVecMem_sram.W0.mask[0], UInt<1>(0h1)")
    chirrtl should include("connect unmaskedVecMem_sram.W0.mask[1], UInt<1>(0h1)")
    chirrtl should include("connect unmaskedVecMem_sram.W0.mask[2], UInt<1>(0h1)")
    chirrtl should include("connect unmaskedVecMem_sram.W0.mask[3], UInt<1>(0h1)")
    chirrtl should include("connect maskedVecRecordMem_sram.W0.mask[0].y[0], maskedVecRecordMem.writePorts[0].mask[0]")
    chirrtl should include("connect maskedVecRecordMem_sram.W0.mask[0].y[1], maskedVecRecordMem.writePorts[0].mask[0]")
    chirrtl should include("connect maskedVecRecordMem_sram.W0.mask[0].y[2], maskedVecRecordMem.writePorts[0].mask[0]")
    chirrtl should include("connect maskedVecRecordMem_sram.W0.mask[0].y[3], maskedVecRecordMem.writePorts[0].mask[0]")
    chirrtl should include("connect maskedVecRecordMem_sram.W0.mask[0].x, maskedVecRecordMem.writePorts[0].mask[0]")
    chirrtl should include("connect maskedVecRecordMem_sram.W0.mask[1].y[0], maskedVecRecordMem.writePorts[0].mask[1]")
    chirrtl should include("connect maskedVecRecordMem_sram.W0.mask[1].y[1], maskedVecRecordMem.writePorts[0].mask[1]")
    chirrtl should include("connect maskedVecRecordMem_sram.W0.mask[1].y[2], maskedVecRecordMem.writePorts[0].mask[1]")
    chirrtl should include("connect maskedVecRecordMem_sram.W0.mask[1].y[3], maskedVecRecordMem.writePorts[0].mask[1]")
    chirrtl should include("connect maskedVecRecordMem_sram.W0.mask[1].x, maskedVecRecordMem.writePorts[0].mask[1]")
    chirrtl should include("connect maskedVecRecordMem_sram.W0.mask[2].y[0], maskedVecRecordMem.writePorts[0].mask[2]")
    chirrtl should include("connect maskedVecRecordMem_sram.W0.mask[2].y[1], maskedVecRecordMem.writePorts[0].mask[2]")
    chirrtl should include("connect maskedVecRecordMem_sram.W0.mask[2].y[2], maskedVecRecordMem.writePorts[0].mask[2]")
    chirrtl should include("connect maskedVecRecordMem_sram.W0.mask[2].y[3], maskedVecRecordMem.writePorts[0].mask[2]")
    chirrtl should include("connect maskedVecRecordMem_sram.W0.mask[2].x, maskedVecRecordMem.writePorts[0].mask[2]")

    // check CIRCT can compile the output
    val sv = emitSystemVerilog(new Top)
  }

  it should "emit proper masks for Record memories" in {
    class Top extends Module {
      // SRAM does not currently support masked Records
      val unmaskedRecordMem = SRAM(
        size = 64,
        tpe = new Bundle {
          val x = UInt(3.W)
          val y = Vec(4, Bool())
        },
        numReadPorts = 0,
        numWritePorts = 0,
        numReadwritePorts = 1
      )
      val unmaskedRecordMemIo = IO(unmaskedRecordMem.cloneType)
      unmaskedRecordMemIo :<>= unmaskedRecordMem
    }
    val chirrtl = emitCHIRRTL(new Top)
    chirrtl should include("connect unmaskedRecordMem_sram.RW0.wmask.y[0], UInt<1>(0h1)")
    chirrtl should include("connect unmaskedRecordMem_sram.RW0.wmask.y[1], UInt<1>(0h1)")
    chirrtl should include("connect unmaskedRecordMem_sram.RW0.wmask.y[2], UInt<1>(0h1)")
    chirrtl should include("connect unmaskedRecordMem_sram.RW0.wmask.y[3], UInt<1>(0h1)")
    chirrtl should include("connect unmaskedRecordMem_sram.RW0.wmask.x, UInt<1>(0h1)")

    // check CIRCT can compile the output
    val sv = emitSystemVerilog(new Top)
  }

  it should "emit proper masks for OpaqueTypes memories" in {
    class Box[T <: Data](gen: T) extends Record with OpaqueType {
      val underlying = gen.cloneType
      val elements = SeqMap("" -> gen)
    }
    class Top extends Module {
      // SRAM does not currently support masked Records
      val unmaskedRecordMem = SRAM(
        size = 64,
        tpe = new Box(new Bundle {
          val x = new Box(UInt(3.W))
          val y = new Box(Vec(4, Bool()))
        }),
        numReadPorts = 0,
        numWritePorts = 0,
        numReadwritePorts = 1
      )
      val unmaskedRecordMemIo = IO(unmaskedRecordMem.cloneType)
      unmaskedRecordMemIo :<>= unmaskedRecordMem
    }
    val chirrtl = emitCHIRRTL(new Top)
    chirrtl should include("connect unmaskedRecordMem_sram.RW0.wmask.y[0], UInt<1>(0h1)")
    chirrtl should include("connect unmaskedRecordMem_sram.RW0.wmask.y[1], UInt<1>(0h1)")
    chirrtl should include("connect unmaskedRecordMem_sram.RW0.wmask.y[2], UInt<1>(0h1)")
    chirrtl should include("connect unmaskedRecordMem_sram.RW0.wmask.y[3], UInt<1>(0h1)")
    chirrtl should include("connect unmaskedRecordMem_sram.RW0.wmask.x, UInt<1>(0h1)")

    // check CIRCT can compile the output
    val sv = emitSystemVerilog(new Top)
  }

  it should "be possible to access SRAM description information" in {

    class Top extends Module {
      val size = IO(Output(Property[Int]()))
      val sram = SRAM(
        size = 32,
        tpe = UInt(8.W),
        numReadPorts = 0,
        numWritePorts = 0,
        numReadwritePorts = 1
      )
      size := sram.description.get.depth
    }
    // TODO we need a way with ChiselSim to evaluate properties
    val chirrtl = emitCHIRRTL(new Top)
    chirrtl should include("output size : Integer")
    chirrtl should include("propassign size, sram.description.depth")
  }

  it should "be possible to create an SramInterface Wire" in {
    class Top extends Module {
      val sramIntf = Wire(
        new SRAMInterface(
          memSize = 32,
          tpe = Vec(4, UInt(8.W)),
          numReadPorts = 0,
          numWritePorts = 0,
          numReadwritePorts = 1,
          masked = false
        )
      )
      sramIntf := DontCare
    }
    // should not give uninitialized errors for description properties
    emitSystemVerilog(new Top)
  }
}
