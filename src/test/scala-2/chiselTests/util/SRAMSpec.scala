// SPDX-License-Identifier: Apache-2.0

package chiselTests.util

import _root_.circt.stage.ChiselStage.{emitCHIRRTL, emitSystemVerilog}
import _root_.circt.stage.{CIRCTTarget, CIRCTTargetAnnotation, ChiselStage}
import chisel3._
import chisel3.experimental.{annotate, OpaqueType}
import chisel3.stage.{ChiselGeneratorAnnotation, IncludeUtilMetadata, UseSRAMBlackbox}
import chisel3.testing.scalatest.FileCheck
import chisel3.util.{MemoryReadWritePort, SRAM}
import firrtl.EmittedVerilogCircuitAnnotation
import firrtl.annotations.{Annotation, ReferenceTarget, SingleTargetAnnotation}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import scala.collection.immutable.SeqMap
import scala.util.chaining.scalaUtilChainingOps

class SRAMSpec extends AnyFlatSpec with Matchers with FileCheck {
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
    ChiselStage
      .emitCHIRRTL(new Top, Array("--include-util-metadata"))
      .fileCheck()(
        """|CHECK:      "class":"chiselTests.util.SRAMSpec$DummyAnno"
           |CHECK-NEXT: "target":"~|Top>sram_sram"
           |
           |CHECK:      public module Top :
           |CHECK:        wire sram : { readPorts : { flip address : UInt<5>, flip enable : UInt<1>, data : UInt<8>}[0], writePorts : { flip address : UInt<5>, flip enable : UInt<1>, flip data : UInt<8>}[0], readwritePorts : { flip address : UInt<5>, flip enable : UInt<1>, flip isWrite : UInt<1>, readData : UInt<8>, flip writeData : UInt<8>}[1], description : Inst<SRAMDescription>}
           |CHECK-NEXT:   mem sram_sram
           |CHECK-NEXT:     data-type => UInt<8>
           |CHECK-NEXT:     depth => 32
           |CHECK-NEXT:     read-latency => 1
           |CHECK-NEXT:     write-latency => 1
           |CHECK-NEXT:     readwriter => RW0
           |CHECK-NEXT:     read-under-write => undefined
           |CHECK:        connect sram_sram.RW0.addr, sram.readwritePorts[0].address
           |CHECK:        connect sram_sram.RW0.clk, clock
           |CHECK:        connect sram_sram.RW0.en, sram.readwritePorts[0].enable
           |CHECK:        connect sram.readwritePorts[0].readData, sram_sram.RW0.rdata
           |CHECK:        connect sram_sram.RW0.wdata, sram.readwritePorts[0].writeData
           |CHECK:        connect sram_sram.RW0.wmode, sram.readwritePorts[0].isWrite
           |CHECK:        propassign sram_descriptionInstance.hierarchyIn, path("OMReferenceTarget:~|Top>sram_sram")
           |CHECK:        propassign sram.description, sram_descriptionInstance
           |""".stripMargin
      )
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
    ChiselStage
      .emitCHIRRTL(new Top)
      .fileCheck()(
        """|CHECK:      "class":"chiselTests.util.SRAMSpec$DummyAnno"
           |CHECK-NEXT: "target":"~|Top>carrot"
           |
           |CHECK:      public module Top :
           |CHECK:        wire sramInterface : { readPorts : { flip address : UInt<5>, flip enable : UInt<1>, data : UInt<8>}[0], writePorts : { flip address : UInt<5>, flip enable : UInt<1>, flip data : UInt<8>}[0], readwritePorts : { flip address : UInt<5>, flip enable : UInt<1>, flip isWrite : UInt<1>, readData : UInt<8>, flip writeData : UInt<8>}[1]}
           |CHECK-NEXT:   mem carrot :
           |""".stripMargin
      )
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

  it should "elide metadata by default" in {
    class Top extends Module {
      val sram = SRAM(
        size = 32,
        tpe = UInt(8.W),
        numReadPorts = 0,
        numWritePorts = 0,
        numReadwritePorts = 1
      )
    }
    val chirrtl = emitCHIRRTL(new Top)
    // there should be no properties
    chirrtl shouldNot include(" class ")
    chirrtl shouldNot include("Integer")
    chirrtl shouldNot include("Path")
    chirrtl shouldNot include("propassign")
  }

  it should "get emitted by SRAMBlackbox" in {
    def test(rd: Int, wr: Int, rw: Int, depth: Int, width: Int, maskGranularity: Int) = {
      class Top extends Module {
        val sram = SRAM.masked(depth, Vec(width / maskGranularity, UInt(maskGranularity.W)), rd, wr, rw)

        val ioR = IO(chiselTypeOf(sram.readPorts)).tap(_.zip(sram.readPorts).foreach { case (io, mem) =>
          io <> mem
        })
        val ioRW = IO(chiselTypeOf(sram.readwritePorts)).tap(_.zip(sram.readwritePorts).foreach { case (io, mem) =>
          io <> mem
        })
        val ioW = IO(chiselTypeOf(sram.writePorts)).tap(_.zip(sram.writePorts).foreach { case (io, mem) =>
          io <> mem
        })
      }

      val resultDir = "SRAMSpecTemp"
      (new ChiselStage)
        .execute(
          Array("--target-dir", resultDir, "--split-verilog"),
          Seq(
            ChiselGeneratorAnnotation(() => new Top),
            CIRCTTargetAnnotation(CIRCTTarget.SystemVerilog),
            UseSRAMBlackbox
          )
        )

      os.proc("slang", "--lint-only", "sram_0R_0W_1RW_2M_32x8.sv").call(os.pwd / resultDir)
    }
    Seq.tabulate(2, 2, 2) { case (rd, wr, rw) => if (rd + rw != 0 && wr + rw != 0) test(rd, wr, rw, 32, 8, 2) }
  }
}
