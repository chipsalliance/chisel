package circtTests.tywavesTests.memTests

import chisel3._
import chisel3.stage.ChiselGeneratorAnnotation
import chisel3.tywaves.ClassParam
import circt.stage.ChiselStage
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers

class TypeAnnotationMemSpec extends AnyFunSpec with Matchers with chiselTests.Utils {
  import circtTests.tywavesTests.TywavesAnnotationCircuits.MemCircuits._
  import circtTests.tywavesTests.TestUtils._

  def createExpectedSRAMs(
    target:            String,
    size:              BigInt,
    tpe:               String,
    numReadPorts:      Int,
    numWritePorts:     Int,
    numReadwritePorts: Int,
    masked:            Boolean = false,
    dataParams:        Option[Seq[ClassParam]] = None // For SRAMs of complex data types with params
  ): Seq[(String, Int)] = {
    import scala.math._
    val addrWidth = if (size > 1) ceil(log(size.toInt) / log(2)).toInt else 1
    def genParams(tpe: String) = {
      if (tpe == "MemoryReadPort")
        Some(
          Seq(ClassParam("tpe", "T", None), ClassParam("addrWidth", "Int", None))
        )
      else if (tpe == "MemoryWritePort" || tpe == "MemoryReadWritePort")
        Some(
          Seq(ClassParam("tpe", "T", None), ClassParam("addrWidth", "Int", None), ClassParam("masked", "Boolean", None))
        )
      else
        None
    }
    def generatePorts(rw: String, _n: Int, portType: String) = {
      var ports = Seq[(String, Int)]()
      val n = if (_n == 0) 0 else 1
      for (i <- 0 until n) { // TODO: I am annotating now only the first element, since the size is known ant the memory has homogeneous type, it's useless to annotate every element
        ports ++= Seq(
          (createExpected(s"$target.$rw\\[$i\\]", s"$portType", "Wire", genParams(portType)), 1),
          (createExpected(s"$target.$rw\\[$i\\].address", s"UInt<$addrWidth>", "Wire"), 1),
          (createExpected(s"$target.$rw\\[$i\\].enable", "Bool", "Wire"), 1)
        ) ++ (if (rw == "readwritePorts")
                Seq(
                  (createExpected(s"$target.$rw\\[$i\\].readData", tpe, "Wire", dataParams), 1),
                  (createExpected(s"$target.$rw\\[$i\\].writeData", tpe, "Wire", dataParams), 1),
                  (createExpected(s"$target.$rw\\[$i\\].isWrite", "Bool", "Wire"), 1)
                )
              else Seq((createExpected(s"$target.$rw\\[$i\\].data", tpe, "Wire", dataParams), 1)))
      }
      ports
    }
    println("target: " + target)
    // format: off
    Seq(
      (createExpected(s"$target", "SRAMInterface", "Wire",
          params = Some(Seq(
            ClassParam("memSize", "BigInt", Some(s"$size")), // TODO: check why the value is not available even if memSize is val
            ClassParam("tpe", "T", None),
            ClassParam("numReadPorts", "Int", Some(s"$numReadPorts")),
            ClassParam("numWritePorts", "Int", Some(s"$numWritePorts")),
            ClassParam("numReadwritePorts", "Int", Some(s"$numReadwritePorts")),
            ClassParam("masked", "Boolean", Some(s"$masked")),
          ))
        ), 1),
      (createExpected(s"$target.readPorts", s"MemoryReadPort\\[$numReadPorts\\]", "Wire",
        params = Some(Seq(ClassParam("gen", "=> T", None), ClassParam("length", "Int", Some(s"$numReadPorts"))))), 1),
      (createExpected(s"$target.writePorts", s"MemoryWritePort\\[$numWritePorts\\]", "Wire",
        params = Some(Seq(ClassParam("gen", "=> T", None), ClassParam("length", "Int", Some(s"$numWritePorts"))))), 1),
      (createExpected(s"$target.readwritePorts", s"MemoryReadWritePort\\[$numReadwritePorts\\]", "Wire",
        params = Some(Seq(ClassParam("gen", "=> T", None), ClassParam("length", "Int", Some(s"$numReadwritePorts"))))), 1),
    ) ++ generatePorts("readPorts", numReadPorts, "MemoryReadPort") ++
      generatePorts("writePorts", numWritePorts, "MemoryWritePort") ++
      generatePorts("readwritePorts", numReadwritePorts, "MemoryReadWritePort")
      // format: on
  }

  describe("Memory Annotations") {
    val targetDir = os.pwd / "test_run_dir" / "TywavesAnnotationSpec" / "Memories Annotations"
    val args: Array[String] = Array("--target", "chirrtl", "--target-dir", targetDir.toString)

    it("should annotate a ROM") {
      new ChiselStage(true).execute(args, Seq(ChiselGeneratorAnnotation(() => new TopCircuitROM)))
      // format: off
      val expectedMatches = Seq(
        (createExpected("~TopCircuitROM\\|TopCircuitROM>romFromVec", "UInt<8>\\[4\\]", "Wire",
          params = Some(Seq(ClassParam("gen", "=> T", None), ClassParam("length", "Int", Some("4"))))), 1),
        (createExpected("~TopCircuitROM\\|TopCircuitROM>romFromVec\\[0\\]", "UInt<8>", "Wire"), 1),
        (createExpected("~TopCircuitROM\\|TopCircuitROM>romFromVecInit", "UInt<4>\\[4\\]", "Wire",
          params = Some(Seq(ClassParam("gen", "=> T", None), ClassParam("length", "Int", Some("4"))))), 1),
        (createExpected("~TopCircuitROM\\|TopCircuitROM>romFromVecInit\\[0\\]", "UInt<4>", "Wire"), 1),
        (createExpected("~TopCircuitROM\\|TopCircuitROM>romOfBundles", "AnonymousBundle\\[4\\]", "Wire",
          params = Some(Seq(ClassParam("gen", "=> T", None), ClassParam("length", "Int", Some("4"))))), 1),
        (createExpected("~TopCircuitROM\\|TopCircuitROM>romOfBundles\\[0\\]", "AnonymousBundle", "Wire"), 1),
        (createExpected("~TopCircuitROM\\|TopCircuitROM>romOfBundles\\[0\\].a", "UInt<8>", "Wire"), 1)
      )
      // format: on
      checkAnno(expectedMatches, os.read(targetDir / "TopCircuitROM.fir"))
    }

    it("should annotate a SyncMem") {
      new ChiselStage(true).execute(args, Seq(ChiselGeneratorAnnotation(() => new TopCircuitSyncMem(UInt(8.W), false))))
      val expectedMatches = Seq(
        (createExpected("~TopCircuitSyncMem\\|TopCircuitSyncMem>mem", "UInt<8>\\[4\\]", "SyncReadMem"), 1),
        (createExpected("~TopCircuitSyncMem\\|TopCircuitSyncMem>clock", "Clock", "IO"), 1),
        (createExpected("~TopCircuitSyncMem\\|TopCircuitSyncMem>reset", "Bool", "IO"), 1)
      )
      checkAnno(expectedMatches, os.read(targetDir / "TopCircuitSyncMem.fir"))
    }

    it("should annotate a Mem") {
      new ChiselStage(true).execute(args, Seq(ChiselGeneratorAnnotation(() => new TopCircuitMem(UInt(8.W), false))))
      val expectedMatches = Seq(
        (createExpected("~TopCircuitMem\\|TopCircuitMem>mem", "UInt<8>\\[4\\]", "Mem"), 1),
        (createExpected("~TopCircuitMem\\|TopCircuitMem>clock", "Clock", "IO"), 1),
        (createExpected("~TopCircuitMem\\|TopCircuitMem>reset", "Bool", "IO"), 1)
      )
      checkAnno(expectedMatches, os.read(targetDir / "TopCircuitMem.fir"))
    }

    it("should annotate an SRAM") {
      new ChiselStage(true)
        .execute(args, Seq(ChiselGeneratorAnnotation(() => new TopCircuitSRAM(UInt(8.W), 1, 1, 1, 1))))
      val expectedMatches = Seq(
        (createExpected("~TopCircuitSRAM\\|TopCircuitSRAM>mem_sram", "UInt<8>\\[1\\]", "SramTarget"), 1),
        (createExpected("~TopCircuitSRAM\\|TopCircuitSRAM>clock", "Clock", "IO"), 1),
        (createExpected("~TopCircuitSRAM\\|TopCircuitSRAM>reset", "Bool", "IO"), 1)
      ) ++ createExpectedSRAMs("~TopCircuitSRAM\\|TopCircuitSRAM>mem", 1, "UInt<8>", 1, 1, 1)
      checkAnno(expectedMatches, os.read(targetDir / "TopCircuitSRAM.fir"))
    }

    it("should annotate an SRAM different dimensions and ports number") {
      val targetDir = os.pwd / "test_run_dir" / "TywavesAnnotationSpec" / "Memories Annotations" / "Multi ports number"
      val args = Array("--target", "chirrtl", "--target-dir", targetDir.toString())
      new ChiselStage(true)
        .execute(args, Seq(ChiselGeneratorAnnotation(() => new TopCircuitSRAM(UInt(8.W), 4, 2, 3, 4))))
      val expectedMatches = Seq(
        (createExpected("~TopCircuitSRAM\\|TopCircuitSRAM>mem_sram", "UInt<8>\\[4\\]", "SramTarget"), 1),
        (createExpected("~TopCircuitSRAM\\|TopCircuitSRAM>clock", "Clock", "IO"), 1),
        (createExpected("~TopCircuitSRAM\\|TopCircuitSRAM>reset", "Bool", "IO"), 1)
      ) ++ createExpectedSRAMs("~TopCircuitSRAM\\|TopCircuitSRAM>mem", size = 4, "UInt<8>", 2, 3, 4)
      checkAnno(expectedMatches, os.read(targetDir / "TopCircuitSRAM.fir"))
    }

  }

  describe("Memory Annotations of complex types") {
    import circtTests.tywavesTests.TywavesAnnotationCircuits.DataTypesCircuits.{MyBundle, MyNestedBundle}

    val targetDir = os.pwd / "test_run_dir" / "TywavesAnnotationSpec" / "Memories Annotations of complex types"
    val args: Array[String] = Array("--target", "chirrtl", "--target-dir", targetDir.toString)

    it("should annotate the top SyncMem, not its child elements") {
      new ChiselStage(true)
        .execute(args, Seq(ChiselGeneratorAnnotation(() => new TopCircuitSyncMem(new MyBundle, false))))
      val expectedMatches = Seq(
        (createExpected("~TopCircuitSyncMem\\|TopCircuitSyncMem>mem", "MyBundle\\[4\\]", "SyncReadMem"), 1),
        (createExpected("~TopCircuitSyncMem\\|TopCircuitSyncMem>clock", "Clock", "IO"), 1),
        (createExpected("~TopCircuitSyncMem\\|TopCircuitSyncMem>reset", "Bool", "IO"), 1)
      )
      checkAnno(expectedMatches, os.read(targetDir / "TopCircuitSyncMem.fir"))
    }

    it("should annotate only the top Mem, not its elements") {
      /* When only the mem is instantiated, its "child" elements are not annotated. This is because they cannot be targets.
          cmem mem : {
            a : UInt<1>,
            b : { a : UInt<8>, b : SInt<8>, c : UInt<1>},
            flip c : { a : UInt<8>, b : SInt<8>, c : UInt<1>}
          } [4]
       */
      new ChiselStage(true)
        .execute(args, Seq(ChiselGeneratorAnnotation(() => new TopCircuitMem(new MyNestedBundle, false))))
      val expectedMatches = Seq(
        (createExpected("~TopCircuitMem\\|TopCircuitMem>mem", "MyNestedBundle\\[4\\]", "Mem"), 1),
        (createExpected("~TopCircuitMem\\|TopCircuitMem>clock", "Clock", "IO"), 1),
        (createExpected("~TopCircuitMem\\|TopCircuitMem>reset", "Bool", "IO"), 1)
      )
      checkAnno(expectedMatches, os.read(targetDir / "TopCircuitMem.fir"))
    }

    it("should annotate an SRAM") {
      new ChiselStage(true)
        .execute(args, Seq(ChiselGeneratorAnnotation(() => new TopCircuitSRAM(new MyBundle, 1, 1, 1, 0))))
      val expectedMatches = Seq(
        (createExpected("~TopCircuitSRAM\\|TopCircuitSRAM>mem_sram", "MyBundle\\[1\\]", "SramTarget"), 1),
        (createExpected("~TopCircuitSRAM\\|TopCircuitSRAM>clock", "Clock", "IO"), 1),
        (createExpected("~TopCircuitSRAM\\|TopCircuitSRAM>reset", "Bool", "IO"), 1),
        (createExpected("~TopCircuitSRAM\\|TopCircuitSRAM>mem.readPorts\\[0\\].data.a", "UInt<8>", "Wire"), 1),
        (createExpected("~TopCircuitSRAM\\|TopCircuitSRAM>mem.readPorts\\[0\\].data.b", "SInt<8>", "Wire"), 1),
        (createExpected("~TopCircuitSRAM\\|TopCircuitSRAM>mem.readPorts\\[0\\].data.c", "Bool", "Wire"), 1),
        (createExpected("~TopCircuitSRAM\\|TopCircuitSRAM>mem.writePorts\\[0\\].data.a", "UInt<8>", "Wire"), 1),
        (createExpected("~TopCircuitSRAM\\|TopCircuitSRAM>mem.writePorts\\[0\\].data.b", "SInt<8>", "Wire"), 1),
        (createExpected("~TopCircuitSRAM\\|TopCircuitSRAM>mem.writePorts\\[0\\].data.c", "Bool", "Wire"), 1)
      ) ++ createExpectedSRAMs("~TopCircuitSRAM\\|TopCircuitSRAM>mem", 1, "MyBundle", 1, 1, 0)
      checkAnno(expectedMatches, os.read(targetDir / "TopCircuitSRAM.fir"))
    }
  }

  describe("Memory Annotations with MPORT connections") {

    val targetDir = os.pwd / "test_run_dir" / "TywavesAnnotationSpec" / "Memories Annotations with MPORT connections"
    val args: Array[String] = Array("--target", "chirrtl", "--target-dir", targetDir.toString)

    it("should annotate a SyncMem with MPORT connections of ground type") {
      new ChiselStage(true)
        .execute(args, Seq(ChiselGeneratorAnnotation(() => new TopCircuitSyncMem(UInt(8.W), withConnection = true))))
      val expectedMatches = Seq(
        (createExpected("~TopCircuitSyncMem\\|TopCircuitSyncMem>mem", "UInt<8>\\[4\\]", "SyncReadMem"), 1),
        (createExpected("~TopCircuitSyncMem\\|TopCircuitSyncMem>idx", "UInt<2>", "IO"), 1),
        (createExpected("~TopCircuitSyncMem\\|TopCircuitSyncMem>in", "UInt<8>", "IO"), 1),
        (createExpected("~TopCircuitSyncMem\\|TopCircuitSyncMem>out", "UInt<8>", "IO"), 1),
        (createExpected("~TopCircuitSyncMem\\|TopCircuitSyncMem>_out_WIRE", "UInt<2>", "Wire"), 1),
        (createExpected("~TopCircuitSyncMem\\|TopCircuitSyncMem>MPORT", "UInt<8>", "MemPort"), 1),
        (createExpected("~TopCircuitSyncMem\\|TopCircuitSyncMem>out_MPORT", "UInt<8>", "MemPort"), 1),
        (createExpected("~TopCircuitSyncMem\\|TopCircuitSyncMem>clock", "Clock", "IO"), 1),
        (createExpected("~TopCircuitSyncMem\\|TopCircuitSyncMem>reset", "Bool", "IO"), 1)
      )
      checkAnno(expectedMatches, os.read(targetDir / "TopCircuitSyncMem.fir"))
    }

    it("should annotate a Mem with MPORT connections of ground type") {
      new ChiselStage(true)
        .execute(args, Seq(ChiselGeneratorAnnotation(() => new TopCircuitMem(UInt(8.W), withConnection = true))))
      val expectedMatches = Seq(
        (createExpected("~TopCircuitMem\\|TopCircuitMem>mem", "UInt<8>\\[4\\]", "Mem"), 1),
        (createExpected("~TopCircuitMem\\|TopCircuitMem>idx", "UInt<2>", "IO"), 1),
        (createExpected("~TopCircuitMem\\|TopCircuitMem>in", "UInt<8>", "IO"), 1),
        (createExpected("~TopCircuitMem\\|TopCircuitMem>out", "UInt<8>", "IO"), 1),
        (createExpected("~TopCircuitMem\\|TopCircuitMem>MPORT", "UInt<8>", "MemPort"), 1),
        (createExpected("~TopCircuitMem\\|TopCircuitMem>out_MPORT", "UInt<8>", "MemPort"), 1),
        (createExpected("~TopCircuitMem\\|TopCircuitMem>clock", "Clock", "IO"), 1),
        (createExpected("~TopCircuitMem\\|TopCircuitMem>reset", "Bool", "IO"), 1)
      )
      checkAnno(expectedMatches, os.read(targetDir / "TopCircuitMem.fir"))
    }
  }

  describe("Memory Annotations with MPORT connections of complex types") {
    import circtTests.tywavesTests.TywavesAnnotationCircuits.DataTypesCircuits.MyBundle

    val targetDir =
      os.pwd / "test_run_dir" / "TywavesAnnotationSpec" / "Memories Annotations with MPORT connections" / "Complex type"
    val args: Array[String] = Array("--target", "chirrtl", "--target-dir", targetDir.toString)
    it("should annotate a SyncMem with MPORT connections of complex type") {

      new ChiselStage(true)
        .execute(args, Seq(ChiselGeneratorAnnotation(() => new TopCircuitSyncMem(new MyBundle, withConnection = true))))
      val expectedMatches = Seq(
        (createExpected("~TopCircuitSyncMem\\|TopCircuitSyncMem>mem", "MyBundle\\[4\\]", "SyncReadMem"), 1),
        (createExpected("~TopCircuitSyncMem\\|TopCircuitSyncMem>idx", "UInt<2>", "IO"), 1),
        (createExpected("~TopCircuitSyncMem\\|TopCircuitSyncMem>in", "MyBundle", "IO"), 1),
        (createExpected("~TopCircuitSyncMem\\|TopCircuitSyncMem>in.a", "UInt<8>", "IO"), 1),
        (createExpected("~TopCircuitSyncMem\\|TopCircuitSyncMem>in.b", "SInt<8>", "IO"), 1),
        (createExpected("~TopCircuitSyncMem\\|TopCircuitSyncMem>in.c", "Bool", "IO"), 1),
        (createExpected("~TopCircuitSyncMem\\|TopCircuitSyncMem>out", "MyBundle", "IO"), 1),
        (createExpected("~TopCircuitSyncMem\\|TopCircuitSyncMem>out.a", "UInt<8>", "IO"), 1),
        (createExpected("~TopCircuitSyncMem\\|TopCircuitSyncMem>out.b", "SInt<8>", "IO"), 1),
        (createExpected("~TopCircuitSyncMem\\|TopCircuitSyncMem>out.c", "Bool", "IO"), 1),
        // Tmp wire generated in syncreadmem
        (createExpected("~TopCircuitSyncMem\\|TopCircuitSyncMem>_out_WIRE", "UInt<2>", "Wire"), 1),
        (createExpected("~TopCircuitSyncMem\\|TopCircuitSyncMem>MPORT", "MyBundle", "MemPort"), 1),
        (createExpected("~TopCircuitSyncMem\\|TopCircuitSyncMem>MPORT.a", "UInt<8>", "MemPort"), 1),
        (createExpected("~TopCircuitSyncMem\\|TopCircuitSyncMem>MPORT.b", "SInt<8>", "MemPort"), 1),
        (createExpected("~TopCircuitSyncMem\\|TopCircuitSyncMem>MPORT.c", "Bool", "MemPort"), 1),
        (createExpected("~TopCircuitSyncMem\\|TopCircuitSyncMem>out_MPORT", "MyBundle", "MemPort"), 1),
        (createExpected("~TopCircuitSyncMem\\|TopCircuitSyncMem>out_MPORT.a", "UInt<8>", "MemPort"), 1),
        (createExpected("~TopCircuitSyncMem\\|TopCircuitSyncMem>out_MPORT.b", "SInt<8>", "MemPort"), 1),
        (createExpected("~TopCircuitSyncMem\\|TopCircuitSyncMem>out_MPORT.c", "Bool", "MemPort"), 1),
        (createExpected("~TopCircuitSyncMem\\|TopCircuitSyncMem>clock", "Clock", "IO"), 1),
        (createExpected("~TopCircuitSyncMem\\|TopCircuitSyncMem>reset", "Bool", "IO"), 1)
      )
      checkAnno(expectedMatches, os.read(targetDir / "TopCircuitSyncMem.fir"))
    }

    it("should annotate a Mem with MPORT connections of complex type") {
      new ChiselStage(true)
        .execute(args, Seq(ChiselGeneratorAnnotation(() => new TopCircuitMem(new MyBundle, withConnection = true))))
      val expectedMatches = Seq(
        (createExpected("~TopCircuitMem\\|TopCircuitMem>mem", "MyBundle\\[4\\]", "Mem"), 1),
        (createExpected("~TopCircuitMem\\|TopCircuitMem>idx", "UInt<2>", "IO"), 1),
        (createExpected("~TopCircuitMem\\|TopCircuitMem>in", "MyBundle", "IO"), 1),
        (createExpected("~TopCircuitMem\\|TopCircuitMem>in.a", "UInt<8>", "IO"), 1),
        (createExpected("~TopCircuitMem\\|TopCircuitMem>in.b", "SInt<8>", "IO"), 1),
        (createExpected("~TopCircuitMem\\|TopCircuitMem>in.c", "Bool", "IO"), 1),
        (createExpected("~TopCircuitMem\\|TopCircuitMem>out", "MyBundle", "IO"), 1),
        (createExpected("~TopCircuitMem\\|TopCircuitMem>out.a", "UInt<8>", "IO"), 1),
        (createExpected("~TopCircuitMem\\|TopCircuitMem>out.b", "SInt<8>", "IO"), 1),
        (createExpected("~TopCircuitMem\\|TopCircuitMem>out.c", "Bool", "IO"), 1),
        (createExpected("~TopCircuitMem\\|TopCircuitMem>MPORT", "MyBundle", "MemPort"), 1),
        (createExpected("~TopCircuitMem\\|TopCircuitMem>MPORT.a", "UInt<8>", "MemPort"), 1),
        (createExpected("~TopCircuitMem\\|TopCircuitMem>MPORT.b", "SInt<8>", "MemPort"), 1),
        (createExpected("~TopCircuitMem\\|TopCircuitMem>MPORT.c", "Bool", "MemPort"), 1),
        (createExpected("~TopCircuitMem\\|TopCircuitMem>out_MPORT", "MyBundle", "MemPort"), 1),
        (createExpected("~TopCircuitMem\\|TopCircuitMem>out_MPORT.a", "UInt<8>", "MemPort"), 1),
        (createExpected("~TopCircuitMem\\|TopCircuitMem>out_MPORT.b", "SInt<8>", "MemPort"), 1),
        (createExpected("~TopCircuitMem\\|TopCircuitMem>out_MPORT.c", "Bool", "MemPort"), 1),
        (createExpected("~TopCircuitMem\\|TopCircuitMem>clock", "Clock", "IO"), 1),
        (createExpected("~TopCircuitMem\\|TopCircuitMem>reset", "Bool", "IO"), 1)
      )
      checkAnno(expectedMatches, os.read(targetDir / "TopCircuitMem.fir"))
    }
  }

  describe("Masked Memories Annotations") {
    val targetDir =
      os.pwd / "test_run_dir" / "TywavesAnnotationSpec" / "Masked Memories Annotations"
    val args: Array[String] = Array("--target", "chirrtl", "--target-dir", targetDir.toString)

    it("should annotate a SyncMem with mask") {
      val mSize = 1
      new ChiselStage(true)
        .execute(
          args,
          Seq(ChiselGeneratorAnnotation(() => new TopCircuitMemWithMask(SInt(7.W), classOf[SyncReadMem[SInt]], mSize)))
        )
      // format: off
      val expectedMatches = Seq(
        (createExpected("~TopCircuitMemWithMask\\|TopCircuitMemWithMask>mem", s"SInt<7>\\[$mSize\\]\\[4\\]", "SyncReadMem"), 1),
        (createExpected("~TopCircuitMemWithMask\\|TopCircuitMemWithMask>mask", s"Bool\\[$mSize\\]", "Wire",
          params = Some(Seq(ClassParam("gen", "=> T", None), ClassParam("length", "Int", Some(s"$mSize"))))), 1),
        (createExpected("~TopCircuitMemWithMask\\|TopCircuitMemWithMask>mask\\[0\\]", "Bool", "Wire"), 1),
        (createExpected("~TopCircuitMemWithMask\\|TopCircuitMemWithMask>idx", "UInt<2>", "IO"), 1),
        (createExpected("~TopCircuitMemWithMask\\|TopCircuitMemWithMask>in", s"SInt<7>\\[$mSize\\]", "IO",
          params = Some(Seq(ClassParam("gen", "=> T", None), ClassParam("length", "Int", Some(s"$mSize"))))), 1),
        (createExpected("~TopCircuitMemWithMask\\|TopCircuitMemWithMask>in\\[0\\]", "SInt<7>", "IO"), 1),
        (createExpected("~TopCircuitMemWithMask\\|TopCircuitMemWithMask>out", s"SInt<7>\\[$mSize\\]", "IO",
          params = Some(Seq(ClassParam("gen", "=> T", None), ClassParam("length", "Int", Some(s"$mSize"))))), 1),
        (createExpected("~TopCircuitMemWithMask\\|TopCircuitMemWithMask>out\\[0\\]", "SInt<7>", "IO"), 1),
        // tmp wire generated in syncreadmem
        (createExpected("~TopCircuitMemWithMask\\|TopCircuitMemWithMask>_WIRE", "UInt<2>", "Wire"), 1),
        (createExpected("~TopCircuitMemWithMask\\|TopCircuitMemWithMask>MPORT", s"SInt<7>\\[$mSize\\]", "MemPort",
          params = Some(Seq(ClassParam("gen", "=> T", None), ClassParam("length", "Int", Some(s"$mSize"))))), 1),
        (createExpected("~TopCircuitMemWithMask\\|TopCircuitMemWithMask>MPORT\\[0\\]", "SInt<7>", "MemPort"), 1),
        (createExpected("~TopCircuitMemWithMask\\|TopCircuitMemWithMask>MPORT_1", s"SInt<7>\\[$mSize\\]", "MemPort",
          params = Some(Seq(ClassParam("gen", "=> T", None), ClassParam("length", "Int", Some(s"$mSize"))))), 1),
        (createExpected("~TopCircuitMemWithMask\\|TopCircuitMemWithMask>MPORT_1\\[0\\]", "SInt<7>", "MemPort"), 1),
        (createExpected("~TopCircuitMemWithMask\\|TopCircuitMemWithMask>clock", "Clock", "IO"), 1),
        (createExpected("~TopCircuitMemWithMask\\|TopCircuitMemWithMask>reset", "Bool", "IO"), 1)
      )
      // format: on
      checkAnno(expectedMatches, os.read(targetDir / "TopCircuitMemWithMask.fir"))
    }

    it("should annotate a Mem with mask") {
      val mSize = 1
      new ChiselStage(true)
        .execute(
          args,
          Seq(ChiselGeneratorAnnotation(() => new TopCircuitMemWithMask(SInt(7.W), classOf[Mem[SInt]], mSize)))
        )
      // format: off
      val expectedMatches = Seq(
        (createExpected("~TopCircuitMemWithMask\\|TopCircuitMemWithMask>mem", s"SInt<7>\\[$mSize\\]\\[4\\]", "Mem",
            params = None), 1), // TODO: is it correct NONE?
        (createExpected("~TopCircuitMemWithMask\\|TopCircuitMemWithMask>mask", s"Bool\\[$mSize\\]", "Wire",
            params = Some(Seq(ClassParam("gen", "=> T", None), ClassParam("length", "Int", Some(s"$mSize"))))), 1),
        (createExpected("~TopCircuitMemWithMask\\|TopCircuitMemWithMask>mask\\[0\\]", "Bool", "Wire"), 1),
        (createExpected("~TopCircuitMemWithMask\\|TopCircuitMemWithMask>idx", "UInt<2>", "IO"), 1),
        (createExpected("~TopCircuitMemWithMask\\|TopCircuitMemWithMask>in", s"SInt<7>\\[$mSize\\]", "IO",
            params = Some(Seq(ClassParam("gen", "=> T", None), ClassParam("length", "Int", Some(s"$mSize"))))), 1),
        (createExpected("~TopCircuitMemWithMask\\|TopCircuitMemWithMask>in\\[0\\]", "SInt<7>", "IO"), 1),
        (createExpected("~TopCircuitMemWithMask\\|TopCircuitMemWithMask>out", s"SInt<7>\\[$mSize\\]", "IO",
            params = Some(Seq(ClassParam("gen", "=> T", None), ClassParam("length", "Int", Some(s"$mSize"))))), 1),
        (createExpected("~TopCircuitMemWithMask\\|TopCircuitMemWithMask>out\\[0\\]", "SInt<7>", "IO"), 1),
        // tmp wire generated in syncreadmem
        (createExpected("~TopCircuitMemWithMask\\|TopCircuitMemWithMask>MPORT", s"SInt<7>\\[$mSize\\]", "MemPort",
          params = Some(Seq(ClassParam("gen", "=> T", None), ClassParam("length", "Int", Some(s"$mSize"))))), 1),
        (createExpected("~TopCircuitMemWithMask\\|TopCircuitMemWithMask>MPORT\\[0\\]", "SInt<7>", "MemPort"), 1),
        (createExpected("~TopCircuitMemWithMask\\|TopCircuitMemWithMask>MPORT_1", s"SInt<7>\\[$mSize\\]", "MemPort",
            params = Some(Seq(ClassParam("gen", "=> T", None), ClassParam("length", "Int", Some(s"$mSize"))))), 1),
        (createExpected("~TopCircuitMemWithMask\\|TopCircuitMemWithMask>MPORT_1\\[0\\]", "SInt<7>", "MemPort"), 1),
        (createExpected("~TopCircuitMemWithMask\\|TopCircuitMemWithMask>clock", "Clock", "IO"), 1),
        (createExpected("~TopCircuitMemWithMask\\|TopCircuitMemWithMask>reset", "Bool", "IO"), 1)
      )
      // format: on
      checkAnno(expectedMatches, os.read(targetDir / "TopCircuitMemWithMask.fir"))
    }

    it("should annotate an SRAM with mask") {
      val cName = "TopCircuitSRAMWithMask"
      new ChiselStage(true)
        .execute(
          args,
          Seq(ChiselGeneratorAnnotation(() => new TopCircuitSRAMWithMask(SInt(7.W))))
        )
      val expectedMatches = Seq(
        (createExpected(s"~$cName\\|$cName>mem_sram", "SInt<7>\\[2\\]\\[4\\]", "SramTarget"), 1),
        (
          createExpected(
            s"~$cName\\|$cName>mem.writePorts\\[0\\].mask",
            s"Bool\\[2\\]",
            "Wire",
            params = Some(Seq(ClassParam("gen", "=> T", None), ClassParam("length", "Int", Some("2"))))
          ),
          1
        ),
        (createExpected(s"~$cName\\|$cName>mem.writePorts\\[0\\].mask\\[0\\]", "Bool", "Wire"), 1),
        // Since the inner type is a vector
        (createExpected(s"~$cName\\|$cName>mem.readPorts\\[0\\].data\\[0\\]", "SInt<7>", "Wire"), 1),
        (createExpected(s"~$cName\\|$cName>mem.writePorts\\[0\\].data\\[0\\]", "SInt<7>", "Wire"), 1),
        (createExpected(s"~$cName\\|$cName>clock", "Clock", "IO"), 1),
        (createExpected(s"~$cName\\|$cName>reset", "Bool", "IO"), 1)
      ) ++ createExpectedSRAMs(
        s"~$cName\\|$cName>mem",
        4,
        "SInt<7>\\[2\\]",
        1,
        1,
        0,
        masked = true,
        dataParams = Some(Seq(ClassParam("gen", "=> T", None), ClassParam("length", "Int", Some("2"))))
      )

      checkAnno(expectedMatches, os.read(targetDir / s"$cName.fir"))

    }
  }
}
