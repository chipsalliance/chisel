// SPDX-License-Identifier: Apache-2.0

package firrtlTests

import firrtl._
import firrtl.annotations._
import firrtl.testutils.FirrtlCheckers.{containLine, containLines}
import firrtl.testutils.FirrtlFlatSpec
import firrtlTests.execution._

class MemInitSpec extends FirrtlFlatSpec {
  def compile(circuit: String, annos: AnnotationSeq): CircuitState = {
    (new VerilogCompiler).compileAndEmit(CircuitState(parse(circuit), ChirrtlForm, annos))
  }

  def basicTest(tpe: String = "UInt<32>"): String =
    s"""
       |circuit MemTest:
       |  module MemTest:
       |    input clock : Clock
       |    input rAddr : UInt<5>
       |    input rEnable : UInt<1>
       |    input wAddr : UInt<5>
       |    input wData : $tpe
       |    input wEnable : UInt<1>
       |    output rData : $tpe
       |
       |    mem m:
       |      data-type => $tpe
       |      depth => 32
       |      reader => r
       |      writer => w
       |      read-latency => 1
       |      write-latency => 1
       |      read-under-write => new
       |
       |    m.r.clk <= clock
       |    m.r.addr <= rAddr
       |    m.r.en <= rEnable
       |    rData <= m.r.data
       |
       |    m.w.clk <= clock
       |    m.w.addr <= wAddr
       |    m.w.en <= wEnable
       |    m.w.data <= wData
       |    m.w.mask is invalid
       |
       |""".stripMargin

  val mRef = CircuitTarget("MemTest").module("MemTest").ref("m")

  "NoAnnotation" should "create a randomized initialization" in {
    val annos = Seq()
    val result = compile(basicTest(), annos)
    result should containLine("    m[initvar] = _RAND_0[31:0];")
  }

  "MemoryRandomInitAnnotation" should "create a randomized initialization" in {
    val annos = Seq(MemoryRandomInitAnnotation(mRef))
    val result = compile(basicTest(), annos)
    result should containLine("    m[initvar] = _RAND_0[31:0];")
  }

  "MemoryScalarInitAnnotation w/ 0" should "create an initialization with all zeros" in {
    val annos = Seq(MemoryScalarInitAnnotation(mRef, 0))
    val result = compile(basicTest(), annos)
    result should containLine("    m[initvar] = 0;")
  }

  Seq(1, 3, 30, 400, 12345).foreach { value =>
    s"MemoryScalarInitAnnotation w/ $value" should
      s"create an initialization with all values set to $value" in {
      val annos = Seq(MemoryScalarInitAnnotation(mRef, value))
      val result = compile(basicTest(), annos)
      result should containLine(s"    m[initvar] = $value;")
    }
  }

  "MemoryArrayInitAnnotation" should "initialize all addresses" in {
    val values = Seq.tabulate(32)(ii => 2 * ii + 5).map(BigInt(_))
    val annos = Seq(MemoryArrayInitAnnotation(mRef, values))
    val result = compile(basicTest(), annos)
    values.zipWithIndex.foreach {
      case (value, addr) =>
        result should containLine(s"  m[$addr] = $value;")
    }
  }

  "MemoryScalarInitAnnotation" should "fail for a negative value" in {
    assertThrows[EmitterException] {
      compile(basicTest(), Seq(MemoryScalarInitAnnotation(mRef, -1)))
    }
  }

  "MemoryScalarInitAnnotation" should "fail for a value that is too large" in {
    assertThrows[EmitterException] {
      compile(basicTest(), Seq(MemoryScalarInitAnnotation(mRef, BigInt(1) << 32)))
    }
  }

  "MemoryArrayInitAnnotation" should "fail for a negative value" in {
    assertThrows[EmitterException] {
      val values = Seq.tabulate(32)(_ => BigInt(-1))
      compile(basicTest(), Seq(MemoryArrayInitAnnotation(mRef, values)))
    }
  }

  "MemoryArrayInitAnnotation" should "fail for a value that is too large" in {
    assertThrows[EmitterException] {
      val values = Seq.tabulate(32)(_ => BigInt(1) << 32)
      compile(basicTest(), Seq(MemoryArrayInitAnnotation(mRef, values)))
    }
  }

  "MemoryArrayInitAnnotation" should "fail if the number of values is too small" in {
    assertThrows[EmitterException] {
      val values = Seq.tabulate(31)(_ => BigInt(1))
      compile(basicTest(), Seq(MemoryArrayInitAnnotation(mRef, values)))
    }
  }

  "MemoryArrayInitAnnotation" should "fail if the number of values is too large" in {
    assertThrows[EmitterException] {
      val values = Seq.tabulate(33)(_ => BigInt(1))
      compile(basicTest(), Seq(MemoryArrayInitAnnotation(mRef, values)))
    }
  }

  "MemoryScalarInitAnnotation on Memory with Vector type" should "fail" in {
    val caught = intercept[Exception] {
      val annos = Seq(MemoryScalarInitAnnotation(mRef, 0))
      compile(basicTest("UInt<32>[2]"), annos)
    }
    assert(caught.getMessage.endsWith("Cannot initialize memory m of non ground type UInt<32>[2]"))
  }

  "MemoryScalarInitAnnotation on Memory with Bundle type" should "fail" in {
    val caught = intercept[Exception] {
      val annos = Seq(MemoryScalarInitAnnotation(mRef, 0))
      compile(basicTest("{real: SInt<10>, imag: SInt<10>}"), annos)
    }
    assert(
      caught.getMessage.endsWith("Cannot initialize memory m of non ground type { real : SInt<10>, imag : SInt<10>}")
    )
  }

  private def jsonAnno(name: String, suffix: String): String =
    s"""[{"class": "firrtl.annotations.$name", "target": "~MemTest|MemTest>m"$suffix}]"""

  "MemoryRandomInitAnnotation" should "load from JSON" in {
    val json = jsonAnno("MemoryRandomInitAnnotation", "")
    val annos = JsonProtocol.deserialize(json)
    assert(annos == Seq(MemoryRandomInitAnnotation(mRef)))
  }

  "MemoryScalarInitAnnotation" should "load from JSON" in {
    val json = jsonAnno("MemoryScalarInitAnnotation", """, "value": 1234567890""")
    val annos = JsonProtocol.deserialize(json)
    assert(annos == Seq(MemoryScalarInitAnnotation(mRef, 1234567890)))
  }

  "MemoryArrayInitAnnotation" should "load from JSON" in {
    val json = jsonAnno("MemoryArrayInitAnnotation", """, "values": [10000000000, 20000000000]""")
    val annos = JsonProtocol.deserialize(json)
    val largeSeq = Seq(BigInt("10000000000"), BigInt("20000000000"))
    assert(annos == Seq(MemoryArrayInitAnnotation(mRef, largeSeq)))
  }

  "MemoryFileInlineAnnotation" should "emit $readmemh for text.hex" in {
    val annos = Seq(MemoryFileInlineAnnotation(mRef, filename = "text.hex"))
    val result = compile(basicTest(), annos)
    result should containLine("""$readmemh("text.hex", """ + mRef.name + """);""")
  }

  "MemoryFileInlineAnnotation" should "emit $readmemb for text.bin" in {
    val annos = Seq(MemoryFileInlineAnnotation(mRef, filename = "text.bin", hexOrBinary = MemoryLoadFileType.Binary))
    val result = compile(basicTest(), annos)
    result should containLine("""$readmemb("text.bin", """ + mRef.name + """);""")
  }

  "MemoryFileInlineAnnotation" should "fail with blank filename" in {
    assertThrows[Exception] {
      compile(basicTest(), Seq(MemoryFileInlineAnnotation(mRef, filename = "")))
    }
  }

  "MemoryInitialization" should "emit readmem in `ifndef SYNTHESIS` block by default" in {
    val annos = Seq(
      MemoryFileInlineAnnotation(mRef, filename = "text.hex", hexOrBinary = MemoryLoadFileType.Hex)
    )
    val result = compile(basicTest(), annos)
    result should containLines(
      """`endif // RANDOMIZE""",
      """$readmemh("text.hex", """ + mRef.name + """);""",
      """end // initial"""
    )
  }

  "MemoryInitialization" should "emit readmem outside `ifndef SYNTHESIS` block with MemorySynthInit annotation" in {
    val annos = Seq(
      MemoryFileInlineAnnotation(mRef, filename = "text.hex", hexOrBinary = MemoryLoadFileType.Hex)
    ) ++ Seq(MemorySynthInit)
    val result = compile(basicTest(), annos)
    result should containLines(
      """`endif // SYNTHESIS""",
      """initial begin""",
      """$readmemh("text.hex", """ + mRef.name + """);""",
      """end"""
    )
  }

  "MemoryInitialization" should "emit readmem outside `ifndef SYNTHESIS` block with MemoryNoSynthInit annotation" in {
    val annos = Seq(
      MemoryFileInlineAnnotation(mRef, filename = "text.hex", hexOrBinary = MemoryLoadFileType.Hex)
    ) ++ Seq(MemoryNoSynthInit)

    val result = compile(basicTest(), annos)
    result should containLines(
      """`endif // RANDOMIZE""",
      """$readmemh("text.hex", """ + mRef.name + """);""",
      """end // initial"""
    )
  }

  /** Firrtl which contains a child memory module instantiated twice
    * If deduplication occurs, the firrtl for the child module should appear only
    * once
    * Any non-local memory annotations bound to the 'm' memories in each child instance
    * should properly deduplicate in order for them to be emitted in the Verilog
    */
  def dedupTest =
    s"""
       |circuit Top:
       |  module Child:
       |    input clock : Clock
       |    input rAddr : UInt<5>
       |    input rEnable : UInt<1>
       |    input wAddr : UInt<5>
       |    input wData : UInt<8>
       |    input wEnable : UInt<1>
       |    output rData : UInt<8>
       |
       |    mem m:
       |      data-type => UInt<8>
       |      depth => 32
       |      reader => r
       |      writer => w
       |      read-latency => 1
       |      write-latency => 1
       |      read-under-write => new
       |
       |    m.r.clk <= clock
       |    m.r.addr <= rAddr
       |    m.r.en <= rEnable
       |    rData <= m.r.data
       |
       |    m.w.clk <= clock
       |    m.w.addr <= wAddr
       |    m.w.en <= wEnable
       |    m.w.data <= wData
       |    m.w.mask is invalid
       |
       |  module Top:
       |    input clock : Clock
       |    input rAddr : UInt<5>
       |    input rEnable : UInt<1>
       |    input wAddr : UInt<5>
       |    input wData : UInt<8>
       |    input wEnable : UInt<1>
       |    output rData : UInt<8>[2]
       |
       |    inst c1 of Child
       |    c1.clock <= clock
       |    c1.rAddr <= rAddr
       |    c1.rEnable <= rEnable
       |    c1.wAddr <= wAddr
       |    c1.wData <= wData
       |    c1.wEnable <= wEnable
       |
       |    inst c2 of Child
       |    c2.clock <= clock
       |    c2.rAddr <= rAddr
       |    c2.rEnable <= rEnable
       |    c2.wAddr <= wAddr
       |    c2.wData <= wData
       |    c2.wEnable <= wEnable
       |
       |    rData[0] <= c1.rData
       |    rData[1] <= c2.rData
       |""".stripMargin

  // Absolute references to the memory objects in each child module
  val child1MRef = CircuitTarget("Top").module("Top").instOf("c1", "Child").ref("m")
  val child2MRef = CircuitTarget("Top").module("Top").instOf("c2", "Child").ref("m")
  // Final deduplicated reference
  val dedupedRef = CircuitTarget("Top").module("Child").ref("m")

  "MemoryRandomInitAnnotation" should "randomize memory in single deduped module" in {
    val annos = Seq(
      MemoryRandomInitAnnotation(child1MRef),
      MemoryRandomInitAnnotation(child2MRef)
    )
    val result = compile(dedupTest, annos)
    result should containLine("      m[initvar] = _RAND_0[7:0];")
  }

  "MemoryScalarInitAnnotation" should "initialize memory to 0 in deduped module" in {
    val annos = Seq(
      MemoryScalarInitAnnotation(child1MRef, value = 0),
      MemoryScalarInitAnnotation(child2MRef, value = 0)
    )
    val result = compile(dedupTest, annos)
    result should containLine("      m[initvar] = 0;")
  }

  "MemoryArrayInitAnnotation" should "initialize memory with array of values in deduped module" in {
    val values = Seq.tabulate(32)(ii => 2 * ii + 5).map(BigInt(_))
    val annos = Seq(
      MemoryArrayInitAnnotation(child1MRef, values),
      MemoryArrayInitAnnotation(child2MRef, values)
    )
    val result = compile(dedupTest, annos)

    values.zipWithIndex.foreach {
      case (value, addr) =>
        result should containLine(s"      m[$addr] = $value;")
    }
  }

  "MemoryFileInlineAnnotation" should "emit $readmemh in deduped module" in {
    val annos = Seq(
      MemoryFileInlineAnnotation(child1MRef, filename = "text.hex"),
      MemoryFileInlineAnnotation(child2MRef, filename = "text.hex")
    )
    val result = compile(dedupTest, annos)
    result should containLine("""$readmemh("text.hex", """ + dedupedRef.name + """);""")
  }

  "MemoryFileInlineAnnotation" should "fail dedup if not all instances have the annotation" in {
    val annos = Seq(
      MemoryFileInlineAnnotation(child1MRef, filename = "text.hex")
    )
    assertThrows[FirrtlUserException] {
      compile(dedupTest, annos)
    }
  }

  "MemoryFileInlineAnnotation" should "fail dedup if instances have different init files" in {
    val annos = Seq(
      MemoryFileInlineAnnotation(child1MRef, filename = "text.hex"),
      MemoryFileInlineAnnotation(child2MRef, filename = "text.bin")
    )
    assertThrows[FirrtlUserException] {
      compile(dedupTest, annos)
    }
  }
}

abstract class MemInitExecutionSpec(values: Seq[Int], init: ReferenceTarget => Annotation)
    extends SimpleExecutionTest
    with VerilogExecution {
  override val body: String =
    s"""
       |mem m:
       |  data-type => UInt<32>
       |  depth => ${values.length}
       |  reader => r
       |  read-latency => 1
       |  write-latency => 1
       |  read-under-write => new
       |m.r.clk <= clock
       |m.r.en <= UInt<1>(1)
       |""".stripMargin

  val mRef = CircuitTarget("dut").module("dut").ref("m")
  override val customAnnotations: AnnotationSeq = Seq(init(mRef))

  override def commands: Seq[SimpleTestCommand] = (Seq(-1) ++ values).zipWithIndex.map {
    case (value, addr) =>
      if (value == -1) { Seq(Poke("m.r.addr", addr)) }
      else if (addr >= values.length) { Seq(Expect("m.r.data", value)) }
      else { Seq(Poke("m.r.addr", addr), Expect("m.r.data", value)) }
  }.flatMap(_ ++ Seq(Step(1)))
}

class MemScalarInit0ExecutionSpec
    extends MemInitExecutionSpec(
      Seq.tabulate(31)(_ => 0),
      r => MemoryScalarInitAnnotation(r, 0)
    ) {}

class MemScalarInit17ExecutionSpec
    extends MemInitExecutionSpec(
      Seq.tabulate(31)(_ => 17),
      r => MemoryScalarInitAnnotation(r, 17)
    ) {}

class MemArrayInitExecutionSpec
    extends MemInitExecutionSpec(
      Seq.tabulate(31)(ii => ii * 5 + 7),
      r => MemoryArrayInitAnnotation(r, Seq.tabulate(31)(ii => ii * 5 + 7).map(BigInt(_)))
    ) {}
