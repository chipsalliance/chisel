// See LICENSE for license details.

package firrtlTests

import firrtl._
import firrtl.annotations._
import firrtl.testutils.FirrtlCheckers.containLine
import firrtl.testutils.FirrtlFlatSpec
import firrtlTests.execution._

class MemInitSpec extends FirrtlFlatSpec {
  def input(tpe: String): String =
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
  def compile(annos: AnnotationSeq, tpe: String = "UInt<32>"): CircuitState = {
    (new VerilogCompiler).compileAndEmit(CircuitState(parse(input(tpe)), ChirrtlForm, annos))
  }

  "NoAnnotation" should "create a randomized initialization" in {
    val annos = Seq()
    val result = compile(annos)
    result should containLine ("    m[initvar] = _RAND_0[31:0];")
  }

  "MemoryRandomInitAnnotation" should "create a randomized initialization" in {
    val annos = Seq(MemoryRandomInitAnnotation(mRef))
    val result = compile(annos)
    result should containLine ("    m[initvar] = _RAND_0[31:0];")
  }

  "MemoryScalarInitAnnotation w/ 0" should "create an initialization with all zeros" in {
    val annos = Seq(MemoryScalarInitAnnotation(mRef, 0))
    val result = compile(annos)
    result should containLine("    m[initvar] = 0;")
  }

  Seq(1, 3, 30, 400, 12345).foreach { value =>
    s"MemoryScalarInitAnnotation w/ $value" should
      s"create an initialization with all values set to $value" in {
      val annos = Seq(MemoryScalarInitAnnotation(mRef, value))
      val result = compile(annos)
      result should containLine(s"    m[initvar] = $value;")
    }
  }

  "MemoryArrayInitAnnotation" should "initialize all addresses" in {
    val values = Seq.tabulate(32)(ii => 2 * ii + 5).map(BigInt(_))
    val annos = Seq(MemoryArrayInitAnnotation(mRef, values))
    val result = compile(annos)
    values.zipWithIndex.foreach { case (value, addr) =>
      result should containLine(s"  m[$addr] = $value;")
    }
  }

  "MemoryScalarInitAnnotation" should "fail for a negative value" in {
    assertThrows[EmitterException] {
      compile(Seq(MemoryScalarInitAnnotation(mRef, -1)))
    }
  }

  "MemoryScalarInitAnnotation" should "fail for a value that is too large" in {
    assertThrows[EmitterException] {
      compile(Seq(MemoryScalarInitAnnotation(mRef, BigInt(1) << 32)))
    }
  }

  "MemoryArrayInitAnnotation" should "fail for a negative value" in {
    assertThrows[EmitterException] {
      val values = Seq.tabulate(32)(_ => BigInt(-1))
      compile(Seq(MemoryArrayInitAnnotation(mRef, values)))
    }
  }

  "MemoryArrayInitAnnotation" should "fail for a value that is too large" in {
    assertThrows[EmitterException] {
      val values = Seq.tabulate(32)(_ => BigInt(1) << 32)
      compile(Seq(MemoryArrayInitAnnotation(mRef, values)))
    }
  }

  "MemoryArrayInitAnnotation" should "fail if the number of values is too small" in {
    assertThrows[EmitterException] {
      val values = Seq.tabulate(31)(_ => BigInt(1))
      compile(Seq(MemoryArrayInitAnnotation(mRef, values)))
    }
  }

  "MemoryArrayInitAnnotation" should "fail if the number of values is too large" in {
    assertThrows[EmitterException] {
      val values = Seq.tabulate(33)(_ => BigInt(1))
      compile(Seq(MemoryArrayInitAnnotation(mRef, values)))
    }
  }

  "MemoryScalarInitAnnotation on Memory with Vector type" should "fail" in {
    val caught = intercept[Exception] {
      val annos = Seq(MemoryScalarInitAnnotation(mRef, 0))
      compile(annos, "UInt<32>[2]")
    }
    assert(caught.getMessage.endsWith("Cannot initialize memory m of non ground type UInt<32>[2]"))
  }

  "MemoryScalarInitAnnotation on Memory with Bundle type" should "fail" in {
    val caught = intercept[Exception] {
      val annos = Seq(MemoryScalarInitAnnotation(mRef, 0))
      compile(annos, "{real: SInt<10>, imag: SInt<10>}")
    }
    assert(caught.getMessage.endsWith("Cannot initialize memory m of non ground type { real : SInt<10>, imag : SInt<10>}"))
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

}

abstract class MemInitExecutionSpec(values: Seq[Int], init: ReferenceTarget => Annotation)
  extends SimpleExecutionTest with VerilogExecution {
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

  override def commands: Seq[SimpleTestCommand] = (Seq(-1) ++  values).zipWithIndex.map { case (value, addr) =>
    if(value == -1) { Seq(Poke("m.r.addr", addr)) }
    else if(addr >= values.length) { Seq(Expect("m.r.data", value)) }
    else { Seq(Poke("m.r.addr", addr), Expect("m.r.data", value)) }
  }.flatMap(_ ++ Seq(Step(1)))
}

class MemScalarInit0ExecutionSpec extends MemInitExecutionSpec(
  Seq.tabulate(31)(_ => 0), r => MemoryScalarInitAnnotation(r, 0)
) {}

class MemScalarInit17ExecutionSpec extends MemInitExecutionSpec(
  Seq.tabulate(31)(_ => 17), r => MemoryScalarInitAnnotation(r, 17)
) {}

class MemArrayInitExecutionSpec extends MemInitExecutionSpec(
  Seq.tabulate(31)(ii => ii * 5 + 7),
  r => MemoryArrayInitAnnotation(r, Seq.tabulate(31)(ii => ii * 5 + 7).map(BigInt(_)))
) {}
