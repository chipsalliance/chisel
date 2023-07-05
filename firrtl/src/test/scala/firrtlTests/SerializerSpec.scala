// SPDX-License-Identifier: Apache-2.0

package firrtlTests

import org.scalatest._
import firrtl.CDefMemory
import firrtl.ir._
import firrtl.{Parser, Utils}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

object SerializerSpec {
  case class WrapStmt(stmt: Statement) extends Statement {
    def serialize: String = s"wrap(${stmt.serialize})"
  }

  case class WrapExpr(expr: Expression) extends Expression {
    def serialize: String = s"wrap(${expr.serialize})"
    def tpe:       Type = expr.tpe
  }

  private def tab(s: String): String = {
    // Careful to only tab non-empty lines
    s.split("\n")
      .map { line =>
        if (line.nonEmpty) Serializer.Indent + line else line
      }
      .mkString("\n")
  }

  val testModule: String =
    """module test :
      |  input in : UInt<8>
      |  output out : UInt<8>
      |
      |  inst c of child
      |  connect c.in, in
      |  connect out, c.out""".stripMargin

  val testModuleTabbed: String = tab(testModule)

  val testModuleIR: Module =
    Module(
      NoInfo,
      "test",
      Seq(Port(NoInfo, "in", Input, UIntType(IntWidth(8))), Port(NoInfo, "out", Output, UIntType(IntWidth(8)))),
      Block(
        Seq(
          DefInstance("c", "child"),
          Connect(NoInfo, SubField(Reference("c"), "in"), Reference("in")),
          Connect(NoInfo, Reference("out"), SubField(Reference("c"), "out"))
        )
      )
    )

  val childModule: String =
    """extmodule child :
      |  input in : UInt<8>
      |  output out : UInt<8>
      |  defname = child""".stripMargin

  val childModuleTabbed: String = tab(childModule)

  val childModuleIR: ExtModule = ExtModule(
    NoInfo,
    "child",
    Seq(Port(NoInfo, "in", Input, UIntType(IntWidth(8))), Port(NoInfo, "out", Output, UIntType(IntWidth(8)))),
    "child",
    Seq.empty
  )

  val simpleCircuit: String =
    s"FIRRTL version ${Serializer.version.serialize}\ncircuit test :\n" + childModuleTabbed + "\n\n" + testModuleTabbed + "\n"

  val simpleCircuitIR: Circuit =
    Circuit(
      NoInfo,
      Seq(
        childModuleIR,
        testModuleIR
      ),
      "test"
    )

}

/** used to test parsing and serialization of smems */
object SMemTestCircuit {
  def src(ruw: String): String =
    s"""circuit Example :
       |  module Example :
       |    smem mem : UInt<8> [8] $ruw@[main.scala 10:25]
       |""".stripMargin

  def circuit(ruw: ReadUnderWrite.Value): Circuit =
    Circuit(
      NoInfo,
      Seq(
        Module(
          NoInfo,
          "Example",
          Seq.empty,
          Block(
            CDefMemory(
              NoInfo,
              "mem",
              UIntType(IntWidth(8)),
              8,
              true,
              ruw
            )
          )
        )
      ),
      "Example"
    )

  def findRuw(c: Circuit): ReadUnderWrite.Value = {
    val main = c.modules.head.asInstanceOf[Module]
    val mem = main.body.asInstanceOf[Block].stmts.collectFirst { case m: CDefMemory => m }.get
    mem.readUnderWrite
  }
}

class SerializerSpec extends AnyFlatSpec with Matchers {
  import SerializerSpec._

  "ir.Serializer" should "support custom Statements" in {
    val stmt = WrapStmt(DefWire(NoInfo, "myWire", Utils.BoolType))
    val ser = "wrap(wire myWire : UInt<1>)"
    Serializer.serialize(stmt) should be(ser)
  }

  it should "support custom Expression" in {
    val expr = WrapExpr(Reference("foo"))
    val ser = "wrap(foo)"
    Serializer.serialize(expr) should be(ser)
  }

  it should "support nested custom Statements and Expressions" in {
    val expr = SubField(WrapExpr(Reference("foo")), "bar")
    val stmt = WrapStmt(DefNode(NoInfo, "n", expr))
    val stmts = Block(stmt :: Nil)
    val ser = "wrap(node n = wrap(foo).bar)"
    Serializer.serialize(stmts) should be(ser)
  }

  it should "support emitting circuits" in {
    val serialized = Serializer.serialize(simpleCircuitIR)
    serialized should be(simpleCircuit)
  }

  it should "support emitting individual modules" in {
    val serialized = Serializer.serialize(testModuleIR)
    serialized should be(testModule)
  }

  it should "support emitting indented individual modules" in {
    val serialized = Serializer.serialize(testModuleIR, 1)
    serialized should be(testModuleTabbed)
  }

  it should "support emitting indented individual extmodules" in {
    val serialized = Serializer.serialize(childModuleIR, 1)
    serialized should be(childModuleTabbed)
  }

  it should "support emitting const types" in {
    val constInt = DefWire(NoInfo, "constInt", ConstType(UIntType(IntWidth(3))))
    Serializer.serialize(constInt) should be("wire constInt : const UInt<3>")

    val constAsyncReset = DefWire(NoInfo, "constAsyncReset", ConstType(AsyncResetType))
    Serializer.serialize(constAsyncReset) should be("wire constAsyncReset : const AsyncReset")

    val constInput = Port(NoInfo, "in", Input, ConstType(SIntType(IntWidth(8))))
    Serializer.serialize(constInput) should be("input in : const SInt<8>")

    val constBundle = DefWire(
      NoInfo,
      "constBundle",
      ConstType(
        BundleType(
          Seq(
            Field("foo", Default, UIntType(IntWidth(32))),
            Field("bar", Default, ConstType(SIntType(IntWidth(1))))
          )
        )
      )
    )
    Serializer.serialize(constBundle) should be(
      "wire constBundle : const { foo : UInt<32>, bar : const SInt<1>}"
    )

    val constVec = DefWire(NoInfo, "constVec", VectorType(ClockType, 10))
    Serializer.serialize(constVec) should be("wire constVec : Clock[10]")
  }

  it should "emit whens with empty Blocks correctly" in {
    val when = Conditionally(NoInfo, Reference("cond"), Block(Seq()), EmptyStmt)
    val serialized = Serializer.serialize(when, 1)
    serialized should be("  when cond :\n    skip\n")
  }

  it should "serialize read-under-write behavior for smems correctly" in {
    (SMemTestCircuit.circuit(ReadUnderWrite.Undefined).serialize should not).include("undefined")
    SMemTestCircuit.circuit(ReadUnderWrite.New).serialize should include("new")
    SMemTestCircuit.circuit(ReadUnderWrite.Old).serialize should include("old")
  }

  it should "support emitting Probe/RWProbe types and related expressions/statements" in {
    val probeInt = DefWire(NoInfo, "foo", ProbeType(UIntType(IntWidth(3))))
    Serializer.serialize(probeInt) should be("wire foo : Probe<UInt<3>>")

    val rwProbeBundle = Port(
      NoInfo,
      "foo",
      Output,
      RWProbeType(BundleType(Seq(Field("bar", Default, UIntType(IntWidth(32))))))
    )
    Serializer.serialize(rwProbeBundle) should be("output foo : RWProbe<{ bar : UInt<32>}>")

    val probeVec = Port(
      NoInfo,
      "foo",
      Output,
      RWProbeType(VectorType(UIntType(IntWidth(32)), 4))
    )
    Serializer.serialize(probeVec) should be("output foo : RWProbe<UInt<32>[4]>")

    val probeDefine = ProbeDefine(NoInfo, SubField(Reference("c"), "in"), ProbeExpr(Reference("in")))
    Serializer.serialize(probeDefine) should be("define c.in = probe(in)")

    val rwProbeDefine = ProbeDefine(NoInfo, SubField(Reference("c"), "in"), RWProbeExpr(Reference("in")))
    Serializer.serialize(rwProbeDefine) should be("define c.in = rwprobe(in)")

    val probeRead = Connect(NoInfo, Reference("out"), ProbeRead(Reference("c.out")))
    Serializer.serialize(probeRead) should be("connect out, read(c.out)")

    val probeForceInitial = ProbeForceInitial(NoInfo, Reference("outProbe"), UIntLiteral(100, IntWidth(8)))
    Serializer.serialize(probeForceInitial) should be("force_initial(outProbe, UInt<8>(0h64))")

    val probeReleaseInitial = ProbeReleaseInitial(NoInfo, Reference("outProbe"))
    Serializer.serialize(probeReleaseInitial) should be("release_initial(outProbe)")

    val probeForce = ProbeForce(NoInfo, Reference("clock"), Reference("cond"), Reference("outProbe"), Reference("in"))
    Serializer.serialize(probeForce) should be("force(clock, cond, outProbe, in)")

    val probeRelease = ProbeRelease(NoInfo, Reference("clock"), Reference("cond"), Reference("outProbe"))
    Serializer.serialize(probeRelease) should be("release(clock, cond, outProbe)")
  }

  it should "support lazy serialization" in {
    var stmtSerialized = false
    case class HackStmt(stmt: Statement) extends Statement {
      def serialize: String = {
        stmtSerialized = true
        stmt.serialize
      }
    }

    val stmt = HackStmt(DefNode(NoInfo, "foo", Reference("bar")))
    val it: Iterable[String] = Serializer.lazily(stmt)
    assert(!stmtSerialized, "We should be able to construct the serializer lazily")

    var mapExecuted = false
    val it2: Iterable[String] = it.map { x =>
      mapExecuted = true
      x + ","
    }
    assert(!stmtSerialized && !mapExecuted, "We should be able to map the serializer lazily")

    var appendExecuted = false
    val it3: Iterable[String] = it2 ++ Seq("hi").view.map { x =>
      appendExecuted = true
      x
    }
    assert(!stmtSerialized && !mapExecuted && !appendExecuted, "We should be able to append to the serializer lazily")

    val result = it3.mkString
    assert(
      stmtSerialized && mapExecuted && appendExecuted,
      "Once we traverse the serializer, everything should execute"
    )
  }

  it should "add backticks to names which begin with a numeric character" in {
    info("circuit okay!")
    Serializer.serialize(Circuit(NoInfo, Seq.empty[DefModule], "42_Circuit")) should include("circuit `42_Circuit`")

    info("modules okay!")
    Serializer.serialize(Module(NoInfo, "42_module", Seq.empty, Block(Seq.empty))) should include("module `42_module`")
    // TODO: an external module with a numeric defname should probably be rejected
    Serializer.serialize(ExtModule(NoInfo, "42_extmodule", Seq.empty, "<TODO>", Seq.empty)) should include(
      "extmodule `42_extmodule`"
    )
    Serializer.serialize(IntModule(NoInfo, "42_intmodule", Seq.empty, "foo", Seq.empty)) should include(
      "intmodule `42_intmodule`"
    )

    info("ports okay!")
    Serializer.serialize(Port(NoInfo, "42_port", Input, UIntType(IntWidth(1)))) should include("input `42_port`")

    info("types okay!")
    Serializer.serialize(BundleType(Seq(Field("42_field", Default, UIntType(IntWidth(1)))))) should include(
      "{ `42_field` : UInt<1>}"
    )

    info("declarations okay!")
    Serializer.serialize(DefNode(NoInfo, "42_dest", Reference("42_src"))) should include("node `42_dest` = `42_src`")
    Serializer.serialize(DefWire(NoInfo, "42_wire", UIntType(IntWidth(1)))) should include("wire `42_wire`")
    Serializer.serialize(DefRegister(NoInfo, "42_reg", UIntType(IntWidth(1)), Reference("42_clock"))) should include(
      "reg `42_reg` : UInt<1>, `42_clock`"
    )
    Serializer.serialize(
      DefRegisterWithReset(
        NoInfo,
        "42_regreset",
        UIntType(IntWidth(1)),
        Reference("42_clock"),
        Reference("42_reset"),
        Reference("42_init")
      )
    ) should include("regreset `42_regreset` : UInt<1>, `42_clock`, `42_reset`, `42_init`")
    Serializer.serialize(DefInstance(NoInfo, "42_inst", "42_module")) should include("inst `42_inst` of `42_module`")
    (Serializer
      .serialize(
        DefMemory(
          NoInfo,
          "42_mem",
          UIntType(IntWidth(1)),
          8,
          1,
          1,
          Seq("42_r"),
          Seq("42_w"),
          Seq("42_rw"),
          ReadUnderWrite.Undefined
        )
      )
      .split('\n')
      .map(_.dropWhile(_ == ' ')) should contain).allOf(
      "mem `42_mem` :",
      "reader => `42_r`",
      "writer => `42_w`",
      "readwriter => `42_rw`"
    )
    Serializer.serialize(
      CDefMemory(NoInfo, "42_cmem", UIntType(IntWidth(1)), 8, true, ReadUnderWrite.Undefined)
    ) should include("smem `42_cmem`")
    Serializer.serialize(
      firrtl.CDefMPort(
        NoInfo,
        "42_memport",
        UIntType(IntWidth(1)),
        "42_mem",
        Seq(UIntLiteral(0, IntWidth(1)), Reference("42_clock")),
        firrtl.MRead
      )
    ) should include("mport `42_memport` = `42_mem`[UInt<1>(0h0)], `42_clock`")

    info("labeled statement okay!")
    Serializer.serialize(
      Stop(NoInfo, 1, Reference("42_clock"), Reference("42_enable"), "42_label")
    ) should include("stop(`42_clock`, `42_enable`, 1) : `42_label`")
    Serializer.serialize(
      Print(
        NoInfo,
        StringLit("hello %x"),
        Seq(Reference("42_arg")),
        Reference("42_clock"),
        Reference("42_enable"),
        "42_label"
      )
    ) should include("""printf(`42_clock`, `42_enable`, "hello %x", `42_arg`) : `42_label`""")
    Serializer.serialize(
      Verification(
        Formal.Assert,
        NoInfo,
        Reference("42_clock"),
        Reference("42_predicate"),
        Reference("42_enable"),
        StringLit("message"),
        "42_label"
      )
    ) should include("""assert(`42_clock`, `42_predicate`, `42_enable`, "message") : `42_label`""")
  }
}
