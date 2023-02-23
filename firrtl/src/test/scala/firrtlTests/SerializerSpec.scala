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
      |  c.in <= in
      |  out <= c.out""".stripMargin

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

  val constModule: String =
    """module test :
      |
      |  wire constInt : const UInt<3>
      |  wire constAsyncReset : const AsyncReset
      |  wire constBundle : const { real : UInt<32>, imag : UInt<32>, other : const SInt<1>}""".stripMargin

  val constsModuleIR: Module =
    Module(
      NoInfo,
      "test",
      Seq.empty,
      Block(
        Seq(
          DefWire(NoInfo, "constInt", ConstGroundType(UIntType(IntWidth(3)))),
          DefWire(NoInfo, "constAsyncReset", ConstGroundType(AsyncResetType)),
          DefWire(NoInfo, "constBundle", ConstAggregateType(BundleType(Seq(
            Field("real", Default, UIntType(IntWidth(32))),
            Field("imag", Default, UIntType(IntWidth(32))),
            Field("other", Default, ConstGroundType(SIntType(IntWidth(1)))),
          )))),
        )
      )
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
    val serialized = Serializer.serialize(constsModuleIR)
    serialized should be(constModule)
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
}
