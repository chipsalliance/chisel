// SPDX-License-Identifier: Apache-2.0

package firrtlTests

import org.scalatest._
import firrtl.ir._
import firrtl.{Parser, Utils}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

object SerializerSpec {
  case class WrapStmt(stmt: Statement) extends Statement {
    def serialize: String = s"wrap(${stmt.serialize})"
    def foreachExpr(f:   Expression => Unit): Unit = stmt.foreachExpr(f)
    def foreachInfo(f:   Info => Unit):       Unit = stmt.foreachInfo(f)
    def foreachStmt(f:   Statement => Unit):  Unit = stmt.foreachStmt(f)
    def foreachString(f: String => Unit):     Unit = stmt.foreachString(f)
    def foreachType(f:   Type => Unit):       Unit = stmt.foreachType(f)
  }

  case class WrapExpr(expr: Expression) extends Expression {
    def serialize: String = s"wrap(${expr.serialize})"
    def tpe:       Type = expr.tpe
    def foreachExpr(f:  Expression => Unit): Unit = expr.foreachExpr(f)
    def foreachType(f:  Type => Unit):       Unit = expr.foreachType(f)
    def foreachWidth(f: Width => Unit):      Unit = expr.foreachWidth(f)
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

  val childModule: String =
    """extmodule child :
      |  input in : UInt<8>
      |  output out : UInt<8>
      |  defname = child""".stripMargin

  val childModuleTabbed: String = tab(childModule)

  val simpleCircuit: String =
    s"FIRRTL version ${Serializer.version.serialize}\ncircuit test :\n" + childModuleTabbed + "\n\n" + testModuleTabbed + "\n"
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
    val parsed = Parser.parse(simpleCircuit)
    val serialized = Serializer.serialize(parsed)
    serialized should be(simpleCircuit)
  }

  it should "support emitting individual modules" in {
    val parsed = Parser.parse(simpleCircuit)
    val m = parsed.modules.find(_.name == "test").get
    val serialized = Serializer.serialize(m)
    serialized should be(testModule)
  }

  it should "support emitting indented individual modules" in {
    val parsed = Parser.parse(simpleCircuit)
    val m = parsed.modules.find(_.name == "test").get
    val serialized = Serializer.serialize(m, 1)
    serialized should be(testModuleTabbed)
  }

  it should "support emitting indented individual extmodules" in {
    val parsed = Parser.parse(simpleCircuit)
    val m = parsed.modules.find(_.name == "child").get
    val serialized = Serializer.serialize(m, 1)
    serialized should be(childModuleTabbed)
  }

  it should "emit whens with empty Blocks correctly" in {
    val when = Conditionally(NoInfo, Reference("cond"), Block(Seq()), EmptyStmt)
    val serialized = Serializer.serialize(when, 1)
    serialized should be("  when cond :\n    skip\n")
  }

  it should "serialize read-under-write behavior for smems correctly" in {
    def parseSerializeParse(src: String): Circuit = Parser.parse(Parser.parse(src).serialize)
    val undefined = parseSerializeParse(SMemTestCircuit.src(""))
    assert(SMemTestCircuit.findRuw(undefined) == ReadUnderWrite.Undefined)

    val undefined2 = parseSerializeParse(SMemTestCircuit.src(" undefined"))
    assert(SMemTestCircuit.findRuw(undefined2) == ReadUnderWrite.Undefined)

    val old = parseSerializeParse(SMemTestCircuit.src(" old"))
    assert(SMemTestCircuit.findRuw(old) == ReadUnderWrite.Old)

    val readNew = parseSerializeParse(SMemTestCircuit.src(" new"))
    assert(SMemTestCircuit.findRuw(readNew) == ReadUnderWrite.New)
  }
}
