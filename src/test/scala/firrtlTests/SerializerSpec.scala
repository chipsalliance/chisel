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
    def foreachExpr(f:   Expression => Unit):       Unit = stmt.foreachExpr(f)
    def foreachInfo(f:   Info => Unit):             Unit = stmt.foreachInfo(f)
    def foreachStmt(f:   Statement => Unit):        Unit = stmt.foreachStmt(f)
    def foreachString(f: String => Unit):           Unit = stmt.foreachString(f)
    def foreachType(f:   Type => Unit):             Unit = stmt.foreachType(f)
    def mapExpr(f:       Expression => Expression): Statement = this.copy(stmt.mapExpr(f))
    def mapInfo(f:       Info => Info):             Statement = this.copy(stmt.mapInfo(f))
    def mapStmt(f:       Statement => Statement):   Statement = this.copy(stmt.mapStmt(f))
    def mapString(f:     String => String):         Statement = this.copy(stmt.mapString(f))
    def mapType(f:       Type => Type):             Statement = this.copy(stmt.mapType(f))
  }

  case class WrapExpr(expr: Expression) extends Expression {
    def serialize: String = s"wrap(${expr.serialize})"
    def tpe:       Type = expr.tpe
    def foreachExpr(f:  Expression => Unit):       Unit = expr.foreachExpr(f)
    def foreachType(f:  Type => Unit):             Unit = expr.foreachType(f)
    def foreachWidth(f: Width => Unit):            Unit = expr.foreachWidth(f)
    def mapExpr(f:      Expression => Expression): Expression = this.copy(expr.mapExpr(f))
    def mapType(f:      Type => Type):             Expression = this.copy(expr.mapType(f))
    def mapWidth(f:     Width => Width):           Expression = this.copy(expr.mapWidth(f))
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
    "circuit test :\n" + childModuleTabbed + "\n\n" + testModuleTabbed + "\n"
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

}
