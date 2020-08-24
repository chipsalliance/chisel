// See LICENSE for license details.

package firrtlTests

import org.scalatest._
import firrtl.ir._
import firrtl.Utils
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

object SerializerSpec {
  case class WrapStmt(stmt: Statement) extends Statement {
    def serialize: String = s"wrap(${stmt.serialize})"
    def foreachExpr(f: Expression => Unit): Unit = stmt.foreachExpr(f)
    def foreachInfo(f: Info => Unit): Unit = stmt.foreachInfo(f)
    def foreachStmt(f: Statement => Unit): Unit = stmt.foreachStmt(f)
    def foreachString(f: String => Unit): Unit = stmt.foreachString(f)
    def foreachType(f: Type => Unit): Unit = stmt.foreachType(f)
    def mapExpr(f: Expression => Expression): Statement = this.copy(stmt.mapExpr(f))
    def mapInfo(f: Info => Info): Statement = this.copy(stmt.mapInfo(f))
    def mapStmt(f: Statement => Statement): Statement = this.copy(stmt.mapStmt(f))
    def mapString(f: String => String): Statement = this.copy(stmt.mapString(f))
    def mapType(f: Type => Type): Statement = this.copy(stmt.mapType(f))
  }

  case class WrapExpr(expr: Expression) extends Expression {
    def serialize: String = s"wrap(${expr.serialize})"
    def tpe: Type = expr.tpe
    def foreachExpr(f: Expression => Unit): Unit = expr.foreachExpr(f)
    def foreachType(f: Type => Unit): Unit = expr.foreachType(f)
    def foreachWidth(f: Width => Unit): Unit = expr.foreachWidth(f)
    def mapExpr(f: Expression => Expression): Expression = this.copy(expr.mapExpr(f))
    def mapType(f: Type => Type): Expression = this.copy(expr.mapType(f))
    def mapWidth(f: Width => Width): Expression = this.copy(expr.mapWidth(f))
  }
}

class SerializerSpec extends AnyFlatSpec with Matchers {
  import SerializerSpec._

  "ir.Serializer" should "support custom Statements" in {
    val stmt = WrapStmt(DefWire(NoInfo, "myWire", Utils.BoolType))
    val ser = "wrap(wire myWire : UInt<1>)"
    Serializer.serialize(stmt) should be (ser)
  }

  it should "support custom Expression" in {
    val expr = WrapExpr(Reference("foo"))
    val ser = "wrap(foo)"
    Serializer.serialize(expr) should be (ser)
  }

  it should "support nested custom Statements and Expressions" in {
    val expr = SubField(WrapExpr(Reference("foo")), "bar")
    val stmt = WrapStmt(DefNode(NoInfo, "n", expr))
    val stmts = Block(stmt :: Nil)
    val ser = "wrap(node n = wrap(foo).bar)"
    Serializer.serialize(stmts) should be (ser)
  }

}
