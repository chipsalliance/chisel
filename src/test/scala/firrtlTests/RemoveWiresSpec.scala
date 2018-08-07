// See LICENSE for license details.

package firrtlTests

import firrtl._
import firrtl.ir._
import firrtl.Mappers._
import FirrtlCheckers._

import collection.mutable

class RemoveWiresSpec extends FirrtlFlatSpec {
  def compile(input: String): CircuitState =
    (new LowFirrtlCompiler).compileAndEmit(CircuitState(parse(input), ChirrtlForm), List.empty)
  def compileBody(body: String) = {
    val str = """
      |circuit Test :
      |  module Test :
      |""".stripMargin + body.split("\n").mkString("    ", "\n    ", "")
    compile(str)
  }

  def getNodesAndWires(circuit: Circuit): (Seq[DefNode], Seq[DefWire]) = {
    require(circuit.modules.size == 1)

    val nodes = mutable.ArrayBuffer.empty[DefNode]
    val wires = mutable.ArrayBuffer.empty[DefWire]
    def onStmt(stmt: Statement): Statement = {
      stmt map onStmt match {
        case node: DefNode => nodes += node
        case wire: DefWire => wires += wire
        case _ =>
      }
      stmt
    }

    circuit.modules.head match {
      case Module(_,_,_, body) => onStmt(body)
    }
    (nodes, wires)
  }

  def orderedNames(circuit: Circuit): Seq[String] = {
    require(circuit.modules.size == 1)
    val names = mutable.ArrayBuffer.empty[String]
    def onStmt(stmt: Statement): Statement = {
      stmt map onStmt match {
        case reg: DefRegister => names += reg.name
        case wire: DefWire => names += wire.name
        case node: DefNode => names += node.name
        case _ =>
      }
      stmt
    }
    circuit.modules.head match {
      case Module(_,_,_, body) => onStmt(body)
    }
    names
  }

  "Remove Wires" should "turn wires and their single connect into nodes" in {
    val result = compileBody(s"""
      |input a : UInt<8>
      |output b : UInt<8>
      |wire w : UInt<8>
      |w <= a
      |b <= w""".stripMargin
    )
    val (nodes, wires) = getNodesAndWires(result.circuit)
    wires.size should be (0)

    nodes.map(_.serialize) should be (Seq("node w = a"))
  }

  it should "order nodes in a legal, flow-forward way" in {
    val result = compileBody(s"""
      |input a : UInt<8>
      |output b : UInt<8>
      |wire w : UInt<8>
      |wire x : UInt<8>
      |node y = x
      |x <= w
      |w <= a
      |b <= y""".stripMargin
    )
    val (nodes, wires) = getNodesAndWires(result.circuit)
    wires.size should be (0)
    nodes.map(_.serialize) should be (
      Seq("node w = a",
          "node x = w",
          "node y = x")
    )
  }

  it should "properly pad rhs of introduced nodes if necessary" in {
    val result = compileBody(s"""
      |output b : UInt<8>
      |wire w : UInt<8>
      |w <= UInt(2)
      |b <= w""".stripMargin
    )
    val (nodes, wires) = getNodesAndWires(result.circuit)
    wires.size should be (0)
    nodes.map(_.serialize) should be (
      Seq("""node w = pad(UInt<2>("h2"), 8)""")
    )
  }

  it should "support arbitrary expression for wire connection rhs" in {
    val result = compileBody(s"""
      |input a : UInt<8>
      |input b : UInt<8>
      |output c : UInt<8>
      |wire w : UInt<8>
      |w <= tail(add(a, b), 1)
      |c <= w""".stripMargin
    )
    val (nodes, wires) = getNodesAndWires(result.circuit)
    wires.size should be (0)
    nodes.map(_.serialize) should be (
      Seq("""node w = tail(add(a, b), 1)""")
    )
  }

  it should "do a reasonable job preserving input order for unrelatd logic" in {
    val result = compileBody(s"""
      |input a : UInt<8>
      |input b : UInt<8>
      |output z : UInt<8>
      |node x = not(a)
      |node y = not(b)
      |z <= and(x, y)""".stripMargin
    )
    val (nodes, wires) = getNodesAndWires(result.circuit)
    wires.size should be (0)
    nodes.map(_.serialize) should be (
      Seq("node x = not(a)",
          "node y = not(b)")
    )
  }

  it should "work for multiple clocks" in {
    val result = compileBody(
      s"""|input clock: Clock
          |reg a : UInt<1>, clock
          |node clock2 = asClock(a)
          |reg b : UInt<1>, clock2
          |""".stripMargin
    )
    val names = orderedNames(result.circuit)
    names should be (Seq("a", "clock2", "b"))
  }

  it should "order registers correctly" in {
    val result = compileBody(s"""
      |input clock : Clock
      |input a : UInt<8>
      |output c : UInt<8>
      |wire w : UInt<8>
      |node n = tail(add(w, UInt(1)), 1)
      |reg r : UInt<8>, clock
      |w <= tail(add(r, a), 1)
      |c <= n""".stripMargin
    )
    // Check declaration before use is maintained
    passes.CheckHighForm.execute(result)
  }
}
