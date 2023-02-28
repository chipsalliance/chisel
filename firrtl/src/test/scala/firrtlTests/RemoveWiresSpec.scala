// SPDX-License-Identifier: Apache-2.0

package firrtlTests

import firrtl._
import firrtl.ir._
import firrtl.Mappers._
import firrtl.testutils._
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
      stmt.map(onStmt) match {
        case node: DefNode => nodes += node
        case wire: DefWire => wires += wire
        case _ =>
      }
      stmt
    }

    circuit.modules.head match {
      case Module(_, _, _, body) => onStmt(body)
    }
    (nodes.toSeq, wires.toSeq)
  }

  def orderedNames(circuit: Circuit): Seq[String] = {
    require(circuit.modules.size == 1)
    val names = mutable.ArrayBuffer.empty[String]
    def onStmt(stmt: Statement): Statement = {
      stmt.map(onStmt) match {
        case reg:  DefRegister => names += reg.name
        case wire: DefWire     => names += wire.name
        case node: DefNode     => names += node.name
        case _ =>
      }
      stmt
    }
    circuit.modules.head match {
      case Module(_, _, _, body) => onStmt(body)
    }
    names.toSeq
  }

  "Remove Wires" should "turn wires and their single connect into nodes" in {
    val result = compileBody(s"""
                                |input a : UInt<8>
                                |output b : UInt<8>
                                |wire w : UInt<8>
                                |w <= a
                                |b <= w""".stripMargin)
    val (nodes, wires) = getNodesAndWires(result.circuit)
    wires.size should be(0)

    nodes.map(_.serialize) should be(Seq("node w = a"))
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
                                |b <= y""".stripMargin)
    val (nodes, wires) = getNodesAndWires(result.circuit)
    wires.size should be(0)
    nodes.map(_.serialize) should be(
      Seq("node w = a", "node x = w", "node y = x")
    )
  }

  it should "properly pad rhs of introduced nodes if necessary" in {
    val result = compileBody(s"""
                                |output b : UInt<8>
                                |wire w : UInt<8>
                                |w <= UInt(2)
                                |b <= w""".stripMargin)
    val (nodes, wires) = getNodesAndWires(result.circuit)
    wires.size should be(0)
    nodes.map(_.serialize) should be(
      Seq("""node w = UInt<8>("h2")""")
    )
  }

  it should "support arbitrary expression for wire connection rhs" in {
    val result = compileBody(s"""
                                |input a : UInt<8>
                                |input b : UInt<8>
                                |output c : UInt<8>
                                |wire w : UInt<8>
                                |w <= tail(add(a, b), 1)
                                |c <= w""".stripMargin)
    val (nodes, wires) = getNodesAndWires(result.circuit)
    wires.size should be(0)
    nodes.map(_.serialize) should be(
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
                                |z <= and(x, y)""".stripMargin)
    val (nodes, wires) = getNodesAndWires(result.circuit)
    wires.size should be(0)
    nodes.map(_.serialize) should be(
      Seq("node x = not(a)", "node y = not(b)")
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
    names should be(Seq("a", "clock2", "b"))
  }

  it should "give nodes made from invalid wires the correct type" in {
    val result = compileBody(
      s"""|input  a   : SInt<4>
          |input  sel : UInt<1>
          |output z   : SInt<4>
          |wire w : SInt<4>
          |w is invalid
          |z <= mux(sel, a, w)
          |""".stripMargin
    )
    result should containLine("""node w = validif(UInt<1>("h0"), SInt<4>("h0"))""")
  }

}
