package chisel3.aop

import firrtl._
import firrtl.ir._
import firrtl.Mappers._

import scala.collection.mutable

class AspectTransform extends Transform {
  override def inputForm: CircuitForm = ChirrtlForm
  override def outputForm: CircuitForm = ChirrtlForm

  override def execute(state: CircuitState): CircuitState = {
    val dut = state.annotations.collectFirst { case DesignAnnotation(dut) => dut }.get

    val concerns = state.annotations.collect { case a: Concern[_, _] => a }

    val addStmtMap = mutable.HashMap[String, Seq[Statement]]()

    val annotations = concerns.flatMap { concern =>
      concern.resolveAspects(dut).flatMap {
        case AddStatements(module, s) =>
          addStmtMap(module) = s +: addStmtMap.getOrElse(module, Nil)
          Nil
        case other => Seq(other)
      }
    }

    val newCircuit = state.circuit.map { m: DefModule =>
      m match {
        case m: Module if addStmtMap.contains(m.name) =>
          val newM = m.copy(body = Block(m.body +: addStmtMap(m.name)))
          println(newM.serialize)
          newM
        case m: ExtModule if addStmtMap.contains(m.name) => Module(m.info, m.name, m.ports, Block(addStmtMap(m.name)))
        case other: DefModule => other
      }
    }

    state.copy(annotations = state.annotations ++ annotations, circuit = newCircuit)
  }
}
