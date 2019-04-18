package chisel3.aop.injecting

import chisel3.aop.ConcernTransform
import firrtl.{ChirrtlForm, CircuitForm, CircuitState, ir}

import scala.collection.mutable

class InjectingTransform extends ConcernTransform {
  override def inputForm: CircuitForm = ChirrtlForm
  override def outputForm: CircuitForm = ChirrtlForm

  override def execute(state: CircuitState): CircuitState = {

    val addStmtMap = mutable.HashMap[String, Seq[ir.Statement]]()
    val addModules = mutable.ArrayBuffer[ir.DefModule]()
    val newAnnotations = state.annotations.flatMap {
      case InjectStatement(mt, s, addedModules, annotations) =>
        addModules ++= addedModules
        addStmtMap(mt.module) = s +: addStmtMap.getOrElse(mt.module, Nil)
        annotations
      case other => Seq(other)
    }
    //addStmtMap.foreach(println)

    val newModules = state.circuit.modules.map { m: ir.DefModule =>
      m match {
        case m: ir.Module if addStmtMap.contains(m.name) =>
          val newM = m.copy(body = ir.Block(m.body +: addStmtMap(m.name)))
          //println(newM.serialize)
          newM
        case m: _root_.firrtl.ir.ExtModule if addStmtMap.contains(m.name) =>
          ir.Module(m.info, m.name, m.ports, ir.Block(addStmtMap(m.name)))
        case other: ir.DefModule => other
      }
    }

    val newCircuit = state.circuit.copy(modules = newModules ++ addModules)

    println("Injecting Transform")
    println("Starting Annotations:")
    state.annotations.foreach(println)
    println("Ending Annotations:")
    newAnnotations.foreach(println)


    state.copy(annotations = newAnnotations, circuit = newCircuit)
  }
}
