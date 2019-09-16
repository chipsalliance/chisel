// See LICENSE for license details.

package chisel3.aop.injecting

import firrtl.{ChirrtlForm, CircuitForm, CircuitState, Transform, ir}

import scala.collection.mutable

/** Appends statements contained in [[InjectStatement]] annotations to the end of their corresponding modules
  *
  * Implemented with Chisel Aspects and the [[chisel3.aop.injecting]] library
  */
class InjectingTransform extends Transform {
  override def inputForm: CircuitForm = ChirrtlForm
  override def outputForm: CircuitForm = ChirrtlForm

  override def execute(state: CircuitState): CircuitState = {

    val addStmtMap = mutable.HashMap[String, Seq[ir.Statement]]()
    val addModules = mutable.ArrayBuffer[ir.DefModule]()

    // Populate addStmtMap and addModules, return annotations in InjectStatements, and omit InjectStatement annotation
    val newAnnotations = state.annotations.flatMap {
      case InjectStatement(mt, s, addedModules, annotations) =>
        addModules ++= addedModules
        addStmtMap(mt.module) = s +: addStmtMap.getOrElse(mt.module, Nil)
        annotations
      case other => Seq(other)
    }

    // Append all statements to end of corresponding modules
    val newModules = state.circuit.modules.map { m: ir.DefModule =>
      m match {
        case m: ir.Module if addStmtMap.contains(m.name) =>
          m.copy(body = ir.Block(m.body +: addStmtMap(m.name)))
        case m: _root_.firrtl.ir.ExtModule if addStmtMap.contains(m.name) =>
          ir.Module(m.info, m.name, m.ports, ir.Block(addStmtMap(m.name)))
        case other: ir.DefModule => other
      }
    }

    // Return updated circuit and annotations
    val newCircuit = state.circuit.copy(modules = newModules ++ addModules)
    state.copy(annotations = newAnnotations, circuit = newCircuit)
  }
}
