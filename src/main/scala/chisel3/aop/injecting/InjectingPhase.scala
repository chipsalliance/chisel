// SPDX-License-Identifier: Apache-2.0

package chisel3.aop.injecting

import chisel3.aop.Aspect
import firrtl.options.Phase
import firrtl.stage.FirrtlCircuitAnnotation
import firrtl.{ir, AnnotationSeq}

import scala.collection.mutable

/** Phase that consumes all Aspects and calls their toAnnotationSeq methods.
  *
  * Consumes the [[chisel3.stage.DesignAnnotation]] and converts every [[Aspect]] into their annotations prior to executing FIRRTL
  */
class InjectingPhase extends Phase {
  def transform(annotations: AnnotationSeq): AnnotationSeq = {
    val addStmtMap = mutable.HashMap[String, Seq[ir.Statement]]()
    val addModules = mutable.ArrayBuffer[ir.DefModule]()
    val newAnnotations = annotations.flatMap {
      case InjectStatement(mt, s, addedModules, annotations) =>
        addModules ++= addedModules
        addStmtMap(mt.module) = addStmtMap.getOrElse(mt.module, Nil) :+ s
        annotations
      case _ => Seq.empty
    }
    logger.debug(s"new annotation added: \n${newAnnotations.map(_.serialize).mkString("=n")}")
    // Append all statements to end of corresponding modules
    annotations.filter {
      case _: InjectStatement => false
      case _ => true
    }.map {
      case f @ FirrtlCircuitAnnotation(c) =>
        val newModules = c.modules.map { m: ir.DefModule =>
          m match {
            case m: ir.Module if addStmtMap.contains(m.name) =>
              logger.debug(s"Injecting to ${m.name} with statement: \n${ir.Block(addStmtMap(m.name)).serialize}")
              m.copy(body = ir.Block(m.body +: addStmtMap(m.name)))
            case m: _root_.firrtl.ir.ExtModule if addStmtMap.contains(m.name) =>
              logger.debug(s"Injecting to ${m.name} with statement: \n${ir.Block(addStmtMap(m.name)).serialize}")
              ir.Module(m.info, m.name, m.ports, ir.Block(addStmtMap(m.name)))
            case other: ir.DefModule => other
          }
        }
        f.copy(c.copy(modules = newModules ++ addModules))
      case a => a
    } ++ newAnnotations
  }
}
