// See LICENSE for license details.

package chisel3.util

import chisel3._
import chisel3.core.RunFirrtlTransform
import chisel3.internal.InstanceId
import firrtl.{CircuitForm, CircuitState, LowForm, TargetDirAnnotation, Transform, WRef, WSubField}
import firrtl.annotations.{Annotation, Named, SingleTargetAnnotation}
import firrtl.ir.{Circuit, DefMemory, DefModule, Statement}
import firrtl.passes.MemPortUtils.MemPortMap
import firrtl.passes.Pass
import firrtl.Mappers._
import firrtl.passes.memlib.DefAnnotatedMemory
import firrtl.passes.memlib.MemTransformUtils.updateStmtRefs

case class ChiselLoadMemoryAnnotation(target: InstanceId, fileName: String)
  extends chisel3.core.ChiselAnnotation
    with RunFirrtlTransform {

  def transformClass : Class[LoadMemoryTransform] = classOf[LoadMemoryTransform]

  def toFirrtl: LoadMemoryAnnotation = LoadMemoryAnnotation(target.toNamed, fileName)
}

case class LoadMemoryAnnotation(target: Named, value1: String) extends SingleTargetAnnotation[Named] {
  def duplicate(n: Named): LoadMemoryAnnotation = this.copy(target = n)
}


class CreateBindableMemoryLoaders(val annotations: Seq[Annotation]) extends Pass {

  def createMemProto(m: DefAnnotatedMemory): DefAnnotatedMemory = {
    val rports = m.readers.indices map (i => s"R$i")
    val wports = m.writers.indices map (i => s"W$i")
    val rwports = m.readwriters.indices map (i => s"RW$i")
    m copy (readers = rports, writers = wports, readwriters = rwports)
  }

  /** Maps the serialized form of all memory port field names to the
    *    corresponding new memory port field Expression.
    *  E.g.:
    *    - ("m.read.addr") becomes (m.R0.addr)
    */
  def getMemPortMap(m: DefAnnotatedMemory, memPortMap: MemPortMap) {
    val defaultFields = Seq("addr", "en", "clk")
    val rFields = defaultFields :+ "data"
    val wFields = rFields :+ "mask"
    val rwFields = defaultFields ++ Seq("wmode", "wdata", "rdata", "wmask")

    def updateMemPortMap(ports: Seq[String], fields: Seq[String], newPortKind: String): Unit =
      for ((p, i) <- ports.zipWithIndex; f <- fields) {
        val newPort = WSubField(WRef(m.name), newPortKind + i)
        val field = WSubField(newPort, f)
        memPortMap(s"${m.name}.$p.$f") = field
      }
    updateMemPortMap(m.readers, rFields, "R")
    updateMemPortMap(m.writers, wFields, "W")
    updateMemPortMap(m.readwriters, rwFields, "RW")
  }

  /**
    * Search modules for memories that match annotations
    * @param currentModule          current being checked
    * @param namePrefix             progressive path to module
    * @return
    */
  def updateMemMods(currentModule: DefModule, namePrefix: String = ""): DefModule = {

    def updateMemStmts(memPortMap: MemPortMap)(s: Statement): Statement = s match {
      case m: DefAnnotatedMemory =>
        val updatedMem = createMemProto(m)
        getMemPortMap(m, memPortMap)
        updatedMem
      case m: DefMemory =>
        println(m)
        m
      case s => s map updateMemStmts(memPortMap)
    }

    val memPortMap = new MemPortMap
    (currentModule map updateMemStmts(memPortMap)
      map updateStmtRefs(memPortMap))
  }

  /**
    * run the pass
    * @param c
    * @return
    */
  def run(c: Circuit): Circuit = {
    c.copy(modules = c.modules.map { module => updateMemMods(module) })
  }
}

//noinspection ScalaStyle
class LoadMemoryTransform extends Transform {
  def inputForm  : CircuitForm = LowForm
  def outputForm : CircuitForm = LowForm

  def execute(state: CircuitState): CircuitState = {
    val targetDir = state.annotations.collectFirst { case td: TargetDirAnnotation => td }
    println(s"target dir is ${targetDir.getOrElse("no dir")}")
    val processedAnnotations = state.annotations.flatMap {
      case loadMemoryAnnotation: LoadMemoryAnnotation => Some(loadMemoryAnnotation)
      case other                                      => None
    } ++ (if(targetDir.isDefined) Seq(targetDir.get) else Seq())

    println(s"got here")
    (new CreateBindableMemoryLoaders(processedAnnotations)).run(state.circuit)
    state.copy(annotations = processedAnnotations)
  }
}
