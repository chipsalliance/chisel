// See LICENSE for license details.

package chisel3.util

import chisel3.MemBase
import chisel3.core.{RunFirrtlTransform, annotate, ChiselAnnotation}
import chisel3.internal.{Builder, InstanceId}
import firrtl.annotations._
import firrtl.ir.{Module => _, _}
import firrtl.passes.Pass
import firrtl.passes.memlib.DefAnnotatedMemory
import firrtl.transforms.BlackBoxInlineAnno
import firrtl.{CircuitForm, CircuitState, LowForm, Transform, VerilogEmitter, WDefInstance}

import scala.collection.mutable
import scala.util.matching.Regex

/**
  * chisel implementation for load memory
  * @param target        memory to load
  * @param fileName      name of input file
  * @param hexOrBinary   use $readmemh or $readmemb
  */
case class ChiselLoadMemoryAnnotation(
  target     : InstanceId,
  fileName   : String,
  hexOrBinary: MemoryLoadFileType.FileType = MemoryLoadFileType.Hex
)
  extends ChiselAnnotation with RunFirrtlTransform {

  if(fileName.isEmpty) {
    Builder.warning(
      s"""LoadMemory from file annotations file empty file name"""
    )
  }
  val SuffixRegex: Regex = """.+\.[^\.]*""".r
  fileName match {
    case SuffixRegex(_*) =>
      Builder.warning(
        s"""LoadMemoryAnnotation fileName "$fileName" has extension ".txt" will still be appended"""
      )
    case _ =>
  }

  def transformClass: Class[LoadMemoryTransform] = classOf[LoadMemoryTransform]

  def toFirrtl: LoadMemoryAnnotation = {
    LoadMemoryAnnotation(target.toNamed.asInstanceOf[ComponentName], fileName, hexOrBinary)
  }
}

/**
  * Enumeration of the two types of readmem statements available in verilog
  */
object MemoryLoadFileType extends Enumeration {
  type FileType = Value

  val Hex:    Value = Value("h")
  val Binary: Value = Value("b")
}

object loadMemoryFromFile {
  def apply(
    memory: MemBase[_],
    fileName: String,
    hexOrBinary: MemoryLoadFileType.FileType = MemoryLoadFileType.Hex
  ): Unit = {
    annotate(ChiselLoadMemoryAnnotation(memory, fileName))
  }
}

/**
  * Firrtl implementation for load memory
  * @param target        memory to load
  * @param fileName      name of input file
  * @param hexOrBinary   use $readmemh or $readmemb
  */
case class LoadMemoryAnnotation(
  target: ComponentName,
  fileName: String,
  hexOrBinary: MemoryLoadFileType.FileType = MemoryLoadFileType.Hex,
  originalMemoryNameOpt: Option[String] = None
) extends SingleTargetAnnotation[Named] {

  def getFileName: String = {
    originalMemoryNameOpt match {
      case Some(originalMemoryName) =>
        if(target.name == originalMemoryName) {
          fileName
        }
        else {
          fileName + target.name.drop(originalMemoryName.length)
        }
      case _ =>
        fileName
    }
  }

  def duplicate(newNamed: Named): LoadMemoryAnnotation = {
    newNamed match {
      case componentName: ComponentName =>
        this.copy(target = componentName, originalMemoryNameOpt = Some(target.name))
      case _ =>
        throw new Exception(s"Cannot annotate anything but a memory, invalid target ${newNamed.serialize}")
    }
  }
}

/**
  * This pass creates BlackBoxInlineAnno from the LoadMemoryAnnotations
  * it does this even if the backend is not verilog.
  *
  * @param circuitState the target circuit state
  */
//TODO: (chick) support a treadle or interpreter means of memory loading
//TODO: (chick) can this only be done when backend is known to support this.
//TODO: (chick) better integration with chisel end firrtl error systems
//scalastyle:off method.length cyclomatic.complexity regex
class CreateBindableMemoryLoaders(circuitState: CircuitState) extends Pass {

  val annotations      : Seq[Annotation] = circuitState.annotations
  val memoryAnnotations: Seq[LoadMemoryAnnotation] = annotations.collect{ case m: LoadMemoryAnnotation => m }

  val bindModules      : mutable.ArrayBuffer[BlackBoxInlineAnno] = new mutable.ArrayBuffer()

  val verilogEmitter   : VerilogEmitter = new VerilogEmitter

  /**
    * finds the specified module name in the circuit
    *
    * @param moduleName name to find
    * @param circuit circuit being analyzed
    * @return the circuit, exception occurs in not found
    */
  def findModule(moduleName: String, circuit: Circuit): DefModule = {
    circuit.modules.find(module => module.name == moduleName) match {
      case Some(module: firrtl.ir.Module) =>
        module
      case Some(externalModule: DefModule) =>
        externalModule
      case _ =>
        throw new Exception(s"Could not find module $moduleName in circuit $circuit")
    }
  }

  /**
    * walk the module and for memories that have LoadMemory annotations
    * generate the bindable modules for verilog emission
    *
    * @param modulePrefix kind of a path to the current module
    * @param myModule     module being searched for memories
    */
  def processModule(modulePrefix: String, myModule: DefModule): Unit = {

    def makePath(componentName: String): String = {
      circuitState.circuit.main + "." + myModule.name + "." + componentName
    }

    def processMemory(name: String): Unit = {
      val fullMemoryName = makePath(s"$name")

      memoryAnnotations.find {
        ma: LoadMemoryAnnotation =>
          val targetName = ma.target.serialize
          targetName == fullMemoryName
      } match {
        case Some(lma @ LoadMemoryAnnotation(ComponentName(componentName, moduleName), _, hexOrBinary, _)) =>
          val writer = new java.io.StringWriter
          circuitState.circuit.modules
            .filter { module => module.name == moduleName.name }
            .collectFirst { case m: firrtl.ir.Module => m }
            .foreach { module =>

              val moduleMap = circuitState.circuit.modules.map(m => m.name -> m).toMap
              val renderer = verilogEmitter.getRenderer(module, moduleMap)(writer)
              val loadFileName = lma.getFileName + ".txt"
              renderer.emitVerilogBind(s"BindsTo_${moduleName.name}",
                s"""
                   |initial begin
                   |  $$readmem$hexOrBinary("$loadFileName", ${myModule.name}.$componentName);
                   |end
                    """.stripMargin)
              val inLineText = writer.toString + "\n" +
                s"""bind ${myModule.name} BindsTo_${myModule.name} BindsTo_${myModule.name}_Inst(.*);"""

              val blackBoxInline = BlackBoxInlineAnno(
                moduleName,
                moduleName.serialize + "." + componentName + ".v",
                inLineText
              )

              bindModules += blackBoxInline
            }

        case _ =>
      }
    }

    def processStatements(statement: Statement): Unit = {
      statement match {
        case block: Block =>
          block.stmts.foreach { subStatement =>
            processStatements(subStatement)
          }

        case m: DefAnnotatedMemory => processMemory(m.name)

        case m: DefMemory          => processMemory(m.name)

        case WDefInstance(_, _, moduleName, _) =>
          val subModule = findModule(moduleName, circuitState.circuit)
          val newPrefix = (if (modulePrefix.nonEmpty) modulePrefix + "." else "") + myModule.name

          processModule(newPrefix, subModule)

        case _ =>
      }
    }

    myModule match {
      case module: firrtl.ir.Module =>
        processStatements(module.body)
      case _ =>
    }
  }

  /**
    * run the pass
    * @param c the circuit
    * @return
    */
  def run(c: Circuit): Circuit = {
    val topModule = findModule(c.main, c)
    processModule(modulePrefix = c.main, topModule)
    c
  }
}

//noinspection ScalaStyle
class LoadMemoryTransform extends Transform {
  def inputForm  : CircuitForm = LowForm
  def outputForm : CircuitForm = LowForm

  def execute(state: CircuitState): CircuitState = {
    val bindLoaderTransform = new CreateBindableMemoryLoaders(state)
    bindLoaderTransform.run(state.circuit)
    state.copy(annotations = state.annotations ++ bindLoaderTransform.bindModules)
  }
}
