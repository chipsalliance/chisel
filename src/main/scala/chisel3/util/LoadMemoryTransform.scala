// See LICENSE for license details.

package chisel3.util

import chisel3.core.RunFirrtlTransform
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
  */case class ChiselLoadMemoryAnnotation(target: InstanceId, fileName: String, hexOrBinary: String = "h")
  extends chisel3.core.ChiselAnnotation
    with RunFirrtlTransform {

  if(! Seq("h","b").contains(hexOrBinary)) {
    Builder.error(
      s"""LoadMemory from file $fileName format must be "h" or "b" not "$hexOrBinary" """
    )
  }

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

  def transformClass : Class[LoadMemoryTransform] = classOf[LoadMemoryTransform]

  def toFirrtl: LoadMemoryAnnotation = LoadMemoryAnnotation(target.toNamed, fileName, hexOrBinary)
}

/**
  * Firrtl implementation for load memory
  * @param target        memory to load
  * @param fileName      name of input file
  * @param hexOrBinary   use $readmemh or $readmemb
  */
case class LoadMemoryAnnotation(
  target: Named,
  fileName: String,
  hexOrBinary: String = "h",
  originalMemoryNameOpt: Option[String] = None
) extends SingleTargetAnnotation[Named] {

  def getFileName: String = {
    (target, originalMemoryNameOpt) match {
      case (ComponentName(componentName, _), Some(originalMemoryName)) =>
        if(componentName == originalMemoryName) {
          fileName
        }
        else {
          fileName + componentName.drop(originalMemoryName.length)
        }
      case _ =>
        fileName
    }
  }

  def duplicate(n: Named): LoadMemoryAnnotation = {
//    if(n.serialize != target.serialize) {
////      Builder.errors.error(s"LoadMemory[${target.serialize}] must be a simple memory, no bundles")
//      throw new Exception(s"LoadMemory[${target.serialize}] must be a simple memory, no bundles")
//    }
    target match {
      case ComponentName(componentName, _) =>
        this.copy(target = n, originalMemoryNameOpt = Some(componentName))
      case _ =>
        this.copy(target = n)
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
        case Some(lma @ LoadMemoryAnnotation(
        ComponentName(componentName, moduleName: ModuleName), _, hexOrBinary, _
        )) =>

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
