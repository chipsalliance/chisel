// See LICENSE for license details.

package chisel3.util

import chisel3.MemBase
import chisel3.core.{ChiselAnnotation, RunFirrtlTransform, annotate}
import chisel3.internal.{Builder, InstanceId}
import firrtl.annotations.{MemoryLoadFileType, _}
import firrtl.ir.{Module => _, _}
import firrtl.passes.Pass
import firrtl.passes.memlib.DefAnnotatedMemory
import firrtl.transforms.BlackBoxInlineAnno
import firrtl.{CircuitForm, CircuitState, EmitCircuitAnnotation, LowForm, Transform, VerilogEmitter, WDefInstance}

import scala.collection.mutable

/**
  * chisel implementation for load memory
  * @param target        memory to load
  * @param fileName      name of input file
  * @param hexOrBinary   use $readmemh or $readmemb
  */
case class ChiselLoadMemoryAnnotation(
  target:      InstanceId,
  fileName:    String,
  hexOrBinary: MemoryLoadFileType.FileType = MemoryLoadFileType.Hex
)
  extends ChiselAnnotation with RunFirrtlTransform {

  if(fileName.isEmpty) {
    Builder.warning(
      s"""LoadMemory from file annotations file empty file name"""
    )
  }

  def transformClass: Class[LoadMemoryTransform] = classOf[LoadMemoryTransform]

  def toFirrtl: LoadMemoryAnnotation = {
    LoadMemoryAnnotation(target.toNamed.asInstanceOf[ComponentName], fileName, hexOrBinary)
  }
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
  * This pass creates BlackBoxInlineAnno from the LoadMemoryAnnotations
  * it does this even if the backend is not verilog.
  *
  * @param circuitState the target circuit state
  */
//TODO: (chick) better integration with chisel end firrtl error systems
//scalastyle:off method.length cyclomatic.complexity regex
class CreateBindableMemoryLoaders(circuitState: CircuitState) extends Pass {
  var memoryCounter: Int = -1

  val annotations:       Seq[Annotation] = circuitState.annotations
  val memoryAnnotations: Seq[LoadMemoryAnnotation] = annotations.collect{ case m: LoadMemoryAnnotation => m }

  val bindModules:       mutable.ArrayBuffer[BlackBoxInlineAnno] = new mutable.ArrayBuffer()

  val verilogEmitter:    VerilogEmitter = new VerilogEmitter

  /**
    * walk the module and for memories that have LoadMemory annotations
    * generate the bindable modules for verilog emission
    *
    * @param myModule     module being searched for memories
    */
  def processModule(myModule: DefModule): Unit = {

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
              val loadFileName = lma.getFileName

              memoryCounter += 1
              val bindsToName = s"BindsTo_${memoryCounter}_${moduleName.name}"
              renderer.emitVerilogBind(bindsToName,
                s"""
                   |initial begin
                   |  $$readmem$hexOrBinary("$loadFileName", ${myModule.name}.$componentName);
                   |end
                    """.stripMargin)
              val inLineText = writer.toString + "\n" +
                s"""bind ${myModule.name} $bindsToName ${bindsToName}_Inst(.*);"""

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
    c.modules.foreach(processModule)
    c
  }
}

/**
  * This transform only is activated if verilog is being generated
  * (determined by presence of the proper emit annotation)
  * when activated it creates additional verilog files that contain
  * modules bound to the modules that contain an initializable memory
  *
  * Currently the only non-verilog based simulation that can support loading
  * memory from a file is treadle but it does not need this transform
  * to do that.
  */
//noinspection ScalaStyle
class LoadMemoryTransform extends Transform {
  def inputForm: CircuitForm  = LowForm
  def outputForm: CircuitForm = LowForm

  def execute(state: CircuitState): CircuitState = {
    val isVerilog = state.annotations.exists {
      case EmitCircuitAnnotation(emitter) =>
        emitter == classOf[VerilogEmitter]
      case _ =>
        false
    }
    if(isVerilog) {
      val bindLoaderTransform = new CreateBindableMemoryLoaders(state)
      bindLoaderTransform.run(state.circuit)
      state.copy(annotations = state.annotations ++ bindLoaderTransform.bindModules)
    }
    else {
      state
    }
  }
}
