// See LICENSE for license details.

package chisel3.util

import chisel3.MemBase
import chisel3.core.{ChiselAnnotation, RunFirrtlTransform, annotate}
import chisel3.internal.{Builder, InstanceId}
import firrtl.annotations.{MemoryLoadFileType, _}
import firrtl.ir.{Module => _, _}
import firrtl.transforms.BlackBoxInlineAnno
import firrtl.{AnnotationSeq, CircuitForm, CircuitState, EmitCircuitAnnotation, LowForm, Transform, VerilogEmitter}

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

  private var memoryCounter: Int = -1

  private val bindModules: mutable.ArrayBuffer[BlackBoxInlineAnno] = new mutable.ArrayBuffer()

  private val verilogEmitter:    VerilogEmitter = new VerilogEmitter

  /**
    * run the pass
    * @param circuit the circuit
    * @param annotations all the annotations
    * @return
    */
  def run(circuit: Circuit, annotations: AnnotationSeq): Circuit = {

    val memoryAnnotations = {
      annotations.collect{ case m: LoadMemoryAnnotation => m }.map { ma => ma.target.serialize -> ma }.toMap
    }

    val modulesByName = {
      circuit.modules.collect { case m: firrtl.ir.Module => m }.map { module => module.name -> module }.toMap
    }

    /**
      * walk the module and for memories that have LoadMemory annotations
      * generate the bindable modules for verilog emission
      *
      * @param myModule     module being searched for memories
      */
    def processModule(myModule: DefModule): Unit = {

      def makePath(componentName: String): String = {
        circuit.main + "." + myModule.name + "." + componentName
      }

      def processMemory(name: String): Unit = {
        val fullMemoryName = makePath(s"$name")

        memoryAnnotations.get(fullMemoryName) match {
          case Some(lma @ LoadMemoryAnnotation(ComponentName(componentName, moduleName), _, hexOrBinary, _)) =>
            val writer = new java.io.StringWriter

            modulesByName.get(moduleName.name).foreach { module =>
                val moduleMap = circuit.modules.map(m => m.name -> m).toMap
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

    circuit.modules.foreach(processModule)
    circuit
  }

  def execute(state: CircuitState): CircuitState = {
    val isVerilog = state.annotations.exists {
      case EmitCircuitAnnotation(emitter) =>
        emitter == classOf[VerilogEmitter]
      case _ =>
        false
    }
    if(isVerilog) {
      run(state.circuit, state.annotations)
      state.copy(annotations = state.annotations ++ bindModules)
    }
    else {
      state
    }
  }
}
