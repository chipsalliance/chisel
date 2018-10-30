// See LICENSE for license details.

package chisel3.util.experimental

import chisel3._
import chisel3.experimental.annotate
// import chisel3.InstanceId
import chisel3.experimental.{ChiselAnnotation, RunFirrtlTransform}
import firrtl.annotations.{MemoryLoadFileType, _}
import firrtl.ir.{Module => _, _}
import firrtl.transforms.BlackBoxInlineAnno
import firrtl.Mappers._
import firrtl.{AnnotationSeq, CircuitForm, CircuitState, EmitCircuitAnnotation, LowForm, Transform, VerilogEmitter}

import scala.collection.mutable

/** This is the annotation created when using [[loadMemoryFromFile]], it records the memory, the load file
  * and the format of the file.
  * @param target        memory to load
  * @param fileName      name of input file
  * @param hexOrBinary   use $readmemh or $readmemb, i.e. hex or binary text input, default is hex
  */
case class ChiselLoadMemoryAnnotation[T <: Data](
  target:      MemBase[T],
  fileName:    String,
  hexOrBinary: MemoryLoadFileType.FileType = MemoryLoadFileType.Hex
)
  extends ChiselAnnotation with RunFirrtlTransform {

  if(fileName.isEmpty) {
    throw new Exception(
      s"""LoadMemory from file annotations file empty file name"""
    )
  }

  def transformClass: Class[LoadMemoryTransform] = classOf[LoadMemoryTransform]

  def toFirrtl: LoadMemoryAnnotation = {
    val tx = target.toNamed.asInstanceOf[ComponentName]
    LoadMemoryAnnotation(tx, fileName, hexOrBinary, Some(tx.name))
  }
}


object loadMemoryFromFile {
  /** Use this annotation generator to load a memory from a text file by using verilator and
    *  verilog's $readmemh or $readmemb.
    *  The treadle backend can also recognize this annotation and load memory at run-time.
    *
    * This annotation triggers the [[LoadMemoryTransform]] which will take add the verilog directive to
      * the relevant module by using the creating separate modules that are bound to the modules containing
    * the memories to be loaded.
    *
    * ==Example module==
    *
    * Consider a simple Module containing a memory
      * {{{
        * import chisel3._
        * class UsesMem(memoryDepth: Int, memoryType: Data) extends Module {
          *   val io = IO(new Bundle {
            *     val address = Input(UInt(memoryType.getWidth.W))
            *     val value   = Output(memoryType)
            *   })
          *   val memory = Mem(memoryDepth, memoryType)
          *   io.value := memory(io.address)
          * }
        * }}}
    *
    * ==Above module with annotation==
      *
    * To load this memory from a file /workspace/workdir/mem1.hex.txt
    * Just add an import and annotate the memory
      * {{{
        * import chisel3._
        * import chisel3.util.experimental.loadMemoryFromFile   // <<-- new import here
        * class UsesMem(memoryDepth: Int, memoryType: Data) extends Module {
          *   val io = IO(new Bundle {
            *     val address = Input(UInt(memoryType.getWidth.W))
            *     val value   = Output(memoryType)
            *   })
          *   val memory = Mem(memoryDepth, memoryType)
          *   io.value := memory(io.address)
          *   loadMemoryFromFile(memory, "/workspace/workdir/mem1.hex.txt")  // <<-- Note the annotation here
          * }
        * }}}
    *
    * ==Example file format==
      * A memory file should consist of ascii text in either hex or binary format
    * Example (a file containing the decimal values 0, 7, 14, 21):
      * {{{
        *   0
        *   7
        *   d
        *  15
        * }}}
    * Binary file is similarly constructed.
    *
    * ==More info==
    * See the LoadMemoryFromFileSpec.scala in the test suite for more examples
    * @see <a href="https://github.com/freechipsproject/chisel3/wiki/Chisel-Memories">Load Memories in the wiki</a>
    */
  def apply[T <: Data](
    memory: MemBase[T],
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
//scalastyle:off method.length
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
    val groups = annotations
      .collect{ case m: LoadMemoryAnnotation => m }
      .groupBy(_.target.serialize)
    val memoryAnnotations = groups.map { case (key, annos) =>
        if (annos.size > 1) {
          throw new Exception(
            s"Multiple (${annos.size} found for memory $key one LoadMemoryAnnotation is allowed per memory"
          )
        }
        key -> annos.head
      }

    val modulesByName = circuit.modules.collect { case module: firrtl.ir.Module =>  module.name -> module }.toMap

    /**
      * walk the module and for memories that have LoadMemory annotations
      * generate the bindable modules for verilog emission
      *
      * @param myModule     module being searched for memories
      */
    def processModule(myModule: DefModule): DefModule = {

      def makePath(componentName: String): String = {
        circuit.main + "." + myModule.name + "." + componentName
      }

      def processMemory(name: String): Unit = {
        val fullMemoryName = makePath(s"$name")

        memoryAnnotations.get(fullMemoryName) match {
          case Some(lma @ LoadMemoryAnnotation(ComponentName(componentName, moduleName), _, hexOrBinary, _)) =>
            val writer = new java.io.StringWriter

            modulesByName.get(moduleName.name).foreach { module =>
                val renderer = verilogEmitter.getRenderer(module, modulesByName)(writer)
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

      def processStatements(statement: Statement): Statement = {
        statement match {
          case m: DefMemory          => processMemory(m.name)
          case s                     => s map processStatements
        }
        statement
      }

      myModule match {
        case module: firrtl.ir.Module =>
          processStatements(module.body)
        case _ =>
      }

      myModule
    }

    circuit map processModule
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
