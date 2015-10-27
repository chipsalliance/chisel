// Private implicit classes and other utility functions for debugging

package firrtl

import java.io.PrintWriter
import Utils._

private object DebugUtils {

  implicit class DebugASTUtils(ast: AST) {
    // Is this actually any use?
    def preOrderTraversal(f: AST => Unit): Unit = {
      f(ast)
      ast match {
        case a: Block => a.stmts.foreach(_.preOrderTraversal(f))
        case a: Assert => a.pred.preOrderTraversal(f)
        case a: When => {
          a.pred.preOrderTraversal(f)
          a.conseq.preOrderTraversal(f)
          a.alt.preOrderTraversal(f)
        }
        case a: BulkConnect => {
          a.lhs.preOrderTraversal(f)
          a.rhs.preOrderTraversal(f)
        }
        case a: Connect => {
          a.lhs.preOrderTraversal(f)
          a.rhs.preOrderTraversal(f)
        }
        case a: OnReset => {
          a.lhs.preOrderTraversal(f)
          a.rhs.preOrderTraversal(f)
        }
        case a: DefAccessor => {
          a.dir.preOrderTraversal(f)
          a.source.preOrderTraversal(f)
          a.index.preOrderTraversal(f)
        }
        case a: DefPoison => a.tpe.preOrderTraversal(f)
        case a: DefNode => a.value.preOrderTraversal(f)
        case a: DefInst => a.module.preOrderTraversal(f)
        case a: DefMemory => {
          a.tpe.preOrderTraversal(f)
          a.clock.preOrderTraversal(f)
        }
        case a: DefReg => {
          a.tpe.preOrderTraversal(f)
          a.clock.preOrderTraversal(f)
          a.reset.preOrderTraversal(f)
        }
        case a: DefWire => a.tpe.preOrderTraversal(f)
        case a: Field => {
          a.dir.preOrderTraversal(f)
          a.tpe.preOrderTraversal(f)
        }
        case a: VectorType => a.tpe.preOrderTraversal(f)
        case a: BundleType => a.fields.foreach(_.preOrderTraversal(f))
        case a: Port => {
          a.dir.preOrderTraversal(f)
          a.tpe.preOrderTraversal(f)
        }
        case a: Module => {
          a.ports.foreach(_.preOrderTraversal(f))
          a.stmt.preOrderTraversal(f)
        }
        case a: Circuit => a.modules.foreach(_.preOrderTraversal(f)) 
        //case _ => throw new Exception(s"Unsupported FIRRTL node ${ast.getClass.getSimpleName}!")
        case _ => 
      }
    }
  }


  /** Private class for recording and organizing debug information  */
  class Logger private (
    writer: PrintWriter,
    printMode: Symbol,
    printVars: List[Symbol]){
   
    // Legal printModes: 'none, 'error, 'warn, 'info, 'debug, 'trace
    require(List('none, 'error, 'warn, 'info, 'debug, 'trace) contains printMode) 
    val errorEnable = List('error, 'warn, 'info, 'debug, 'trace) contains printMode
    val warnEnable  = List('warn, 'info, 'debug, 'trace) contains printMode
    val infoEnable  = List('info, 'debug, 'trace) contains printMode
    val debugEnable = List('debug, 'trace) contains printMode
    val traceEnable = List('trace) contains printMode
    val circuitEnable = printVars contains 'circuit
    val debugFlags = printVars.map(_ -> true).toMap.withDefaultValue(false)

    def println(message: => String){
      writer.println(message)
    }
    def error(message: => String){
      if (errorEnable) writer.println(message.split("\n").map("[error] " + _).mkString("\n"))
    }
    def warn(message: => String){
      if (warnEnable) writer.println(message.split("\n").map("[warn] " + _).mkString("\n"))
    }
    def info(message: => String){
      if (infoEnable) writer.println(message.split("\n").map("[info] " + _).mkString("\n"))
    }
    def debug(message: => String){
      if (debugEnable) writer.println(message.split("\n").map("[debug] " + _).mkString("\n"))
    }
    def trace(message: => String){
      if (traceEnable) writer.println(message.split("\n").map("[trace] " + _).mkString("\n"))
    }
    def printlnDebug(circuit: Circuit){
      if (circuitEnable) this.println(circuit.serialize(debugFlags))
    }
    // Used if not autoflushing
    def flush() = writer.flush()
    
  }
  /** Factory object for logger
    *
    * Logger records and organizes debug information
    */
  object Logger
  {
    def apply(writer: PrintWriter): Logger = 
      new Logger(writer, 'warn, List())
    def apply(writer: PrintWriter, printMode: Symbol): Logger = 
      new Logger(writer, printMode, List())
    def apply(writer: PrintWriter, printMode: Symbol, printVars: List[Symbol]): Logger = 
      new Logger(writer, printMode, printVars)
    def apply(): Logger = new Logger(null, 'none, List())
  }
}
