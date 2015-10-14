// Private implicit classes and other utility functions for debugging

package firrtl

import java.io.PrintWriter
import Utils._

private object DebugUtils {


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
    def printDebug(circuit: Circuit){
      if (circuitEnable) this.debug(circuit.serialize(debugFlags))
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
