
package firrtl

import com.typesafe.scalalogging.LazyLogging
import java.io.Writer

import Utils._
import firrtl.passes._

trait Compiler extends LazyLogging {
  def run(c: Circuit, w: Writer)
}

object FIRRTLCompiler extends Compiler {
  def run(c: Circuit, w: Writer) = {
    FIRRTLEmitter.run(c, w)
    w.close
  }
}

object VerilogCompiler extends Compiler {
  // Copied from Stanza implementation
  val passes = Seq(
    CheckHighForm,          
    Resolve,
    ToWorkingIR,            
    ResolveKinds,
    InferTypes,
    ResolveGenders,
    PullMuxes,
    ExpandConnects,
    RemoveAccesses,
    ExpandWhens,
    CheckInitialization,   
    ConstProp,
    Resolve,
    LowerTypes
  )
  def run(c: Circuit, w: Writer)
  {
    val loweredIR = PassUtils.executePasses(c, passes)
    VerilogEmitter.run(loweredIR, w)
    w.close
  }

}
