
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
    ToWorkingIR,            
    ResolveKinds,
    InferTypes,
    CheckTypes,
    ResolveGenders,
    CheckGenders,
    InferWidths,
    CheckWidths,
    PullMuxes,
    ExpandConnects,
    RemoveAccesses,
    ExpandWhens,
    CheckInitialization,   
    ConstProp,
    ResolveKinds,
    InferTypes,
    CheckTypes,
    ResolveGenders,
    CheckGenders,
    InferWidths,
    CheckWidths,
    LowerTypes,
    ResolveKinds,
    InferTypes,
    CheckTypes,
    ResolveGenders,
    CheckGenders,
    InferWidths,
    CheckWidths,
    VerilogWrap,
    SplitExp,
    VerilogRename
  )
  def run(c: Circuit, w: Writer)
  {
    val loweredIR = PassUtils.executePasses(c, passes)
    VerilogEmitter.run(loweredIR, w)
    w.close
  }

}
