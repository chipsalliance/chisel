// See LICENSE for license details.

package chisel3

import internal.firrtl._
import firrtl._
import firrtl.util.{BackendCompilationUtilities => FirrtlBackendCompilationUtilities}
import java.io._


/**
  * The Driver provides methods to invoke the chisel3 compiler and the firrtl compiler.
  * By default firrtl is automatically run after chisel.  an [[ExecutionOptionsManager]]
  * is needed to manage options.  It can parser command line arguments or coordinate
  * multiple chisel toolchain tools options.
  *
  * @example
  *          {{{
  *          val optionsManager = new ExecutionOptionsManager("chisel3")
  *              with HasFirrtlOptions
  *              with HasChiselExecutionOptions {
  *            commonOptions = CommonOption(targetDirName = "my_target_dir")
  *            chiselOptions = ChiselExecutionOptions(runFirrtlCompiler = false)
  *          }
  *          chisel3.Driver.execute(optionsManager, () => new Dut)
  *          }}}
  * or via command line arguments
  * @example {{{
  *          args = "--no-run-firrtl --target-dir my-target-dir".split(" +")
  *          chisel3.execute(args, () => new DUT)
  *          }}}
  */

trait BackendCompilationUtilities extends FirrtlBackendCompilationUtilities {
  /** Compile Chirrtl to Verilog by invoking Firrtl inside the same JVM
    *
    * @param prefix basename of the file
    * @param dir    directory where file lives
    * @return       true if compiler completed successfully
    */
  def compileFirrtlToVerilog(prefix: String, dir: File): Boolean = {
    val optionsManager = new ExecutionOptionsManager("chisel3") with HasChiselExecutionOptions with HasFirrtlOptions {
      commonOptions = CommonOptions(topName = prefix, targetDirName = dir.getAbsolutePath)
      firrtlOptions = FirrtlExecutionOptions(compilerName = "verilog")
    }

    firrtl.Driver.execute(optionsManager) match {
      case _: FirrtlExecutionSuccess => true
      case _: FirrtlExecutionFailure => false
    }
  }
}

/**
  * This family provides return values from the chisel3 and possibly firrtl compile steps
  */
@deprecated("This will be removed in Chisel 3.5", "Chisel3 3.4")
trait ChiselExecutionResult

/**
  *
  * @param circuitOption  Optional circuit, has information like circuit name
  * @param emitted            The emitted Chirrrl text
  * @param firrtlResultOption Optional Firrtl result, @see freechipsproject/firrtl for details
  */
@deprecated("This will be removed in Chisel 3.5", "Chisel 3.4")
case class ChiselExecutionSuccess(
                                  circuitOption: Option[Circuit],
                                  emitted: String,
                                  firrtlResultOption: Option[FirrtlExecutionResult]
                                  ) extends ChiselExecutionResult

/**
  * Getting one of these indicates failure of some sort.
  *
  * @param message A clue might be provided here.
  */
@deprecated("This will be removed in Chisel 3.5", "Chisel 3.4")
case class ChiselExecutionFailure(message: String) extends ChiselExecutionResult
