// See LICENSE for license details.

package firrtl

import firrtl.transforms.IdentityTransform
import firrtl.options.StageUtils
import firrtl.stage.{Forms, TransformManager}

@deprecated("Use a TransformManager or some other Stage/Phase class. Will be removed in 1.3.", "1.2")
sealed abstract class CoreTransform extends SeqTransform

/** This transforms "CHIRRTL", the chisel3 IR, to "Firrtl". Note the resulting
  * circuit has only IR nodes, not WIR.
  */
@deprecated("Use a TransformManager to handle lowering. Will be removed in 1.3.", "1.2")
class ChirrtlToHighFirrtl extends CoreTransform {
  def inputForm = ChirrtlForm
  def outputForm = HighForm
  def transforms = new TransformManager(Forms.MinimalHighForm, Forms.ChirrtlForm).flattenedTransformOrder
}

/** Converts from the bare intermediate representation (ir.scala)
  * to a working representation (WIR.scala)
  */
@deprecated("Use a TransformManager to handle lowering. Will be removed in 1.3.", "1.2")
class IRToWorkingIR extends CoreTransform {
  def inputForm = HighForm
  def outputForm = HighForm
  def transforms = new TransformManager(Forms.WorkingIR, Forms.MinimalHighForm).flattenedTransformOrder
}

/** Resolves types, kinds, and flows, and checks the circuit legality.
  * Operates on working IR nodes and high Firrtl.
  */
@deprecated("Use a TransformManager to handle lowering. Will be removed in 1.3.", "1.2")
class ResolveAndCheck extends CoreTransform {
  def inputForm = HighForm
  def outputForm = HighForm
  def transforms = new TransformManager(Forms.Resolved, Forms.WorkingIR).flattenedTransformOrder
}

/** Expands aggregate connects, removes dynamic accesses, and when
  * statements. Checks for uninitialized values. Must accept a
  * well-formed graph.
  * Operates on working IR nodes.
  */
@deprecated("Use a TransformManager to handle lowering. Will be removed in 1.3.", "1.2")
class HighFirrtlToMiddleFirrtl extends CoreTransform {
  def inputForm = HighForm
  def outputForm = MidForm
  def transforms = new TransformManager(Forms.MidForm, Forms.Deduped).flattenedTransformOrder
}

/** Expands all aggregate types into many ground-typed components. Must
  * accept a well-formed graph of only middle Firrtl features.
  * Operates on working IR nodes.
  */
@deprecated("Use a TransformManager to handle lowering. Will be removed in 1.3.", "1.2")
class MiddleFirrtlToLowFirrtl extends CoreTransform {
  def inputForm = MidForm
  def outputForm = LowForm
  def transforms = new TransformManager(Forms.LowForm, Forms.MidForm).flattenedTransformOrder
}

/** Runs a series of optimization passes on LowFirrtl
  * @note This is currently required for correct Verilog emission
  * TODO Fix the above note
  */
@deprecated("Use a TransformManager to handle lowering. Will be removed in 1.3.", "1.2")
class LowFirrtlOptimization extends CoreTransform {
  def inputForm = LowForm
  def outputForm = LowForm
  def transforms = new TransformManager(Forms.LowFormOptimized, Forms.LowForm).flattenedTransformOrder
}
/** Runs runs only the optimization passes needed for Verilog emission */
@deprecated("Use a TransformManager to handle lowering. Will be removed in 1.3.", "1.2")
class MinimumLowFirrtlOptimization extends CoreTransform {
  def inputForm = LowForm
  def outputForm = LowForm
  def transforms = new TransformManager(Forms.LowFormMinimumOptimized, Forms.LowForm).flattenedTransformOrder
}


import CompilerUtils.getLoweringTransforms

/** Emits input circuit with no changes
  *
  * Primarily useful for changing between .fir and .pb serialized formats
  */
class NoneCompiler extends Compiler {
  val emitter = new ChirrtlEmitter
  def transforms: Seq[Transform] = Seq(new IdentityTransform(ChirrtlForm))
}

/** Emits input circuit
  * Will replace Chirrtl constructs with Firrtl
  */
class HighFirrtlCompiler extends Compiler {
  val emitter = new HighFirrtlEmitter
  def transforms: Seq[Transform] = getLoweringTransforms(ChirrtlForm, HighForm)
}

/** Emits middle Firrtl input circuit */
class MiddleFirrtlCompiler extends Compiler {
  val emitter = new MiddleFirrtlEmitter
  def transforms: Seq[Transform] = getLoweringTransforms(ChirrtlForm, MidForm)
}

/** Emits lowered input circuit */
class LowFirrtlCompiler extends Compiler {
  val emitter = new LowFirrtlEmitter
  def transforms: Seq[Transform] = getLoweringTransforms(ChirrtlForm, LowForm)
}

/** Emits Verilog */
class VerilogCompiler extends Compiler {
  val emitter = new VerilogEmitter
  def transforms: Seq[Transform] = getLoweringTransforms(ChirrtlForm, LowForm) ++
    Seq(new LowFirrtlOptimization)
}

/** Emits Verilog without optimizations */
class MinimumVerilogCompiler extends Compiler {
  val emitter = new MinimumVerilogEmitter
  def transforms: Seq[Transform] = getLoweringTransforms(ChirrtlForm, LowForm) ++
    Seq(new MinimumLowFirrtlOptimization)
}

/** Currently just an alias for the [[VerilogCompiler]] */
class SystemVerilogCompiler extends VerilogCompiler {
  override val emitter = new SystemVerilogEmitter
  StageUtils.dramaticWarning("SystemVerilog Compiler behaves the same as the Verilog Compiler!")
}
