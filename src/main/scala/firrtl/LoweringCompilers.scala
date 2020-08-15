// See LICENSE for license details.

package firrtl

import firrtl.transforms.IdentityTransform
import firrtl.stage.{Forms, TransformManager}

@deprecated("Use a TransformManager or some other Stage/Phase class. Will be removed in 1.4.", "FIRRTL 1.2")
sealed abstract class CoreTransform extends SeqTransform

/** This transforms "CHIRRTL", the chisel3 IR, to "Firrtl". Note the resulting
  * circuit has only IR nodes, not WIR.
  */
@deprecated(
  "Use 'new TransformManager(Forms.MinimalHighForm, Forms.ChirrtlForm)'. This will be removed in 1.4.",
  "FIRRTL 1.3"
)
class ChirrtlToHighFirrtl extends CoreTransform {
  def inputForm = ChirrtlForm
  def outputForm = HighForm
  def transforms = new TransformManager(Forms.MinimalHighForm, Forms.ChirrtlForm).flattenedTransformOrder
}

/** Converts from the bare intermediate representation (ir.scala)
  * to a working representation (WIR.scala)
  */
@deprecated(
  "Use 'new TransformManager(Forms.WorkingIR, Forms.MinimalHighForm)'. This will be removed in 1.4.",
  "FIRRTL 1.3"
)
class IRToWorkingIR extends CoreTransform {
  def inputForm = HighForm
  def outputForm = HighForm
  def transforms = new TransformManager(Forms.WorkingIR, Forms.MinimalHighForm).flattenedTransformOrder
}

/** Resolves types, kinds, and flows, and checks the circuit legality.
  * Operates on working IR nodes and high Firrtl.
  */
@deprecated(
  "Use 'new TransformManager(Forms.Resolved, Forms.WorkingIR)'. This will be removed in 1.4.",
  "FIRRTL 1.3"
)
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
@deprecated(
  "Use 'new TransformManager(Forms.MidForm, Forms.Deduped)'. This will be removed in 1.4.",
  "FIRRTL 1.3"
)
class HighFirrtlToMiddleFirrtl extends CoreTransform {
  def inputForm = HighForm
  def outputForm = MidForm
  def transforms = new TransformManager(Forms.MidForm, Forms.Deduped).flattenedTransformOrder
}

/** Expands all aggregate types into many ground-typed components. Must
  * accept a well-formed graph of only middle Firrtl features.
  * Operates on working IR nodes.
  */
@deprecated(
  "Use 'new TransformManager(Forms.LowForm, Forms.MidForm)'. This will be removed in 1.4.",
  "FIRRTL 1.3"
)
class MiddleFirrtlToLowFirrtl extends CoreTransform {
  def inputForm = MidForm
  def outputForm = LowForm
  def transforms = new TransformManager(Forms.LowForm, Forms.MidForm).flattenedTransformOrder
}

/** Runs a series of optimization passes on LowFirrtl
  * @note This is currently required for correct Verilog emission
  * TODO Fix the above note
  */
@deprecated(
  "Use 'new TransformManager(Forms.LowFormOptimized, Forms.LowForm)'. This will be removed in 1.4.",
  "FIRRTL 1.3"
)
class LowFirrtlOptimization extends CoreTransform {
  def inputForm = LowForm
  def outputForm = LowForm
  def transforms = new TransformManager(Forms.LowFormOptimized, Forms.LowForm).flattenedTransformOrder
}

/** Runs runs only the optimization passes needed for Verilog emission */
@deprecated(
  "Use 'new TransformManager(Forms.LowFormMinimumOptimized, Forms.LowForm)'. This will be removed in 1.4.",
  "FIRRTL 1.3"
)
class MinimumLowFirrtlOptimization extends CoreTransform {
  def inputForm = LowForm
  def outputForm = LowForm
  def transforms = new TransformManager(Forms.LowFormMinimumOptimized, Forms.LowForm).flattenedTransformOrder
}

/** Emits input circuit with no changes
  *
  * Primarily useful for changing between .fir and .pb serialized formats
  */
@deprecated(
  "Use stage.{FirrtlStage, FirrtlMain} or stage.transforms.Compiler(Seq(Dependency[ChirrtlEmitter]))",
  "FIRRTL 1.3"
)
class NoneCompiler extends Compiler {
  val emitter = new ChirrtlEmitter
  def transforms: Seq[Transform] = Seq(new IdentityTransform(ChirrtlForm))
}

/** Emits input circuit
  * Will replace Chirrtl constructs with Firrtl
  */
@deprecated(
  "Use stage.{FirrtlStage, FirrtlMain} stage.transforms.Compiler(Seq(Dependency[HighFirrtlEmitter]))",
  "FIRRTL 1.3"
)
class HighFirrtlCompiler extends Compiler {
  val emitter = new HighFirrtlEmitter
  def transforms: Seq[Transform] = Forms.HighForm.map(_.getObject)
}

/** Emits middle Firrtl input circuit */
@deprecated(
  "Use stage.{FirrtlStage, FirrtlMain} stage.transforms.Compiler(Dependency[MiddleFirrtlEmitter])",
  "FIRRTL 1.3"
)
class MiddleFirrtlCompiler extends Compiler {
  val emitter = new MiddleFirrtlEmitter
  def transforms: Seq[Transform] = Forms.MidForm.map(_.getObject)
}

/** Emits lowered input circuit */
@deprecated(
  "Use stage.{FirrtlStage, FirrtlMain} stage.transforms.Compiler(Dependency[LowFirrtlEmitter])",
  "FIRRTL 1.3"
)
class LowFirrtlCompiler extends Compiler {
  val emitter = new LowFirrtlEmitter
  def transforms: Seq[Transform] = Forms.LowForm.map(_.getObject)
}

/** Emits Verilog */
@deprecated(
  "Use stage.{FirrtlStage, FirrtlMain} stage.transforms.Compiler(Dependency[VerilogEmitter])",
  "FIRRTL 1.3"
)
class VerilogCompiler extends Compiler {
  val emitter = new VerilogEmitter
  def transforms: Seq[Transform] = Forms.LowFormOptimized.map(_.getObject)
}

/** Emits Verilog without optimizations */
@deprecated(
  "Use stage.{FirrtlStage, FirrtlMain} stage.transforms.Compiler(Dependency[MinimumVerilogEmitter])",
  "FIRRTL 1.3"
)
class MinimumVerilogCompiler extends Compiler {
  val emitter = new MinimumVerilogEmitter
  def transforms: Seq[Transform] = Forms.LowFormMinimumOptimized.map(_.getObject)
}

/** Currently just an alias for the [[VerilogCompiler]] */
@deprecated(
  "Use stage.{FirrtlStage, FirrtlMain} stage.transforms.Compiler(Dependency[SystemVerilogEmitter])",
  "FIRRTL 1.3"
)
class SystemVerilogCompiler extends VerilogCompiler {
  override val emitter = new SystemVerilogEmitter
}
