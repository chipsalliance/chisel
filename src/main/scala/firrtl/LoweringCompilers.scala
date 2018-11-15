// See LICENSE for license details.

package firrtl

sealed abstract class CoreTransform extends SeqTransform

/** This transforms "CHIRRTL", the chisel3 IR, to "Firrtl". Note the resulting
  * circuit has only IR nodes, not WIR.
  */
class ChirrtlToHighFirrtl extends CoreTransform {
  def inputForm = ChirrtlForm
  def outputForm = HighForm
  def transforms = Seq(
    passes.CheckChirrtl,
    passes.CInferTypes,
    passes.CInferMDir,
    passes.RemoveCHIRRTL)
}

/** Converts from the bare intermediate representation (ir.scala)
  * to a working representation (WIR.scala)
  */
class IRToWorkingIR extends CoreTransform {
  def inputForm = HighForm
  def outputForm = HighForm
  def transforms = Seq(passes.ToWorkingIR)
}

/** Resolves types, kinds, and genders, and checks the circuit legality.
  * Operates on working IR nodes and high Firrtl.
  */
class ResolveAndCheck extends CoreTransform {
  def inputForm = HighForm
  def outputForm = HighForm
  def transforms = Seq(
    passes.CheckHighForm,
    passes.ResolveKinds,
    passes.InferTypes,
    passes.CheckTypes,
    passes.Uniquify,
    passes.ResolveKinds,
    passes.InferTypes,
    passes.ResolveGenders,
    passes.CheckGenders,
    passes.InferWidths,
    passes.CheckWidths)
}

/** Expands aggregate connects, removes dynamic accesses, and when
  * statements. Checks for uninitialized values. Must accept a
  * well-formed graph.
  * Operates on working IR nodes.
  */
class HighFirrtlToMiddleFirrtl extends CoreTransform {
  def inputForm = HighForm
  def outputForm = MidForm
  def transforms = Seq(
    passes.PullMuxes,
    passes.ReplaceAccesses,
    passes.ExpandConnects,
    passes.RemoveAccesses,
    passes.Uniquify,
    passes.ExpandWhens,
    passes.CheckInitialization,
    passes.ResolveKinds,
    passes.InferTypes,
    passes.CheckTypes,
    passes.ResolveGenders,
    passes.InferWidths,
    passes.CheckWidths,
    passes.ConvertFixedToSInt,
    passes.ZeroWidth,
    passes.InferTypes)
}

/** Expands all aggregate types into many ground-typed components. Must
  * accept a well-formed graph of only middle Firrtl features.
  * Operates on working IR nodes.
  */
class MiddleFirrtlToLowFirrtl extends CoreTransform {
  def inputForm = MidForm
  def outputForm = LowForm
  def transforms = Seq(
    passes.LowerTypes,
    passes.ResolveKinds,
    passes.InferTypes,
    passes.ResolveGenders,
    passes.InferWidths,
    passes.Legalize,
    new firrtl.transforms.RemoveReset,
    new firrtl.transforms.CheckCombLoops,
    new firrtl.transforms.RemoveWires)
}

/** Runs a series of optimization passes on LowFirrtl
  * @note This is currently required for correct Verilog emission
  * TODO Fix the above note
  */
class LowFirrtlOptimization extends CoreTransform {
  def inputForm = LowForm
  def outputForm = LowForm
  def transforms = Seq(
    passes.RemoveValidIf,
    new firrtl.transforms.ConstantPropagation,
    passes.PadWidths,
    new firrtl.transforms.ConstantPropagation,
    passes.Legalize,
    passes.memlib.VerilogMemDelays, // TODO move to Verilog emitter
    new firrtl.transforms.ConstantPropagation,
    passes.SplitExpressions,
    new firrtl.transforms.CombineCats,
    passes.CommonSubexpressionElimination,
    new firrtl.transforms.DeadCodeElimination)
}
/** Runs runs only the optimization passes needed for Verilog emission */
class MinimumLowFirrtlOptimization extends CoreTransform {
  def inputForm = LowForm
  def outputForm = LowForm
  def transforms = Seq(
    passes.Legalize,
    passes.memlib.VerilogMemDelays, // TODO move to Verilog emitter
    passes.SplitExpressions)
}


import CompilerUtils.getLoweringTransforms
import firrtl.transforms.BlackBoxSourceHelper

/** Emits input circuit
  * Will replace Chirrtl constructs with Firrtl
  */
class HighFirrtlCompiler extends Compiler {
  def emitter = new HighFirrtlEmitter
  def transforms: Seq[Transform] = getLoweringTransforms(ChirrtlForm, HighForm)
}

/** Emits middle Firrtl input circuit */
class MiddleFirrtlCompiler extends Compiler {
  def emitter = new MiddleFirrtlEmitter
  def transforms: Seq[Transform] = getLoweringTransforms(ChirrtlForm, MidForm)
}

/** Emits lowered input circuit */
class LowFirrtlCompiler extends Compiler {
  def emitter = new LowFirrtlEmitter
  def transforms: Seq[Transform] = getLoweringTransforms(ChirrtlForm, LowForm)
}

/** Emits Verilog */
class VerilogCompiler extends Compiler {
  def emitter = new VerilogEmitter
  def transforms: Seq[Transform] = getLoweringTransforms(ChirrtlForm, LowForm) ++
    Seq(new LowFirrtlOptimization)
}

/** Emits Verilog without optimizations */
class MinimumVerilogCompiler extends Compiler {
  def emitter = new VerilogEmitter
  def transforms: Seq[Transform] = getLoweringTransforms(ChirrtlForm, LowForm) ++
    Seq(new MinimumLowFirrtlOptimization, new BlackBoxSourceHelper)
}

/** Currently just an alias for the [[VerilogCompiler]] */
class SystemVerilogCompiler extends VerilogCompiler {
  Driver.dramaticWarning("SystemVerilog Compiler behaves the same as the Verilog Compiler!")
}
