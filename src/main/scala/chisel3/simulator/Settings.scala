// SPDX-License-Identifier: Apache-2.0

package chisel3.simulator

import chisel3.{Data, Module, RawModule, SimulationTestHarnessInterface}
import chisel3.experimental.dataview.reifySingleTarget
import svsim.CommonCompilationSettings.VerilogPreprocessorDefine
import svsim.Workspace

/** This object implements an enumeration of classes that can be used to
  * generate macros for use in [[Settings]].
  */
object MacroText {

  /** The type of all enumerations for macros */
  sealed trait Type[A <: RawModule] {

    /** Return the preprocessor define associated with this macro.
      *
      * @param macroName the macros name
      * @param elaboratedModule a Chisel module to use to resolve substitutions
      * @return the macro string
      */
    private[simulator] def toPreprocessorDefine(
      macroName:        String,
      elaboratedModule: ElaboratedModule[A]
    ): VerilogPreprocessorDefine

  }

  /** Given a [[Data]] and its parent [[ElaboratedModule]], lookup the underlying hardware. */
  private def lookupSignal[A <: RawModule](data: Data, elaboratedModule: ElaboratedModule[A]): String = {
    val reified = reifySingleTarget(data).getOrElse {
      throw new IllegalArgumentException(
        s"Cannot use $data as a macro signal because it does not map to a single Data."
      )
    }
    elaboratedModule.portMap(reified).name
  }

  /** A macro that will return macro text with the name of a signal in the design
    * under test
    *
    * @param get a function that accesses a singal in the design under test
    */
  case class Signal[A <: RawModule](get: (A) => Data) extends Type[A] {

    override private[simulator] def toPreprocessorDefine(
      macroName:        String,
      elaboratedModule: ElaboratedModule[A]
    ) = {
      val port = lookupSignal(get(elaboratedModule.wrapped), elaboratedModule)
      VerilogPreprocessorDefine(macroName, s"${Workspace.testbenchModuleName}.$port")
    }

  }

  /** A macro that will return macro text with the logical inversion a signal in
    * the design under test
    *
    * @param get a function that accesses a singal in the design under test
    */
  case class NotSignal[A <: RawModule](get: (A) => Data) extends Type[A] {

    override private[simulator] def toPreprocessorDefine(
      macroName:        String,
      elaboratedModule: ElaboratedModule[A]
    ) = {
      val port = lookupSignal(get(elaboratedModule.wrapped), elaboratedModule)
      VerilogPreprocessorDefine(macroName, s"!${Workspace.testbenchModuleName}.$port")
    }

  }

}

/** Settings for controlling ChiselSim simulations
  *
  * These setings are only intended to be associated with Chisel, FIRRTL, and
  * FIRRTL's Verilog ABI and not to do with lower-level control of the FIRRTL
  * compilation itself or the Verilog compilation and simulation.
  *
  * @param layerControl determines which [[chisel3.layer.Layer]]s should be
  * @param assertVerboseCond a condition that guards the printing of assert
  * messages created from `circt_chisel_ifelsefatal` intrinsics
  * @param printfCond a condition that guards printing of [[chisel3.printf]]s
  * @param stopCond a condition that guards terminating the simulation (via
  * `$fatal`) for asserts created from `circt_chisel_ifelsefatal` intrinsics
  * enabled _during Verilog elaboration_.
  * @param plusArgs Verilog `$value$plusargs` or `$test$plusargs` to set at
  * simulation runtime.
  * @param enableWavesAtTimeZero enable waveform dumping at time zero. This
  * requires a simulator capable of dumping waves.
  * @param randomization random initialization settings to use
  */
final class Settings[A <: RawModule] private[simulator] (
  /** Layers to turn on/off during Verilog elaboration */
  val verilogLayers:         LayerControl.Type,
  val assertVerboseCond:     Option[MacroText.Type[A]],
  val printfCond:            Option[MacroText.Type[A]],
  val stopCond:              Option[MacroText.Type[A]],
  val plusArgs:              Seq[svsim.PlusArg],
  val enableWavesAtTimeZero: Boolean,
  val randomization:         Randomization
) {

  def copy(
    verilogLayers:         LayerControl.Type = verilogLayers,
    assertVerboseCond:     Option[MacroText.Type[A]] = assertVerboseCond,
    printfCond:            Option[MacroText.Type[A]] = printfCond,
    stopCond:              Option[MacroText.Type[A]] = stopCond,
    plusArgs:              Seq[svsim.PlusArg] = plusArgs,
    enableWavesAtTimeZero: Boolean = enableWavesAtTimeZero,
    randomization:         Randomization = randomization
  ) =
    new Settings(verilogLayers, assertVerboseCond, printfCond, stopCond, plusArgs, enableWavesAtTimeZero, randomization)

  private[simulator] def preprocessorDefines(
    elaboratedModule: ElaboratedModule[A]
  ): Seq[VerilogPreprocessorDefine] = {

    Seq(
      (assertVerboseCond -> "ASSERT_VERBOSE_COND"),
      (printfCond -> "PRINTF_COND"),
      (stopCond -> "STOP_COND")
    ).flatMap {
      case (Some(a), macroName) => Some(a.toPreprocessorDefine(macroName, elaboratedModule))
      case (None, _)            => None
    } ++ verilogLayers.preprocessorDefines(elaboratedModule) ++ randomization.toPreprocessorDefines

  }

}

/** This object contains factories of [[Settings]].
  *
  */
object Settings {

  /** Retun a default [[Settings]] for a [[Module]].  Macros will be set to
    * disable [[chisel3.assert]]-style assertions using the [[Module]]'s reset
    * port.
    *
    * Note: this _requires_ that an explicit type parameter is provided.  You
    * must invoke this method like:
    *
    * {{{
    * Settings.default[Foo]
    * }}}
    *
    * If you invoke this method like the following, you will get an error:
    *
    * {{{
    * Settings.default
    * }}}
    */
  final def default[A <: Module]: Settings[A] = new Settings[A](
    verilogLayers = LayerControl.EnableAll,
    assertVerboseCond = Some(MacroText.NotSignal(get = _.reset)),
    printfCond = Some(MacroText.NotSignal(get = _.reset)),
    stopCond = Some(MacroText.NotSignal(get = _.reset)),
    plusArgs = Seq.empty,
    enableWavesAtTimeZero = false,
    randomization = Randomization.random
  )

  /** Retun a default [[Settings]] for a [[RawModule]].
    *
    * This differs from [[default]] in that it cannot set default values for
    * macros because a [[RawModule]] has no defined reset port.  You will likely
    * want to override the macros after using this factory.
    *
    * Note: this _requires_ that an explicit type parameter is provided.  You
    * must invoke this method like:
    *
    * {{{
    * Settings.defaultRaw[Foo]
    * }}}
    *
    * If you invoke this method like the following, you will get an error:
    *
    * {{{
    * Settings.defaultRaw
    * }}}
    */
  final def defaultRaw[A <: RawModule]: Settings[A] = new Settings[A](
    verilogLayers = LayerControl.EnableAll,
    assertVerboseCond = None,
    printfCond = None,
    stopCond = None,
    plusArgs = Seq.empty,
    enableWavesAtTimeZero = false,
    randomization = Randomization.random
  )

  /** Return a default [[Settings]] for a [[SimulationTestHarnessInterface]].  Macros will be set to
    * disable [[chisel3.assert]]-style assertions using the [[SimulationTestHarnessInterface]]'s init
    * port.
    *
    * Note: this _requires_ that an explicit type parameter is provided.  You
    * must invoke this method like:
    *
    * {{{
    * Settings.default[Foo]
    * }}}
    *
    * If you invoke this method like the following, you will get an error:
    *
    * {{{
    * Settings.default
    * }}}
    */
  final def defaultTest[A <: RawModule with SimulationTestHarnessInterface]: Settings[A] = new Settings[A](
    verilogLayers = LayerControl.EnableAll,
    assertVerboseCond = Some(MacroText.NotSignal(get = _.init)),
    printfCond = Some(MacroText.NotSignal(get = _.init)),
    stopCond = Some(MacroText.NotSignal(get = _.init)),
    plusArgs = Seq.empty,
    enableWavesAtTimeZero = false,
    randomization = Randomization.random
  )

  /** Simple factory for construcing a [[Settings]] from arguments.
    *
    * This method primarily exists as a way to make future refactors that add
    * options to [[Settings]] easier.
    *
    * @param layerControl determines which [[chisel3.layer.Layer]]s should be
    * @param assertVerboseCond a condition that guards the printing of assert
    * messages created from `circt_chisel_ifelsefatal` intrinsics
    * @param printfCond a condition that guards printing of [[chisel3.printf]]s
    * @param stopCond a condition that guards terminating the simulation (via
    * `$fatal`) for asserts created from `circt_chisel_ifelsefatal` intrinsics
    * @return a [[Settings]] with the provided parameters set
    */
  def apply[A <: RawModule](
    verilogLayers:         LayerControl.Type,
    assertVerboseCond:     Option[MacroText.Type[A]],
    printfCond:            Option[MacroText.Type[A]],
    stopCond:              Option[MacroText.Type[A]],
    plusArgs:              Seq[svsim.PlusArg],
    enableWavesAtTimeZero: Boolean,
    randomization:         Randomization
  ): Settings[A] = new Settings(
    verilogLayers = verilogLayers,
    assertVerboseCond = assertVerboseCond,
    printfCond = printfCond,
    stopCond = stopCond,
    plusArgs = plusArgs,
    enableWavesAtTimeZero = enableWavesAtTimeZero,
    randomization = randomization
  )

}
