// SPDX-License-Identifier: Apache-2.0

package chisel3.simulator

import chisel3.{Data, Module, RawModule}
import svsim.CommonCompilationSettings.VerilogPreprocessorDefine
import svsim.Workspace

/** This object implements an enumeration of classes that can be used to
  * generate macros for use in [[ChiselSettings]].
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
      val port = elaboratedModule.portMap(get(elaboratedModule.wrapped)).name
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
      val port = elaboratedModule.portMap(get(elaboratedModule.wrapped)).name
      VerilogPreprocessorDefine(macroName, s"!${Workspace.testbenchModuleName}.$port")
    }

  }

}

/** This struct describes settings related to controlling a Chisel simulation.  Thes
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
  */
final class ChiselSettings[A <: RawModule] private[simulator] (
  /** Layers to turn on/off during Verilog elaboration */
  val verilogLayers:     LayerControl.Type,
  val assertVerboseCond: Option[MacroText.Type[A]],
  val printfCond:        Option[MacroText.Type[A]],
  val stopCond:          Option[MacroText.Type[A]]
) {

  def copy(
    verilogLayers:     LayerControl.Type = verilogLayers,
    assertVerboseCond: Option[MacroText.Type[A]] = assertVerboseCond,
    printfCond:        Option[MacroText.Type[A]] = printfCond,
    stopCond:          Option[MacroText.Type[A]] = stopCond
  ) =
    new ChiselSettings(verilogLayers, assertVerboseCond, printfCond, stopCond)

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
    } ++ verilogLayers.preprocessorDefines(elaboratedModule)

  }

}

/** This object contains factories of [[ChiselSettings]].
  *
  */
object ChiselSettings {

  /** Retun a default [[ChiselSettings]] for a [[Module]].  Macros will be set to
    * disable [[chisel3.assert]]-style assertions using the [[Module]]'s reset
    * port.
    *
    * Note: this _requires_ that an explicit type parameter is provided.  You
    * must invoke this method like:
    *
    * {{{
    * ChiselSettings.default[Foo]
    * }}}
    *
    * If you invoke this method like the following, you will get an error:
    *
    * {{{
    * ChiselSettings.default
    * }}}
    */
  final def default[A <: Module]: ChiselSettings[A] = new ChiselSettings[A](
    verilogLayers = LayerControl.EnableAll,
    assertVerboseCond = Some(MacroText.NotSignal(get = _.reset)),
    printfCond = Some(MacroText.NotSignal(get = _.reset)),
    stopCond = Some(MacroText.NotSignal(get = _.reset))
  )

  /** Retun a default [[ChiselSettings]] for a [[Module]].
    *
    *  This differs from [[default]] in that it cannot set default values for
    *  macros because a [[RawModule]] has no defined reset port.  You will
    *  likely want to override the macros after using this factory.
    *
    * Note: this _requires_ that an explicit type parameter is provided.  You
    * must invoke this method like:
    *
    * {{{
    * ChiselSettings.default[Foo]
    * }}}
    *
    * If you invoke this method like the following, you will get an error:
    *
    * {{{
    * ChiselSettings.default
    * }}}
    */
  final def defaultRaw[A <: RawModule]: ChiselSettings[A] = new ChiselSettings[A](
    verilogLayers = LayerControl.EnableAll,
    assertVerboseCond = None,
    printfCond = None,
    stopCond = None
  )

  /** Simple factory for construcing a [[ChiselSettings]] from arguments.
    *
    * This method primarily exists as a way to make future refactors that add
    * options to [[ChiselSettings]] easier.
    *
    * @param layerControl determines which [[chisel3.layer.Layer]]s should be
    * @param assertVerboseCond a condition that guards the printing of assert
    * messages created from `circt_chisel_ifelsefatal` intrinsics
    * @param printfCond a condition that guards printing of [[chisel3.printf]]s
    * @param stopCond a condition that guards terminating the simulation (via
    * `$fatal`) for asserts created from `circt_chisel_ifelsefatal` intrinsics
    * @return a [[ChiselSettings]] with the provided parameters set
    */
  def apply[A <: RawModule](
    verilogLayers:     LayerControl.Type,
    assertVerboseCond: Option[MacroText.Type[A]],
    printfCond:        Option[MacroText.Type[A]],
    stopCond:          Option[MacroText.Type[A]]
  ): ChiselSettings[A] = new ChiselSettings(
    verilogLayers = verilogLayers,
    assertVerboseCond = assertVerboseCond,
    printfCond = printfCond,
    stopCond = stopCond
  )

}
