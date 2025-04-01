// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.{BaseModule, Param}
import chisel3.internal.Builder
import chisel3.internal.firrtl.ir._
import chisel3.internal.throwException
import chisel3.experimental.{SourceInfo, UnlocatableSourceInfo}

object FormalTest {

  /** Mark a module as a formal test.
    *
    * Other tools can use this information to, for example, collect all modules
    * marked as formal tests and run formal verification on them. This is
    * particularly useful in combination with the `UnitTest` trait.
    *
    * @param module The module to be marked.
    * @param params Optional user-defined test parameters.
    * @param name Optional name for the test. Uses the module name by default.
    *
    * @example
    * The following creates a module marked as a formal test:
    *
    * {{{
    * class TestHarness extends RawModule {
    *   FormalTest(this)
    * }
    * }}}
    *
    * Additional parameters may be passed to the test, which other tools may use
    * to control how the test is interpreted or executed:
    *
    * {{{
    * class TestHarness extends RawModule {
    *   FormalTest(
    *     this,
    *     MapTestParam(Map(
    *       "a" -> IntTestParam(9001),
    *       "b" -> DoubleTestParam(13.37),
    *       "c" -> StringTestParam("hello"),
    *       "d" -> ArrayTestParam(Seq(
    *         IntTestParam(9001),
    *         StringTestParam("hello")
    *       )),
    *       "e" -> MapTestParam(Map(
    *         "x" -> IntTestParam(9001),
    *         "y" -> StringTestParam("hello"),
    *       ))
    *     ))
    *   )
    * }
    * }}}
    */
  def apply(
    module: BaseModule,
    params: MapTestParam = MapTestParam(Map.empty),
    name:   String = ""
  )(implicit sourceInfo: SourceInfo): Unit = {
    val proposedName = if (name.isEmpty) {
      module._proposedName
    } else {
      name
    }
    val sanitizedName = Builder.globalNamespace.name(proposedName)
    Builder.components += DefTestMarker(DefTestMarker.Formal, sanitizedName, module, params, sourceInfo)
  }
}

object SimulationTest {

  /** Mark a module as a simulation test.
    *
    * Other tools can use this information to, for example, collect all modules
    * marked as simulation tests and run them in a simulator. This is
    * particularly useful in combination with the `UnitTest` trait.
    *
    * @param module The module to be marked.
    * @param params Optional user-defined test parameters.
    * @param name Optional name for the test. Uses the module name by default.
    *
    * @example
    * The following creates a module marked as a simulation test:
    *
    * {{{
    * class TestHarness extends RawModule {
    *   SimulationTest(this)
    * }
    * }}}
    *
    * Additional parameters may be passed to the test, which other tools may use
    * to control how the test is interpreted or executed:
    *
    * {{{
    * class TestHarness extends RawModule {
    *   SimulationTest(
    *     this,
    *     MapTestParam(Map(
    *       "a" -> IntTestParam(9001),
    *       "b" -> DoubleTestParam(13.37),
    *       "c" -> StringTestParam("hello"),
    *       "d" -> ArrayTestParam(Seq(
    *         IntTestParam(9001),
    *         StringTestParam("hello")
    *       )),
    *       "e" -> MapTestParam(Map(
    *         "x" -> IntTestParam(9001),
    *         "y" -> StringTestParam("hello"),
    *       ))
    *     ))
    *   )
    * }
    * }}}
    */
  def apply(
    module: BaseModule,
    params: MapTestParam = MapTestParam(Map.empty),
    name:   String = ""
  )(implicit sourceInfo: SourceInfo): Unit = {
    val proposedName = if (name.isEmpty) {
      module._proposedName
    } else {
      name
    }
    val sanitizedName = Builder.globalNamespace.name(proposedName)
    Builder.components += DefTestMarker(DefTestMarker.Simulation, sanitizedName, module, params, sourceInfo)
  }
}

/** Parameters for test declarations. */
sealed abstract class TestParam
case class IntTestParam(value: BigInt) extends TestParam
case class DoubleTestParam(value: Double) extends TestParam
case class StringTestParam(value: String) extends TestParam
case class ArrayTestParam(value: Seq[TestParam]) extends TestParam
case class MapTestParam(value: Map[String, TestParam]) extends TestParam
