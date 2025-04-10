// SPDX-License-Identifier: Apache-2.0

package chisel3.simulator

import chisel3.{Module, RawModule}
import chisel3.experimental.inlinetest.{HasTests, TestChoice, TestHarness, TestResults}
import chisel3.simulator.stimulus.{InlineTestStimulus, ResetProcedure}
import chisel3.testing.HasTestingDirectory
import chisel3.util.simpleClassName
import java.nio.file.Files

trait SimulatorAPI {

  /** Simulate a [[RawModule]] without any initialization procedure.
    *
    * Use of this method is not advised when [[simulate]] can be used instead.
    * This method may cause problems with certain simulators as it is up to the
    * user to ensure that race conditions will not exist during the time zero
    * and reset procedure.
    *
    * @param module the Chisel module to generate
    * @param chiselOpts command line options to pass to Chisel
    * @param firtoolOpts command line options to pass to firtool
    * @param settings ChiselSim-related settings used for simulation
    * @param subdirectory an optional subdirectory for the test.  This will be a
    * subdirectory under what is provided by `testingDirectory`.
    * @param stimulus directed stimulus to use
    * @param testingDirectory a type class implementation that can be used to
    * change the behavior of where files will be created
    *
    * @note Take care when passing `chiselOpts`.  The following options are set
    * by default and if you set incompatible options, the simulation will fail.
    */
  def simulateRaw[T <: RawModule](
    module:       => T,
    chiselOpts:   Array[String] = Array.empty,
    firtoolOpts:  Array[String] = Array.empty,
    settings:     Settings[T] = Settings.defaultRaw[T],
    subdirectory: Option[String] = None
  )(stimulus: (T) => Unit)(
    implicit hasSimulator:        HasSimulator,
    testingDirectory:             HasTestingDirectory,
    chiselOptsModifications:      ChiselOptionsModifications,
    firtoolOptsModifications:     FirtoolOptionsModifications,
    commonSettingsModifications:  svsim.CommonSettingsModifications,
    backendSettingsModifications: svsim.BackendSettingsModifications
  ): Unit = {

    val modifiedTestingDirectory = subdirectory match {
      case Some(subdir) => testingDirectory.withSubdirectory(subdir)
      case None         => testingDirectory
    }

    hasSimulator
      .getSimulator(modifiedTestingDirectory)
      .simulate(module = module, chiselOpts = chiselOpts, firtoolOpts = firtoolOpts, settings = settings) { module =>
        stimulus(module.wrapped)
      }
      .result
  }

  /** Simulate a [[Module]] using a standard initialization procedure.
    *
    * For details of the initialization procedure see [[ResetProcedure]].
    *
    * @param module the Chisel module to generate
    * @param chiselOpts command line options to pass to Chisel
    * @param firtoolOpts command line options to pass to firtool
    * @param settings ChiselSim-related settings used for simulation
    * @param additionalResetCycles a number of _additional_ cycles to assert
    * reset for
    * @param subdirectory an optional subdirectory for the test.  This will be a
    * subdirectory under what is provided by `testingDirectory`.
    * @param stimulus directed stimulus to use
    * @param testingDirectory a type class implementation that can be used to
    * change the behavior of where files will be created
    *
    * @note Take care when passing `chiselOpts`.  The following options are set
    * by default and if you set incompatible options, the simulation will fail.
    */
  def simulate[T <: Module](
    module:                => T,
    chiselOpts:            Array[String] = Array.empty,
    firtoolOpts:           Array[String] = Array.empty,
    settings:              Settings[T] = Settings.default[T],
    additionalResetCycles: Int = 0,
    subdirectory:          Option[String] = None
  )(stimulus: (T) => Unit)(
    implicit hasSimulator:        HasSimulator,
    testingDirectory:             HasTestingDirectory,
    chiselOptsModifications:      ChiselOptionsModifications,
    firtoolOptsModifications:     FirtoolOptionsModifications,
    commonSettingsModifications:  svsim.CommonSettingsModifications,
    backendSettingsModifications: svsim.BackendSettingsModifications
  ): Unit = simulateRaw(
    module = module,
    chiselOpts = chiselOpts,
    firtoolOpts = firtoolOpts,
    settings = settings,
    subdirectory = subdirectory
  ) { dut =>
    ResetProcedure.module(additionalResetCycles)(dut)
    stimulus(dut)
  }

  /** Simulate the tests of a [[HasTests]] module.
    *
    * @param module the Chisel module to generate
    * @param test the choice of which test(s) to run
    * @param chiselOpts command line options to pass to Chisel
    * @param firtoolOpts command line options to pass to firtool
    * @param settings ChiselSim-related settings used for simulation
    * @param additionalResetCycles a number of _additional_ cycles to assert
    * reset for
    * @param subdirectory an optional subdirectory for the test.  This will be a
    * subdirectory under what is provided by `testingDirectory`.
    * @param stimulus directed stimulus to use
    * @param testingDirectory a type class implementation that can be used to
    * change the behavior of where files will be created
    *
    * @note Take care when passing `chiselOpts`.  The following options are set
    * by default and if you set incompatible options, the simulation will fail.
    */
  def simulateTests[T <: RawModule with HasTests](
    module:       => T,
    tests:        TestChoice.Type,
    timeout:      Int,
    chiselOpts:   Array[String] = Array.empty,
    firtoolOpts:  Array[String] = Array.empty,
    settings:     Settings[TestHarness[T, _]] = Settings.defaultRaw[TestHarness[T, _]],
    subdirectory: Option[String] = None
  )(
    implicit hasSimulator:        HasSimulator,
    testingDirectory:             HasTestingDirectory,
    chiselOptsModifications:      ChiselOptionsModifications,
    firtoolOptsModifications:     FirtoolOptionsModifications,
    commonSettingsModifications:  svsim.CommonSettingsModifications,
    backendSettingsModifications: svsim.BackendSettingsModifications
  ): TestResults = {
    val modifiedTestingDirectory = subdirectory match {
      case Some(subdir) => testingDirectory.withSubdirectory(subdir)
      case None         => testingDirectory
    }

    new TestResults(
      hasSimulator
        .getSimulator(modifiedTestingDirectory)
        .simulateTests(
          module = module,
          includeTestGlobs = tests.globs,
          chiselOpts = chiselOpts,
          firtoolOpts = firtoolOpts,
          settings = settings
        ) { dut => InlineTestStimulus(timeout)(dut.wrapped) }
    )
  }
}
