// SPDX-License-Identifier: Apache-2.0

package chisel3.testing.scalatest

import chisel3.simulator.HasSimulator
import chisel3.testing.scalatest.HasConfigMap

/** Mix-in that brings a `HasSimulator` type class implementation into scope
  * based on a command line argument.
  *
  * This provides default implementations of simulators.  Users can change the
  * simulators provided by overriding the default protected members
  * `cliSimulatorMap` and `defaultCliSimulator`.
  *
  */
trait HasCliSimulator { this: HasConfigMap =>

  /** A mapping of simulator names to simulators. */
  protected def cliSimulatorMap: Map[String, HasSimulator] = Map(
    "verilator" -> HasSimulator.simulators.verilator(),
    "vcs" -> HasSimulator.simulators.vcs()
  )

  /** An optional default simulator to use if the user does _not_ provide a simulator.
    *
    * If `Some` then the provided default will be used.  If `None`, then a
    * simulator must be provided.
    */
  protected def defaultCliSimulator: Option[HasSimulator] = Some(HasSimulator.default)

  implicit def cliSimulator: HasSimulator = configMap.getOptional[String]("simulator") match {
    case None =>
      defaultCliSimulator.getOrElse(
        throw new IllegalArgumentException(
          s"""a simulator must be provided to this test using '-Dsimulator=<simulator-name>' where <simulator-name> must be one of ${cliSimulatorMap.keys
              .mkString("[", ", ", "]")}"""
        )
      )
    case Some(simulator) =>
      cliSimulatorMap.getOrElse(
        simulator,
        throw new IllegalArgumentException(
          s"""illegal simulator '$simulator', must be one of ${cliSimulatorMap.keys.mkString("[", ", ", "]")}"""
        )
      )
  }

}
