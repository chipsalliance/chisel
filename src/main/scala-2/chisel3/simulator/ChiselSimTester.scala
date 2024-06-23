package chisel3
package simulator

import chisel3.util._
import chisel3.reflect.DataMirror
import chisel3.experimental.SourceInfo

import chisel3.simulator.Simulator._
import svsim._

import org.scalatest.TestSuite
import org.scalatest.Assertions
import org.scalatest.TestSuiteMixin
import org.scalatest.Outcome
import org.scalatest.exceptions.TestFailedException

import scala.util.DynamicVariable
import scala.util.{Failure, Success, Try}

trait ChiselSimTester extends ChiselSimAPI with Assertions with TestSuiteMixin { this: TestSuite =>

  def testName: Option[String] = scalaTestContext.value.map(_.name)

  // Provides test fixture data as part of 'global' context during test runs
  protected var scalaTestContext = new DynamicVariable[Option[NoArgTest]](None)

  protected def testConfig = scalaTestContext.value.get.configMap

  abstract override def withFixture(test: NoArgTest): Outcome = {
    require(scalaTestContext.value.isEmpty, "scalaTestContext already set")
    scalaTestContext.withValue(Some(test)) {
      super.withFixture(test)
    }
  }

  def test[T <: Module, B <: Backend](
    module:  => T,
    backend: B = verilator.Backend.initializeFromProcessEnvironment()
  ): TestBuilder[T, B] =
    new TestBuilder[T, B](() => module, backend, ChiselSimSettings(backend))

  def test[T <: Module, B <: Backend](
    module:   => T,
    settings: ChiselSimSettings[B]
  ): TestBuilder[T, B] =
    new TestBuilder[T, B](() => module, settings.backend, settings)
}
