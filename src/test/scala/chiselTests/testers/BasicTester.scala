// SPDX-License-Identifier: Apache-2.0

package chiselTests.testers

import chisel3._
import chisel3.internal.sourceinfo.SourceInfo

import scala.language.experimental.macros

class BasicTester extends Module() {
  // The testbench has no IOs, rather it should communicate using printf, assert, and stop.
  val io = IO(new Bundle() {})

  def popCount(n: Long): Int = n.toBinaryString.count(_=='1')

  def stop()(implicit sourceInfo: SourceInfo) {
    // TODO: rewrite this using library-style SourceInfo passing.
    chisel3.stop()
  }

  /** The finish method provides a hook that subclasses of BasicTester can use to
    * alter a circuit after their constructor has been called.
    * For example, a specialized tester subclassing BasicTester could override finish in order to
    * add flow control logic for a decoupled io port of a device under test
    */
  def finish(): Unit = {}
}
