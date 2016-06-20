// See LICENSE for license details.

package chisel3.iotesters

import chisel3.testers.BasicTester
import chisel3.{Bits, Module, printf}

import scala.util.Random

/**
  * provide common facilities for step based testing and decoupled interface testing
  */
abstract class HWIOTester extends BasicTester {
  val device_under_test:     Module
  var io_info:               IOAccessor = null
  def finish():              Unit

  val rnd                    = Random  // convenience for writing tests

  var enable_scala_debug     = false
  var enable_printf_debug    = false
  var enable_all_debug       = false

  def logScalaDebug(msg: => String): Unit = {
    //noinspection ScalaStyle
    if(enable_all_debug || enable_scala_debug) println(msg)
  }

  def logPrintfDebug(fmt: String, args: Bits*): Unit = {
    if(enable_all_debug || enable_printf_debug) printf(fmt, args :_*)
  }
}
