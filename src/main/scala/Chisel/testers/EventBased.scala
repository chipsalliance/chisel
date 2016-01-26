// See LICENSE for license details.

package Chisel.testers

import Chisel.{printf, Bits, Module}

import scala.util.Random

/**
  * provide common facilities for step based testing and decoupled interface testing
  */
trait EventBased {
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
    if(enable_all_debug || enable_scala_debug) printf(fmt, args :_*)
  }

  def testBlock(block: => Unit): Unit = {
    block
    finish()
  }


}
