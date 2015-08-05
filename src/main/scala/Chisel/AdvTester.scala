/*
 Copyright (c) 2011, 2012, 2013, 2014 The Regents of the University of
 California (Regents). All Rights Reserved.  Redistribution and use in
 source and binary forms, with or without modification, are permitted
 provided that the following conditions are met:

    * Redistributions of source code must retain the above
      copyright notice, this list of conditions and the following
      two paragraphs of disclaimer.
    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      two paragraphs of disclaimer in the documentation and/or other materials
      provided with the distribution.
    * Neither the name of the Regents nor the names of its contributors
      may be used to endorse or promote products derived from this
      software without specific prior written permission.

 IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT,
 SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS,
 ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF
 REGENTS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
 LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF
 ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION
 TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR
 MODIFICATIONS.
*/

/*
  Written by Stephen Twigg, Eric Love
  Version 0.9
*/
package Chisel.AdvTester // May eventually add this to the base Chisel package
import Chisel._
import scala.collection.mutable.ArrayBuffer

class AdvTester[+T <: Module](val dut: T, isTrace: Boolean = false) extends Tester[T](dut, isTrace) {
  val defaultMaxCycles = 1024
  var cycles = 0
  var pass = true

  // List of scala objects that need to be processed along with the test benches, like sinks and sources
  val preprocessors = new ArrayBuffer[Processable]()
  val postprocessors = new ArrayBuffer[Processable]()
  // pre v post refers to when user-customized update code ('work') is processed
  // e.g. sinks are in the preprocessing list and sources in the postprocessing list
  //    this allows the testbench to respond to a request within one cycle

  // This section of code lets testers easily emulate have registers right before dut inputs
  //   This testing style conforms with the general ASPIRE testbench style
  // Also, to ensure difference enforced, poke 'deprecated' and replaced with wire_poke
  def wire_poke(port: Bits,      target: Boolean)       = { super.poke(port, int(target)) }
  def wire_poke(port: Bits,      target: Int)           = { super.poke(port, int(target)) }
  def wire_poke(port: Bits,      target: Long)          = { super.poke(port, int(target)) }
  def wire_poke(port: Bits,      target: BigInt)        = { super.poke(port, target) }
  def wire_poke(port: Aggregate, target: Array[BigInt]) = { super.poke(port, target) }

  override def poke(port: Bits, target: BigInt) = require(false, "poke hidden for AdvTester, use wire_poke or reg_poke")
  override def poke(port: Aggregate, target: Array[BigInt]) = require(false, "poke hidden for AdvTester, use wire_poke or reg_poke")

  val registered_bits_updates = new scala.collection.mutable.HashMap[Bits,BigInt]()
  val registered_aggr_updates = new scala.collection.mutable.HashMap[Aggregate,Array[BigInt]]()

  def do_registered_updates() = {
    registered_bits_updates.foreach( kv => wire_poke(kv._1,kv._2) )
    registered_aggr_updates.foreach( kv => wire_poke(kv._1,kv._2) )

    registered_bits_updates.clear()
    registered_aggr_updates.clear()
  }

  def reg_poke(port: Bits,      target: BigInt)        = { registered_bits_updates(port) = target }
  def reg_poke(port: Aggregate, target: Array[BigInt]) = { registered_aggr_updates(port) = target }

  // Convenience functions
  def Boolean2Int(i: Boolean): Int = (if(i) 1 else 0) // TODO: Investigate name and inclusion as a Chisel Tester auto-convert

  // This function replaces step in the advanced tester and makes sure all tester features are clocked in the appropriate order
  def takestep(work: => Unit = {}): Unit = {
    cycles += 1
    step(1)
    do_registered_updates()
    preprocessors.foreach(_.process()) // e.g. sinks
    work
    postprocessors.foreach(_.process())
  }
  def takesteps(n: Int)(work: =>Unit = {}): Unit = {
    require(n>0, "Number of steps taken must be positive integer.")
    (0 until n).foreach(_ => takestep(work))
  }

  // Functions to step depending on predicates
  def until(pred: =>Boolean, maxCycles: Int = defaultMaxCycles)(work: =>Unit): Boolean = {
    var timeout_cycles = 0
    while(!pred && (timeout_cycles < maxCycles)) {
      takestep(work)
      timeout_cycles += 1
    }
    assert(timeout_cycles < maxCycles,
      "until timed out after %d cycles".format(timeout_cycles))
    pred
  }
  def eventually(pred: =>Boolean, maxCycles: Int = defaultMaxCycles) = {until(pred, maxCycles){}}
  def do_until(work: =>Unit)(pred: =>Boolean, maxCycles: Int = defaultMaxCycles): Boolean = {
    takestep(work)
    until(pred, maxCycles){work}
  }

  def assert(expr: Boolean, errMsg:String = "") = {
    pass &= expr
    if(!expr && errMsg != "") { println("ASSERT FAILED: " + errMsg) }
    expr
  }


  class DecoupledSink[T <: Data, R]( socket: DecoupledIO[T], cvt: T=>R ) extends Processable
  {
    var max_count = -1
    val outputs = new scala.collection.mutable.Queue[R]()
    private var amReady = false
    private def isValid = () => (peek(socket.valid) == 1)

    def process() = {
      // Handle this cycle
      if(isValid() && amReady) {
        outputs.enqueue(cvt(socket.bits))
      }
      // Decide what to do next cycle and post onto register
      amReady = max_count < 1 || outputs.length < max_count
      reg_poke(socket.ready, Boolean2Int(amReady))
    }

    // Initialize
    wire_poke(socket.ready, 0)
    preprocessors += this
  }
  object DecoupledSink {
    def apply[T<:Bits](socket: DecoupledIO[T]) = new DecoupledSink(socket, (socket_bits: T) => peek(socket_bits))
  }

  class ValidSink[T <: Data, R]( socket: ValidIO[T], cvt: T=>R ) extends Processable
  {
    val outputs = new scala.collection.mutable.Queue[R]()
    private def isValid = peek(socket.valid) == 1

    def process() = {
      if(isValid) {
        outputs.enqueue(cvt(socket.bits))
      }
    }

    // Initialize
    preprocessors += this
  }
  object ValidSink {
    def apply[T<:Bits](socket: ValidIO[T]) = new ValidSink(socket, (socket_bits: T) => peek(socket_bits))
  }

  class DecoupledSource[T <: Data, R]( socket: DecoupledIO[T], post: (T,R)=>Unit ) extends Processable
  {
    val inputs = new scala.collection.mutable.Queue[R]()

    private var amValid = false
    private var justFired = false
    private def isReady = (peek(socket.ready) == 1)
    def isIdle = !amValid && inputs.isEmpty && !justFired

    def process() = {
      justFired = false
      if(isReady && amValid) { // Fired last cycle
        amValid = false
        justFired = true
      }
      if(!amValid && !inputs.isEmpty) {
        amValid = true
        post(socket.bits, inputs.dequeue())
      }
      reg_poke(socket.valid, Boolean2Int(amValid))
    }

    // Initialize
    wire_poke(socket.valid, 0)
    postprocessors += this
  }
  object DecoupledSource {
    def apply[T<:Bits](socket: DecoupledIO[T]) = new DecoupledSource(socket, (socket_bits: T, in: BigInt) => reg_poke(socket_bits, in))
  }

  class ValidSource[T <: Data, R]( socket: ValidIO[T], post: (T,R)=>Unit ) extends Processable
  {
    val inputs = new scala.collection.mutable.Queue[R]()
    private var amValid = false
    private var justFired = false

    def isIdle = inputs.isEmpty && !amValid

    def process() = {
      // Always advance the input
      justFired = (amValid==true)
      amValid = false
      if(!inputs.isEmpty) {
        amValid = true
        post(socket.bits, inputs.dequeue())
      }
      reg_poke(socket.valid, Boolean2Int(amValid))
    }

    // Initialize
    wire_poke(socket.valid, 0)
    postprocessors += this
  }
  object ValidSource {
    def apply[T<:Bits](socket: ValidIO[T]) = new ValidSource(socket, (socket_bits: T, in: BigInt) => reg_poke(socket_bits, in))
  }

}

trait Processable {
  def process(): Unit
}

