/*
 Copyright (c) 2011-2015 The Regents of the University of
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

package chiselTests

import Chisel._
import Chisel.testers._
import org.scalatest._
import org.scalatest.prop._
import org.scalatest.prop.TableDrivenPropertyChecks._

class GCD extends Module {
  val io = new Bundle {
    val a  = Bits(INPUT,  16)
    val b  = Bits(INPUT,  16)
    val e  = Bool(INPUT)
    val z  = Bits(OUTPUT, 16)
    val v  = Bool(OUTPUT)
  }
  val x = Reg(Bits(width = 16))
  val y = Reg(Bits(width = 16))
  when (x > y)   { x := x -% y }
  .otherwise     { y := y -% x }
  when (io.e) { x := io.a; y := io.b }
  io.z := x
  io.v := y === Bits(0)
}

class GCDSpec extends ChiselSpec {

  class GCDTester(a: Int, b: Int, z: Int) extends BasicTester {
    val dut = Module(new GCD)
    val first = Reg(init=Bool(true))
    dut.io.a := UInt(a)
    dut.io.b := UInt(b)
    dut.io.e := first
    when(first) { first := Bool(false) }
    when(dut.io.v) {
      io.done := Bool(true)
      io.error := (dut.io.z != UInt(z)).toUInt
    }
  }
  
  val gcds = Table(
    ("a", "b", "z"),  // First tuple defines column names
    ( 64,  48,  16),  // Subsequent tuples define the data
    ( 12,   9,   3),
    ( 48,  64,  12))

  "GCD" should "return the correct result" in {
    forAll (gcds) { (a: Int, b: Int, z: Int) => 
      assert(TesterDriver.execute{ new GCDTester(a, b, z) })
    }
  }
}
