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
import org.scalatest.prop.GeneratorDrivenPropertyChecks._

class BitwiseOps(w: Int) extends Module {
  val io = new Bundle {
    val a = Bits(INPUT, w)
    val b = Bits(INPUT, w)
    val not = Bits(OUTPUT, w)
    val and = Bits(OUTPUT, w)
    val or  = Bits(OUTPUT, w)
    val xor = Bits(OUTPUT, w)
  }
  io.not := ~io.a
  io.and := io.a & io.b
  io.or := io.a | io.b
  io.xor := io.a ^ io.b
}

class BitwiseOpsSpec extends ChiselSpec {

  class BitwiseOpsTester(w: Int, a: Int, b: Int) extends BasicTester {
    val mask = (1 << w)-1;
    val dut = Module(new BitwiseOps(w))
    io.done := Bool(true)
    dut.io.a := UInt(a) 
    dut.io.b := UInt(b)
    when(dut.io.not != UInt(mask & ~a)) { io.error := UInt(1) }
    when(dut.io.and != UInt(mask & (a & b))) { io.error := UInt(2) }
    when(dut.io.or  != UInt(mask & (a | b))) { io.error := UInt(3) }
    when(dut.io.xor != UInt(mask & (a ^ b))) { io.error := UInt(4) }
  }

  "BitwiseOps" should "return the correct result" in {
    forAll(safeUInts, safeUInts) { (a: Int, b: Int) =>
      assert(TesterDriver.execute{ new BitwiseOpsTester(32, a, b) }) 
    }
  }
}
