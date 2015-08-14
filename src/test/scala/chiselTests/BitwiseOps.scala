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
import org.scalatest._
import org.scalatest.prop._
import Chisel.testers.BasicTester

class BitwiseOpsSpec extends ChiselPropSpec {

  class BitwiseOpsTester(w: Int, _a: Int, _b: Int) extends BasicTester {
    io.done := Bool(true)
    val mask = (1 << w) - 1
    val a = UInt(_a) 
    val b = UInt(_b)
    when(~a != UInt(mask & ~_a)) { io.error := UInt(1) }
    when((a & b) != UInt(mask & (_a & _b))) { io.error := UInt(2) }
    when((a | b)  != UInt(mask & (_a | _b))) { io.error := UInt(3) }
    when((a ^ b) != UInt(mask & (_a ^ _b))) { io.error := UInt(4) }
  }

  property("All bit-wise ops should return the correct result") {
    forAll(safeUIntPair) { case(w: Int, a: Int, b: Int) =>
      assert(execute{ new BitwiseOpsTester(w, a, b) }) 
    }
  }
}
