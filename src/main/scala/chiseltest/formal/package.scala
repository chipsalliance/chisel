// SPDX-License-Identifier: Apache-2.0

package chiseltest

import chisel3._

/**
 * Formal verification stubs for compatibility
 * 
 * Note: Formal verification is not fully supported in Chisel 7 yet.
 * These are compatibility stubs that allow code to compile but may not provide
 * full formal verification functionality.
 */
package object formal {
  
  // Annotation stubs
  case class BoundedCheck(depth: Int)
  
  // Formal verification trait
  trait Formal {
    def verify[T <: Module](dut: => T, annotations: Seq[Any]): Unit = {
      println("Warning: Formal verification is not fully supported in Chisel 7")
      // Stub implementation - just simulate for now
      import chisel3.simulator.EphemeralSimulator._
      import chiseltest._ // Import implicit conversions
      
      simulate(dut) { c =>
        // Run for bounded depth if BoundedCheck is present
        val depth = annotations.collectFirst {
          case BoundedCheck(d) => d
        }.getOrElse(10)
        
        // Use the implicit conversion
        chiseltest.testableClock(c.clock).step(depth)
      }
    }
  }
  
  // Past function stub
  def past[T <: Data](x: T, delay: Int = 1): T = {
    // This is a stub - actual formal verification would use SMT solver
    x
  }
  
  // Overload for single argument
  def past[T <: Data](x: T): T = past(x, 1)
}
