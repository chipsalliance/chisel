// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.testers.BasicTester
import chisel3.experimental.RawModule

class DataPrintSpec extends ChiselFlatSpec {
  "Data types" should "have a meaningful string representation" in {
    elaborate {
      new RawModule {
        println(UInt(8.W))
      }
    }
  }
}
