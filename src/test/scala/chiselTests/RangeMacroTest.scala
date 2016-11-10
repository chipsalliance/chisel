// See LICENSE for license details.

package chiselTests

import chisel3._
import scala.language.experimental.macros
import org.scalatest._
import org.scalatest.prop._
import chisel3.testers.BasicTester


/** Comprehensive test of static range parsing functionality.
  * Note: negative (failure) conditions can't be tested because they will fail at compile time,
  * before the testing environment is entered.
  */
class RangeMacroSpec extends ChiselPropSpec {
  property("Range macros should work") {
    range"(0,${1+1}]"
    range"  (  0  ,  ${1+1}  ]  "
  }
}
