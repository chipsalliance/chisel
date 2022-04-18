// SPDX-License-Identifier: Apache-2.0

package chiselTests.util.experimental

import chisel3.util.experimental.decode.{DecodeTableAnnotation, Minimizer, QMCMinimizer, TruthTable}
import chiselTests.util.experimental.minimizer.DecodeTestModule
import firrtl.annotations.ReferenceTarget
import org.scalatest.flatspec.AnyFlatSpec
import chiseltest._
import chiseltest.formal._

class DecoderSpec extends AnyFlatSpec with ChiselScalatestTester with Formal  {
  val xor = TruthTable.fromString(
    """10->1
      |01->1
      |    0""".stripMargin)

  def minimizer: Minimizer = QMCMinimizer

  "decoder" should "pass without DecodeTableAnnotation" in {
    verify(new DecodeTestModule(minimizer, table = xor), Seq(BoundedCheck(1)))
  }

  "decoder" should "fail with a incorrect DecodeTableAnnotation" in {
    val annos = Seq(
      DecodeTableAnnotation(ReferenceTarget("", "", Nil, "", Nil),
        """10->1
          |01->1
          |    0""".stripMargin,
        """10->1
          |    0""".stripMargin
      )
    )
    assertThrows[FailedBoundedCheckException] {
      verify(new DecodeTestModule(minimizer, table = xor), BoundedCheck(1) +: annos)
    }
  }

  "decoder" should "success with a correct DecodeTableAnnotation" in {
    val annos = Seq(
      DecodeTableAnnotation(ReferenceTarget("", "", Nil, "", Nil),
        """10->1
          |01->1
          |    0""".stripMargin,
        QMCMinimizer.minimize(TruthTable.fromString(
          """10->1
            |01->1
            |    0""".stripMargin)).toString
      )
    )
    verify(new DecodeTestModule(minimizer, table = xor), BoundedCheck(1) +: annos)
  }
}
