// SPDX-License-Identifier: Apache-2.0

package chiselTests.util.experimental

import chisel3.util.experimental.decode.{DecodeTableAnnotation, Minimizer, QMCMinimizer, TruthTable}
import chiselTests.SMTModelCheckingSpec
import chiselTests.util.experimental.minimizer.DecodeTestModule
import firrtl.annotations.ReferenceTarget

class DecoderSpec extends SMTModelCheckingSpec {
  val xor = TruthTable(
    """10->1
      |01->1
      |    0""".stripMargin)

  def minimizer: Minimizer = QMCMinimizer

  "decoder" should "pass without DecodeTableAnnotation" in {
    test(
      () => new DecodeTestModule(minimizer, table = xor),
      s"${minimizer.getClass.getSimpleName}.noAnno",
      success
    )
  }

  "decoder" should "fail with a incorrect DecodeTableAnnotation" in {
    test(
      () => new DecodeTestModule(minimizer, table = xor),
      s"${minimizer.getClass.getSimpleName}.incorrectAnno",
      fail(0),
      annos = Seq(
        DecodeTableAnnotation(ReferenceTarget("", "", Nil, "", Nil),
          """10->1
            |01->1
            |    0""".stripMargin,
          """10->1
            |    0""".stripMargin
        )
      )
    )
  }

  "decoder" should "success with a correct DecodeTableAnnotation" in {
    test(
      () => new DecodeTestModule(minimizer, table = xor),
      s"${minimizer.getClass.getSimpleName}.correctAnno",
      success,
      annos = Seq(
        DecodeTableAnnotation(ReferenceTarget("", "", Nil, "", Nil),
          """10->1
            |01->1
            |    0""".stripMargin,
          QMCMinimizer.minimize(TruthTable(
            """10->1
              |01->1
              |    0""".stripMargin)).toString
        )
      )
    )
  }
}
