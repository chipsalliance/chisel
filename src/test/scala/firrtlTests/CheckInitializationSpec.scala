// SPDX-License-Identifier: Apache-2.0

package firrtlTests

import firrtl.{CircuitState, Transform, UnknownForm}
import firrtl.passes._
import firrtl.testutils._

class CheckInitializationSpec extends FirrtlFlatSpec {
  private val passes = Seq(
    ToWorkingIR,
    CheckHighForm,
    ResolveKinds,
    InferTypes,
    CheckTypes,
    ResolveFlows,
    CheckFlows,
    new InferWidths,
    CheckWidths,
    PullMuxes,
    ExpandConnects,
    RemoveAccesses,
    ExpandWhens,
    CheckInitialization,
    InferTypes
  )
  "Missing assignment in consequence branch" should "trigger a PassException" in {
    val input =
      """circuit Test :
        |  module Test :
        |    input p : UInt<1>
        |    wire x : UInt<32>
        |    when p :
        |      x <= UInt(1)""".stripMargin
    intercept[CheckInitialization.RefNotInitializedException] {
      passes.foldLeft(CircuitState(parse(input), UnknownForm)) { (c: CircuitState, p: Transform) =>
        p.runTransform(c)
      }
    }
  }
  "Missing assignment in alternative branch" should "trigger a PassException" in {
    val input =
      """circuit Test :
        |  module Test :
        |    input p : UInt<1>
        |    wire x : UInt<32>
        |    when p :
        |    else :
        |      x <= UInt(1)""".stripMargin
    intercept[CheckInitialization.RefNotInitializedException] {
      passes.foldLeft(CircuitState(parse(input), UnknownForm)) { (c: CircuitState, p: Transform) =>
        p.runTransform(c)
      }
    }
  }

  "Assign after incomplete assignment" should "work" in {
    val input =
      """circuit Test :
        |  module Test :
        |    input p : UInt<1>
        |    wire x : UInt<32>
        |    when p :
        |      x <= UInt(1)
        |    x <= UInt(1)
        |    """.stripMargin
    passes.foldLeft(CircuitState(parse(input), UnknownForm)) { (c: CircuitState, p: Transform) =>
      p.runTransform(c)
    }
  }

  "Assign after nested incomplete assignment" should "work" in {
    val input =
      """circuit Test :
        |  module Test :
        |    input p : UInt<1>
        |    input q : UInt<1>
        |    wire x : UInt<32>
        |    when p :
        |      when q :
        |        x is invalid
        |    else :
        |      when q :
        |        x <= UInt(2)
        |    x <= UInt(1)
        |    """.stripMargin
    passes.foldLeft(CircuitState(parse(input), UnknownForm)) { (c: CircuitState, p: Transform) =>
      p.runTransform(c)
    }
  }

  "Missing assignment to submodule port" should "trigger a PassException" in {
    val input =
      """circuit Test :
        |  module Child :
        |    input in : UInt<32>
        |  module Test :
        |    input p : UInt<1>
        |    inst c of Child
        |    when p :
        |      c.in <= UInt(1)""".stripMargin
    intercept[CheckInitialization.RefNotInitializedException] {
      passes.foldLeft(CircuitState(parse(input), UnknownForm)) { (c: CircuitState, p: Transform) =>
        p.runTransform(c)
      }
    }
  }
}
