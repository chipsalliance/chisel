// SPDX-License-Identifier: Apache-2.0

package firrtlTests

import firrtl._
import firrtl.ir._
import firrtl.Utils
import org.scalatest.matchers.should.Matchers._
import org.scalatest.flatspec.AnyFlatSpec

class UtilsSpec extends AnyFlatSpec {

  behavior.of("Utils.expandPrefix")

  val expandPrefixTests = List(
    ("return a name without prefixes", "_", "foo", Set("foo")),
    ("expand a name ending with prefixes", "_", "foo__", Set("foo__")),
    ("expand a name with on prefix", "_", "foo_bar", Set("foo_bar", "foo_")),
    (
      "expand a name with complex prefixes",
      "_",
      "foo__$ba9_9X__$$$$$_",
      Set("foo__$ba9_9X__$$$$$_", "foo__$ba9_9X__", "foo__$ba9_", "foo__")
    ),
    ("expand a name starting with a delimiter", "_", "__foo_bar", Set("__", "__foo_", "__foo_bar")),
    ("expand a name with a $ delimiter", "$", "foo$bar$$$baz", Set("foo$", "foo$bar$$$", "foo$bar$$$baz")),
    ("expand a name with a multi-character delimiter", "FOO", "fooFOOFOOFOObar", Set("fooFOOFOOFOO", "fooFOOFOOFOObar"))
  )

  for ((description, delimiter, in, out) <- expandPrefixTests) {
    it should description in { Utils.expandPrefixes(in, delimiter).toSet should be(out) }
  }

  "expandRef" should "return intermediate expressions" in {
    val bTpe = VectorType(Utils.BoolType, 2)
    val topTpe = BundleType(Seq(Field("a", Default, Utils.BoolType), Field("b", Default, bTpe)))
    val wr = WRef("out", topTpe, PortKind, SourceFlow)

    val expected = Seq(
      wr,
      WSubField(wr, "a", Utils.BoolType, SourceFlow),
      WSubField(wr, "b", bTpe, SourceFlow),
      WSubIndex(WSubField(wr, "b", bTpe, SourceFlow), 0, Utils.BoolType, SourceFlow),
      WSubIndex(WSubField(wr, "b", bTpe, SourceFlow), 1, Utils.BoolType, SourceFlow)
    )

    (Utils.expandRef(wr)) should be(expected)
  }

  def combineTest(circuits: Seq[String], expected: String) = {
    (Utils.orderAgnosticEquality(Utils.combine(circuits.map(c => Parser.parse(c))), Parser.parse(expected))) should be(
      true
    )
  }

  "combine" should "merge multiple module circuits" in {
    val input = Seq(
      """|circuit Top:
         |  extmodule External:
         |    output foo: UInt<32>
         |    defname = External
         |
         |  extmodule Child1:
         |    output foo: UInt<32>
         |    defname = Child1
         |
         |  extmodule Child2:
         |    output foo: UInt<32>
         |    defname = Child2
         |
         |  module Top:
         |    output foo: UInt<32>
         |    inst c1 of Child1
         |    inst c2 of Child2
         |    inst e of External
         |    foo <= tail(add(add(c1.foo, c2.bar), e.foo), 1)
         |""".stripMargin,
      """|circuit Child1:
         |  extmodule External:
         |    output foo: UInt<32>
         |    defname = External
         |
         |  module Child1:
         |    output foo: UInt<32>
         |    inst e of External
         |    foo <= e.foo
         |""".stripMargin,
      """|circuit Child2:
         |  extmodule External:
         |    output foo: UInt<32>
         |    defname = External
         |
         |  module Child2:
         |    output bar: UInt<32>
         |    inst e of External
         |    bar <= e.foo
         |""".stripMargin
    )

    val output =
      """|circuit Top:
         |  module Top:
         |    output foo: UInt<32>
         |    inst c1 of Child1
         |    inst c2 of Child2
         |    inst e of External
         |    foo <= tail(add(add(c1.foo, c2.bar), e.foo), 1)
         |
         |  module Child1:
         |    output foo: UInt<32>
         |    inst e of External
         |    foo <= e.foo
         |
         |  module Child2:
         |    output bar: UInt<32>
         |    inst e of External
         |    bar <= e.foo
         |
         |  extmodule External:
         |    output foo: UInt<32>
         |    defname = External
         |""".stripMargin

    combineTest(input, output)
  }

  "combine" should "dedup ExtModules if an implementation exists" in {
    val input = Seq(
      """|circuit Top:
         |  extmodule Child:
         |    output foo: UInt<32>
         |    defname = Child
         |
         |  module Top:
         |    output foo: UInt<32>
         |    inst c of Child
         |    foo <= c.foo
         |""".stripMargin,
      """|circuit Child:
         |  module Child:
         |    output foo: UInt<32>
         |
         |    skip
         |""".stripMargin
    )

    val output =
      """|circuit Top:
         |  module Top:
         |    output foo: UInt<32>
         |    inst c of Child
         |    foo <= c.foo
         |
         |  module Child:
         |    output foo: UInt<32>
         |
         |    skip
         |""".stripMargin

    combineTest(input, output)
  }

  "combine" should "support lone ExtModules" in {
    val input = Seq(
      """|circuit Top:
         |  extmodule External:
         |    output foo: UInt<32>
         |    defname = External
         |
         |  module Top:
         |    output foo: UInt<32>
         |    inst e of External
         |    foo <= e.foo
         |""".stripMargin
    )

    val output =
      """|circuit Top:
         |  extmodule External:
         |    output foo: UInt<32>
         |    defname = External
         |
         |  module Top:
         |    output foo: UInt<32>
         |    inst e of External
         |    foo <= e.foo
         |""".stripMargin

    combineTest(input, output)
  }

  "combine" should "fail with multiple lone Modules" in {
    val input = Seq(
      """|circuit Top:
         |  extmodule External:
         |    output foo: UInt<32>
         |    defname = External
         |
         |  module Top:
         |    output foo: UInt<32>
         |    inst e of External
         |    foo <= e.foo
         |""".stripMargin,
      """|circuit Top2:
         |  extmodule External:
         |    output foo: UInt<32>
         |    defname = External
         |
         |  module Top2:
         |    output bar: UInt<32>
         |    inst e of External
         |    bar <= e.foo
         |""".stripMargin
    )

    a[java.lang.AssertionError] shouldBe thrownBy { combineTest(input, "") }
  }

}
