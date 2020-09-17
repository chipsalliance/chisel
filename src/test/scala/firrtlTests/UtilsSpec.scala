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
}
