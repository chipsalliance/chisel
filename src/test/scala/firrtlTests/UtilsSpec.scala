package firrtlTests

import org.scalatest.FlatSpec
import org.scalatest.Matchers._

import firrtl.Utils

class UtilsSpec extends FlatSpec {

  behavior of "Utils.expandPrefix"

  val expandPrefixTests = List(
    ("return a name without prefixes", "_", "foo", Set("foo")),
    ("expand a name ending with prefixes", "_", "foo__", Set("foo__")),
    ("expand a name with on prefix", "_", "foo_bar", Set("foo_bar", "foo_")),
    ("expand a name with complex prefixes", "_",
     "foo__$ba9_9X__$$$$$_", Set("foo__$ba9_9X__$$$$$_", "foo__$ba9_9X__", "foo__$ba9_", "foo__")),
    ("expand a name starting with a delimiter", "_", "__foo_bar", Set("__", "__foo_", "__foo_bar")),
    ("expand a name with a $ delimiter", "$", "foo$bar$$$baz", Set("foo$", "foo$bar$$$", "foo$bar$$$baz")),
    ("expand a name with a multi-character delimiter", "FOO", "fooFOOFOOFOObar", Set("fooFOOFOOFOO", "fooFOOFOOFOObar"))
  )

  for ((description, delimiter, in, out) <- expandPrefixTests) {
    it should description in { Utils.expandPrefixes(in, delimiter).toSet should be (out)}
  }
}
