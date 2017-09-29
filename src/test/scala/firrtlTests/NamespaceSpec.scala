// See LICENSE for license details.

package firrtlTests

import firrtl.Namespace

class NamespaceSpec extends FirrtlFlatSpec {

  "A Namespace" should "not allow collisions" in {
    val namespace = Namespace()
    namespace.newName("foo") should be ("foo")
    namespace.newName("foo") should be ("foo_0")
  }

  it should "start temps with a suffix of 0" in {
    Namespace().newTemp.last should be ('0')
  }

  it should "handle multiple prefixes with independent suffixes" in {
    val namespace = Namespace()
    namespace.newName("foo") should be ("foo")
    namespace.newName("foo") should be ("foo_0")
    namespace.newName("bar") should be ("bar")
    namespace.newName("bar") should be ("bar_0")
  }
}
