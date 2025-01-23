// SPDX-License-Identifier: Apache-2.0

package chisel3.internal

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers._

class NamespaceSpec extends AnyFlatSpec {
  behavior.of("Namespace")

  they should "support basic disambiguation" in {
    val namespace = Namespace.empty
    val name = namespace.name(_, false)
    name("x") should be("x")
    name("x") should be("x_1")
    name("x") should be("x_2")
  }

  they should "support explicit <prefix>_# names before <prefix> names" in {
    val namespace = Namespace.empty
    val name = namespace.name(_, false)
    name("x_1") should be("x_1")
    name("x_2") should be("x_2")
    name("x") should be("x")
    name("x") should be("x_3")
  }

  they should "support explicit <prefix>_# names in the middle of <prefix> names" in {
    val namespace = Namespace.empty
    val name = namespace.name(_, false)
    name("x") should be("x")
    name("x") should be("x_1")
    name("x_1") should be("x_1_1")
    name("x_2") should be("x_2")
    name("x") should be("x_3")
  }

  // For some reason, multi-character names tickled a different failure mode than single character
  they should "support explicit <prefix>_# names in the middle of longer <prefix> names" in {
    val namespace = Namespace.empty
    val name = namespace.name(_, false)
    name("foo") should be("foo")
    name("foo") should be("foo_1")
    name("foo_1") should be("foo_1_1")
    name("foo_2") should be("foo_2")
    name("foo") should be("foo_3")
  }

  they should "support collisions in recursively growing names" in {
    val namespace = Namespace.empty
    val name = namespace.name(_, false)
    name("x") should be("x")
    name("x") should be("x_1")
    name("x_1") should be("x_1_1")
    name("x_1") should be("x_1_2")
    name("x_1_1") should be("x_1_1_1")
    name("x_1_1") should be("x_1_1_2")
  }

  they should "support collisions in recursively shrinking names" in {
    val namespace = Namespace.empty
    val name = namespace.name(_, false)
    name("x_1_1") should be("x_1_1")
    name("x_1_1") should be("x_1_1_1")
    name("x_1") should be("x_1")
    name("x_1") should be("x_1_2")
    name("x") should be("x")
    name("x") should be("x_2")
  }

  // The namespace never generates names with _0 so it's actually a false collision case
  they should "properly handle false collisions with signals ending in _0" in {
    val namespace = Namespace.empty
    val name = namespace.name(_, false)
    name("x") should be("x")
    name("x") should be("x_1")
    name("x_0") should be("x_0")
    name("x") should be("x_2")
    name("x_0") should be("x_0_1")
  }

  they should "support resolving collisions between <name> and <name>_" in {
    val namespace = Namespace.empty
    namespace.name("mouth") should be("mouth")
    namespace.name("mouth_") should be("mouth_")
    namespace.name("mouth") should be("mouth_1")
    namespace.name("mouth_") should be("mouth__1")
  }
}
