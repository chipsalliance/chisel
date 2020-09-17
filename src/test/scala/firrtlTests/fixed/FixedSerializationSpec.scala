// SPDX-License-Identifier: Apache-2.0

package firrtlTests
package fixed

import firrtl.ir
import org.scalatest.flatspec.AnyFlatSpec

class FixedSerializationSpec extends AnyFlatSpec {
  behavior.of("FixedType")

  it should "serialize correctly" in {
    assert(ir.FixedType(ir.IntWidth(3), ir.IntWidth(2)).serialize == "Fixed<3><<2>>")
    assert(ir.FixedType(ir.IntWidth(10), ir.UnknownWidth).serialize == "Fixed<10>")
    assert(ir.FixedType(ir.UnknownWidth, ir.IntWidth(-4)).serialize == "Fixed<<-4>>")
    assert(ir.FixedType(ir.UnknownWidth, ir.UnknownWidth).serialize == "Fixed")
  }

  behavior.of("FixedLiteral")

  it should "serialize correctly" in {
    assert(ir.FixedLiteral(1, ir.IntWidth(3), ir.IntWidth(2)).serialize == "Fixed<3><<2>>(\"h1\")")
    assert(ir.FixedLiteral(1, ir.IntWidth(10), ir.UnknownWidth).serialize == "Fixed<10>(\"h1\")")
    assert(ir.FixedLiteral(1, ir.UnknownWidth, ir.IntWidth(-4)).serialize == "Fixed<<-4>>(\"h1\")")
    assert(ir.FixedLiteral(1, ir.UnknownWidth, ir.UnknownWidth).serialize == "Fixed(\"h1\")")
  }
}
