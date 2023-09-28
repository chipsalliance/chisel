// SPDX-License-Identifier: Apache-2.0

package firrtlTests

import firrtl.ir._
import firrtl.testutils._

class ExtModuleTests extends FirrtlFlatSpec {
  "extmodule" should "serialize and re-parse equivalently" in {
    val input =
      """|FIRRTL version 3.3.0
         |circuit Top :
         |  extmodule Top :
         |    input y : UInt<0>
         |    output x : UInt<1>
         |    defname = ParameterizedExtModule
         |    parameter VALUE = 1
         |    parameter VALUE2 = '2\'d2'
         |    parameter STRING = "one"
         |    parameter REAL = -1.7
         |    parameter TYP = 'bit'
         |""".stripMargin

    val circuit = Circuit(
      NoInfo,
      Seq(
        ExtModule(
          NoInfo,
          "Top",
          Seq(Port(NoInfo, "y", Input, UIntType(IntWidth(0))), Port(NoInfo, "x", Output, UIntType(IntWidth(1)))),
          "ParameterizedExtModule",
          Seq(
            IntParam("VALUE", 1),
            RawStringParam("VALUE2", "2'd2"),
            StringParam("STRING", StringLit("one")),
            DoubleParam("REAL", -1.7),
            RawStringParam("TYP", "bit")
          )
        )
      ),
      "Top"
    )

    circuit.serialize should be(input)
  }
}
