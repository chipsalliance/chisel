// SPDX-License-Identifier: Apache-2.0

package firrtlTests.transforms

import firrtl.annotations.ReferenceTarget
import firrtl.annotations.TargetToken.{Instance, OfModule}
import firrtl.testutils.FirrtlFlatSpec
import firrtl.stage.{FirrtlCircuitAnnotation, FirrtlStage}
import firrtl.transforms.{CustomRadixApplyAnnotation, CustomRadixDefAnnotation}
import firrtl.util.BackendCompilationUtilities

class CustomRadixTransformSpec extends FirrtlFlatSpec {
  behavior.of("CustomRadix")

  val testDir = os.Path(BackendCompilationUtilities.createTestDirectory("CustomRadix").getAbsolutePath)
  val fir =
    """circuit M2 :
      |  module PT :
      |    input clock : Clock
      |    input reset : Reset
      |    output io : { flip i : UInt<7>, o : UInt<7>}
      |
      |    io.o <= io.i
      |
      |  module PT_1 :
      |    input clock : Clock
      |    input reset : Reset
      |    output io : { flip i : UInt<7>, o : UInt<7>}
      |
      |    io.o <= io.i
      |
      |  module M1 :
      |    input clock : Clock
      |    input reset : Reset
      |    output out : UInt<7>
      |
      |    reg cnt : UInt<5>, clock with :
      |      reset => (reset, UInt<5>("h0"))
      |    node _cnt_T = add(cnt, UInt<1>("h1"))
      |    node _cnt_T_1 = tail(_cnt_T, 1)
      |    cnt <= _cnt_T_1
      |    inst pt of PT
      |    pt.clock <= clock
      |    pt.reset <= reset
      |    inst pt2 of PT_1
      |    pt2.clock <= clock
      |    pt2.reset <= reset
      |    pt2.io.i <= pt.io.o
      |    pt2.io.o is invalid
      |    pt.io.i <= UInt<1>("h0")
      |    node _T = eq(cnt, UInt<1>("h1"))
      |    when _T :
      |      pt.io.i <= UInt<1>("h1")
      |    else :
      |      node _T_1 = eq(cnt, UInt<2>("h2"))
      |      when _T_1 :
      |        pt.io.i <= UInt<2>("h2")
      |      else :
      |        node _T_2 = eq(cnt, UInt<2>("h3"))
      |        when _T_2 :
      |          pt.io.i <= UInt<7>("h64")
      |        else :
      |          node _T_3 = eq(cnt, UInt<3>("h4"))
      |          when _T_3 :
      |            pt.io.i <= UInt<7>("h65")
      |    out <= pt.io.o
      |
      |  module PT_2 :
      |    input clock : Clock
      |    input reset : Reset
      |    output io : { flip i : UInt<7>, o : UInt<7>}
      |
      |    io.o <= io.i
      |
      |  module M2 :
      |    input clock : Clock
      |    input reset : UInt<1>
      |    output out : UInt<7>
      |
      |    inst m1 of M1
      |    m1.clock <= clock
      |    m1.reset <= reset
      |    inst pt3 of PT_2
      |    pt3.clock <= clock
      |    pt3.reset <= reset
      |    pt3.io.i <= m1.out
      |    out <= pt3.io.o
      |""".stripMargin

  val annotations = Seq(
    FirrtlCircuitAnnotation(firrtl.Parser.parse(fir)),
    CustomRadixDefAnnotation("EnumExample", Seq(0, 1, 2, 100, 101).map(x => BigInt(x) -> s"e$x"), 7)
  ) ++ Seq(
    ("M1", Seq(), "in"),
    ("M2", Seq(Instance("pt3") -> OfModule("PT")), "io_o"),
    ("M2", Seq(Instance("m1") -> OfModule("M1"), Instance("pt2") -> OfModule("PT")), "io_i")
  ).map {
    case (module, path, ref) =>
      CustomRadixApplyAnnotation(ReferenceTarget("M2", module, path, ref, Seq()), "EnumExample")
  }

  it should "generate a JSON config file" in {
    (new FirrtlStage).execute(Array("--wave-viewer-script", "", "--target-dir", testDir.toString), annotations)
    val expected =
      """[{
        |  "EnumExample":{
        |    "width":7,
        |    "values":[{
        |      "digit":0,
        |      "alias":"e0"
        |    },{
        |      "digit":1,
        |      "alias":"e1"
        |    },{
        |      "digit":2,
        |      "alias":"e2"
        |    },{
        |      "digit":100,
        |      "alias":"e100"
        |    },{
        |      "digit":101,
        |      "alias":"e101"
        |    }],
        |    "signals":["M2.m1.pt.io_i","M2.m1.pt.io_o","M2.in"]
        |  }
        |}]""".stripMargin
    assert(expected == os.read(testDir / "custom_radix.json"))
  }
}
