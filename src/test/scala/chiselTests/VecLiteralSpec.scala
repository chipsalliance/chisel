// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.util.Cat
import chisel3.experimental.BundleLiterals.AddBundleLiteralConstructor
import chisel3.experimental.VecLiterals._
import chisel3.experimental.VecLiteralException
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.stimulus.RunUntilFinished
import chisel3.util.Counter
import circt.stage.ChiselStage
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.should.Matchers

import scala.language.reflectiveCalls

class VecLiteralSpec extends AnyFreeSpec with Matchers with ChiselSim {
  object MyEnum extends ChiselEnum {
    val sA, sB, sC = Value
  }
  object MyEnumB extends ChiselEnum {
    val sA, sB = Value
  }

  "Vec literals should work with chisel Enums" in {
    val enumVec = Vec(3, MyEnum()).Lit(0 -> MyEnum.sA, 1 -> MyEnum.sB, 2 -> MyEnum.sC)
    enumVec(0).toString should include(MyEnum.sA.toString)
    enumVec(1).toString should include(MyEnum.sB.toString)
    enumVec(2).toString should include(MyEnum.sC.toString)
  }

  "improperly constructed vec literals should be detected" - {
    "indices in vec literal muse be greater than zero and less than length" in {
      val e = intercept[VecLiteralException] {
        Vec(2, UInt(4.W)).Lit(0 -> 1.U, 1 -> 2.U, 2 -> 3.U, 3 -> 4.U, -2 -> 7.U)
      }
      e.getMessage should include(
        "VecLiteral: The following indices (2,3,-2) are less than zero or greater or equal to than Vec length"
      )
    }

    "indices in vec literals must not be repeated" in {
      val e = intercept[VecLiteralException] {
        Vec(2, UInt(4.W)).Lit(0 -> 1.U, 1 -> 2.U, 2 -> 3.U, 2 -> 3.U, 2 -> 3.U, 3 -> 4.U)
      }
      e.getMessage should include("VecLiteral: has duplicated indices 2(3 times)")
    }
    "lits must fit in vec element width" in {
      val e = intercept[VecLiteralException] {
        Vec(2, SInt(4.W)).Lit(0 -> 0xab.S, 1 -> 0xbc.S)
      }
      e.getMessage should include(
        "VecLiteral: Vec[SInt<4>] has the following incorrectly typed or sized initializers: " +
          "0 -> SInt<9>(171),1 -> SInt<9>(188)"
      )
    }

    "all lits must be the same type but width can be equal or smaller than the Vec's element width" in {
      val v = Vec(2, SInt(4.W)).Lit(0 -> 1.S, 1 -> -2.S)
      v(0).toString should include(1.S(4.W).toString)
      v(1).toString should include((-2).S(4.W).toString)
      v.toString should include("SInt<4>[2](0=SLit(1,<4>), 1=SLit(-2,<4>)")
    }

    "all lits must be the same type but width cannot be greater than Vec's element width" in {
      val e = intercept[VecLiteralException] {
        val v = Vec(2, SInt(4.W)).Lit(0 -> 11.S, 1 -> -0xffff.S)
      }
      e.getMessage should include(
        "VecLiteral: Vec[SInt<4>] has the following incorrectly typed or sized initializers: 0 -> SInt<5>(11),1 -> SInt<17>(-65535)"
      )
    }
  }

  // NOTE: I had problems where this would not work if this class declaration was inside test scope
  class HasVecInit extends Module {
    val initValue = Vec(4, UInt(8.W)).Lit(0 -> 0xab.U(8.W), 1 -> 0xcd.U(8.W), 2 -> 0xef.U(8.W), 3 -> 0xff.U(8.W))
    val y = RegInit(initValue)
  }

  "Vec literals should work when used to initialize a reg of vec" in {
    val firrtl = ChiselStage.emitCHIRRTL(new HasVecInit)
    firrtl should include("""connect _y_WIRE[0], UInt<8>(0hab)""")
    firrtl should include("""connect _y_WIRE[1], UInt<8>(0hcd)""")
    firrtl should include("""connect _y_WIRE[2], UInt<8>(0hef)""")
    firrtl should include("""connect _y_WIRE[3], UInt<8>(0hff)""")
    firrtl should include("""regreset y : UInt<8>[4], clock, reset, _y_WIRE""".stripMargin)
  }

  // NOTE: I had problems where this would not work if this class declaration was inside test scope
  class HasPartialVecInit extends Module {
    val initValue = Vec(4, UInt(8.W)).Lit(0 -> 0xab.U(8.W), 2 -> 0xef.U(8.W), 3 -> 0xff.U(8.W))
    val y = RegInit(initValue)
  }

  "Vec literals should work when used to partially initialize a reg of vec" in {
    val firrtl = ChiselStage.emitCHIRRTL(new HasPartialVecInit)
    firrtl should include("""connect _y_WIRE[0], UInt<8>(0hab)""")
    firrtl should include("""invalidate _y_WIRE[1]""")
    firrtl should include("""connect _y_WIRE[2], UInt<8>(0hef)""")
    firrtl should include("""connect _y_WIRE[3], UInt<8>(0hff)""")
    firrtl should include("""regreset y : UInt<8>[4], clock, reset, _y_WIRE""".stripMargin)
  }

  class ResetRegWithPartialVecLiteral extends Module {
    val in = IO(Input(Vec(4, UInt(8.W))))
    val out = IO(Output(Vec(4, UInt(8.W))))
    val initValue = Vec(4, UInt(8.W)).Lit(0 -> 0xab.U(8.W), 2 -> 0xef.U(8.W), 3 -> 0xff.U(8.W))
    val y = RegInit(initValue)
    when(in(1) > 0.U) {
      y(1) := in(1)
    }
    when(in(2) > 0.U) {
      y(2) := in(2)
    }
    out := y
  }

  "Vec literals should only init specified fields when used to partially initialize a reg of vec" in {
    simulate(new Module {
      val m = Module(new ResetRegWithPartialVecLiteral)
      val (counter, wrapped) = Counter(true.B, 8)
      m.in := DontCare
      when(counter < 2.U) {
        m.in(1) := 0xff.U
        m.in(2) := 0xff.U
      }.elsewhen(counter === 2.U) {
        chisel3.assert(m.out(1) === 0xff.U)
        chisel3.assert(m.out(2) === 0xff.U)
      }.elsewhen(counter === 3.U) {
        m.in(1) := 0.U
        m.in(2) := 0.U
        m.reset := true.B
      }.elsewhen(counter > 2.U) {
        // m.out(1) should not be reset, m.out(2) should be reset
        chisel3.assert(m.out(1) === 0xff.U)
        chisel3.assert(m.out(2) === 0xef.U)
      }
      when(wrapped) {
        stop()
      }
    })(RunUntilFinished(9))
  }

  "lowest of vec literal contains least significant bits and " in {
    val y = Vec(4, UInt(8.W)).Lit(0 -> 0xab.U(8.W), 1 -> 0xcd.U(8.W), 2 -> 0xef.U(8.W), 3 -> 0xff.U(8.W))
    y.litValue should be(BigInt("FFEFCDAB", 16))
  }

  "the order lits are specified does not matter" in {
    val y = Vec(4, UInt(8.W)).Lit(3 -> 0xff.U(8.W), 2 -> 0xef.U(8.W), 1 -> 0xcd.U(8.W), 0 -> 0xab.U(8.W))
    y.litValue should be(BigInt("FFEFCDAB", 16))
  }

  "regardless of the literals widths, packing should be done based on the width of the Vec's gen" in {
    val z = Vec(4, UInt(8.W)).Lit(0 -> 0x2.U, 1 -> 0x2.U, 2 -> 0x2.U, 3 -> 0x3.U)
    z.litValue should be(BigInt("03020202", 16))
  }

  "packing sparse vec lits should not pack, litOption returns None" in {
    // missing sub-listeral for index 2
    val z = Vec(4, UInt(8.W)).Lit(0 -> 0x2.U, 1 -> 0x2.U, 3 -> 0x3.U)

    z.litOption should be(None)
  }

  "registers can be initialized with a Vec literal" in {
    simulate(new Module {
      val y = RegInit(Vec(4, UInt(8.W)).Lit(0 -> 0xab.U(8.W), 1 -> 0xcd.U(8.W), 2 -> 0xef.U(8.W), 3 -> 0xff.U(8.W)))
      chisel3.assert(y.asUInt === BigInt("FFEFCDAB", 16).U)
      stop()
    })(RunUntilFinished(3))
  }

  "how does asUInt work" in {
    simulate(new Module {
      val vec1 = Vec(4, UInt(16.W)).Lit(0 -> 0xdd.U, 1 -> 0xcc.U, 2 -> 0xbb.U, 3 -> 0xaa.U)

      val vec2 = VecInit(Seq(0xdd.U, 0xcc.U, 0xbb.U, 0xaa.U))
      printf("vec1 %x\n", vec1.asUInt)
      printf("vec2 %x\n", vec2.asUInt)
      stop()
    })(RunUntilFinished(3))
  }

  "Vec literals uint conversion" in {
    class M1 extends Module {
      val out1 = IO(Output(UInt(64.W)))
      val out2 = IO(Output(UInt(64.W)))

      val v1 = Vec(4, UInt(16.W)).Lit(0 -> 0xdd.U, 1 -> 0xcc.U, 2 -> 0xbb.U, 3 -> 0xaa.U)
      out1 := v1.asUInt

      val v2 = VecInit(0xdd.U(16.W), 0xcc.U, 0xbb.U, 0xaa.U)
      out2 := v2.asUInt
    }

    simulate(new Module {
      val m = Module(new M1)
      chisel3.assert(m.out1 === m.out2)
      stop()
    })(RunUntilFinished(3))
  }

  "VecLits should work properly with .asUInt" in {
    val outsideVecLit = Vec(4, UInt(16.W)).Lit(0 -> 0xdd.U, 1 -> 0xcc.U, 2 -> 0xbb.U, 3 -> 0xaa.U)

    simulate {
      new Module {
        chisel3.assert(outsideVecLit(0) === 0xdd.U, "v(0)")
        stop()
      }
    }(RunUntilFinished(3))
  }

  "bundle literals should work in RTL" in {
    val outsideVecLit = Vec(4, UInt(16.W)).Lit(0 -> 0xdd.U, 1 -> 0xcc.U, 2 -> 0xbb.U, 3 -> 0xaa.U)

    simulate {
      new Module {
        chisel3.assert(outsideVecLit(0) === 0xdd.U, "v(0)")
        chisel3.assert(outsideVecLit(1) === 0xcc.U)
        chisel3.assert(outsideVecLit(2) === 0xbb.U)
        chisel3.assert(outsideVecLit(3) === 0xaa.U)

        chisel3.assert(outsideVecLit.litValue.U === outsideVecLit.asUInt)

        val insideVecLit = Vec(4, UInt(16.W)).Lit(0 -> 0xdd.U, 1 -> 0xcc.U, 2 -> 0xbb.U, 3 -> 0xaa.U)
        chisel3.assert(insideVecLit(0) === 0xdd.U)
        chisel3.assert(insideVecLit(1) === 0xcc.U)
        chisel3.assert(insideVecLit(2) === 0xbb.U)
        chisel3.assert(insideVecLit(3) === 0xaa.U)

        chisel3.assert(insideVecLit(0) === outsideVecLit(0))
        chisel3.assert(insideVecLit(1) === outsideVecLit(1))
        chisel3.assert(insideVecLit(2) === outsideVecLit(2))
        chisel3.assert(insideVecLit(3) === outsideVecLit(3))

        val vecWire1 = Wire(Vec(4, UInt(16.W)))
        vecWire1 := outsideVecLit

        chisel3.assert(vecWire1(0) === 0xdd.U)
        chisel3.assert(vecWire1(1) === 0xcc.U)
        chisel3.assert(vecWire1(2) === 0xbb.U)
        chisel3.assert(vecWire1(3) === 0xaa.U)

        val vecWire2 = Wire(Vec(4, UInt(16.W)))
        vecWire2 := insideVecLit

        chisel3.assert(vecWire2(0) === 0xdd.U)
        chisel3.assert(vecWire2(1) === 0xcc.U)
        chisel3.assert(vecWire2(2) === 0xbb.U)
        chisel3.assert(vecWire2(3) === 0xaa.U)

        stop()
      }
    }(RunUntilFinished(3))
  }

  "partial vec literals should work in RTL" in {
    simulate {
      new Module {
        val vecLit = Vec(4, UInt(8.W)).Lit(0 -> 42.U, 2 -> 5.U)
        chisel3.assert(vecLit(0) === 42.U)
        chisel3.assert(vecLit(2) === 5.U)

        val vecWire = Wire(Vec(4, UInt(8.W)))
        vecWire := vecLit

        chisel3.assert(vecWire(0) === 42.U)
        chisel3.assert(vecWire(2) === 5.U)

        stop()
      }
    }(RunUntilFinished(3))
  }

  "nested vec literals should be constructable" in {
    val outerVec = Vec(2, Vec(3, UInt(4.W))).Lit(
      0 -> Vec(3, UInt(4.W)).Lit(0 -> 1.U, 1 -> 2.U, 2 -> 3.U),
      1 -> Vec(3, UInt(4.W)).Lit(0 -> 4.U, 1 -> 5.U, 2 -> 6.U)
    )

    outerVec.litValue should be(BigInt("654321", 16))
    outerVec(0).litValue should be(BigInt("321", 16))
    outerVec(1).litValue should be(BigInt("654", 16))
    outerVec(0)(0).litValue should be(BigInt(1))
    outerVec(0)(1).litValue should be(BigInt(2))
    outerVec(0)(2).litValue should be(BigInt(3))
    outerVec(1)(0).litValue should be(BigInt(4))
    outerVec(1)(1).litValue should be(BigInt(5))
    outerVec(1)(2).litValue should be(BigInt(6))
  }

  "contained vecs should work" in {
    simulate {
      new Module {
        val outerVec = Vec(2, Vec(3, UInt(4.W))).Lit(
          0 -> Vec(3, UInt(4.W)).Lit(0 -> 1.U, 1 -> 2.U, 2 -> 3.U),
          1 -> Vec(3, UInt(4.W)).Lit(0 -> 4.U, 1 -> 5.U, 2 -> 6.U)
        )

        chisel3.assert(outerVec(0)(0) === 1.U)
        chisel3.assert(outerVec(0)(1) === 2.U)
        chisel3.assert(outerVec(0)(2) === 3.U)
        chisel3.assert(outerVec(1)(0) === 4.U)
        chisel3.assert(outerVec(1)(1) === 5.U)
        chisel3.assert(outerVec(1)(2) === 6.U)

        val v0 = outerVec(0)
        val v1 = outerVec(1)
        chisel3.assert(v0(0) === 1.U)
        chisel3.assert(v0(1) === 2.U)
        chisel3.assert(v0(2) === 3.U)
        chisel3.assert(v1(0) === 4.U)
        chisel3.assert(v1(1) === 5.U)
        chisel3.assert(v1(2) === 6.U)

        stop()
      }
    }(RunUntilFinished(3))
  }

  "partially initialized Vec literals should assign" in {
    simulate {
      new Module {
        def vecFactory = Vec(2, UInt(8.W))

        val vecWire1 = Wire(Output(vecFactory))
        val vecWire2 = Wire(Output(vecFactory))
        val vecLit1 = vecFactory.Lit(0 -> 6.U(8.W))
        val vecLit2 = vecFactory.Lit(1 -> 13.U(8.W))

        vecWire1 := vecLit1
        vecWire2 := vecLit2
        vecWire1(1) := 2.U(8.W)
        printf("vw1(0) %x  vw1(1) %x\n", vecWire1(0).asUInt, vecWire1(1).asUInt)
        chisel3.assert(vecWire1(0) === 6.U(8.W))
        chisel3.assert(vecWire1(1) === 2.U(8.W)) // Last connect won
        chisel3.assert(vecWire2(1) === 13.U(8.W))
        stop()
      }
    }(RunUntilFinished(3))
  }

  "Vec literals should work as register reset values" in {
    simulate {
      new Module {
        val r = RegInit(Vec(3, UInt(11.W)).Lit(0 -> 0xa.U, 1 -> 0xb.U, 2 -> 0xc.U))
        r := (r.asUInt + 1.U).asTypeOf(Vec(3, UInt(11.W))) // prevent constprop

        // check reset values on first cycle out of reset
        chisel3.assert(r(0) === 0xa.U)
        chisel3.assert(r(1) === 0xb.U)
        chisel3.assert(r(2) === 0xc.U)
        stop()
      }
    }(RunUntilFinished(3))
  }

  "partially initialized Vec literals should work as register reset values" in {
    simulate {
      new Module {
        val r = RegInit(Vec(3, UInt(11.W)).Lit(0 -> 0xa.U, 2 -> 0xc.U))
        r := (r.asUInt + 1.U).asTypeOf(Vec(3, UInt(11.W))) // prevent constprop
        // check reset values on first cycle out of reset
        chisel3.assert(r(0) === 0xa.U)
        chisel3.assert(r(2) === 0xc.U)
        stop()
      }
    }(RunUntilFinished(3))
  }

  "Fields extracted from Vec Literals should work as register reset values" in {
    simulate {
      new Module {
        val r = RegInit(Vec(3, UInt(11.W)).Lit(0 -> 0xa.U, 2 -> 0xc.U).apply(0))
        r := r + 1.U // prevent const prop
        chisel3.assert(r === 0xa.U) // coming out of reset
        stop()
      }
    }(RunUntilFinished(3))
  }

  "DontCare fields extracted from Vec Literals should work as register reset values" in {
    simulate {
      new Module {
        val r = RegInit(Vec(3, Bool()).Lit(0 -> true.B).apply(2))
        r := reset.asBool
        printf(p"r = $r\n") // Can't assert because reset value is DontCare
        stop()
      }
    }(RunUntilFinished(3))
  }

  "DontCare fields extracted from Vec Literals should work in other Expressions" in {
    simulate {
      new Module {
        val x = Vec(3, Bool()).Lit(0 -> true.B).apply(2) || true.B
        chisel3.assert(x === true.B)
        stop()
      }
    }(RunUntilFinished(3))
  }

  "vec literals with non-literal values should fail" in {
    val exc = intercept[VecLiteralException] {
      ChiselStage.emitCHIRRTL {
        new RawModule {
          (Vec(3, UInt(11.W)).Lit(0 -> UInt()))
        }
      }
    }
    exc.getMessage should include("field 0 specified with non-literal value UInt")
  }

  "vec literals are instantiated on connect and are not bulk connected" in {
    class VecExample5 extends RawModule {
      val out = IO(Output(Vec(2, UInt(4.W))))
      val bundle = Vec(2, UInt(4.W)).Lit(
        0 -> 0xa.U,
        1 -> 0xb.U
      )
      out := bundle
    }

    val firrtl = ChiselStage.emitCHIRRTL(new VecExample5)
    firrtl should include("""connect out[0], UInt<4>(0ha)""")
    firrtl should include("""connect out[1], UInt<4>(0hb)""")
  }

  class SubBundle extends Bundle {
    val foo = UInt(8.W)
    val bar = UInt(4.W)
  }

  class VecExample extends RawModule {
    val out = IO(Output(Vec(2, new SubBundle)))
    val bundle = Vec(2, new SubBundle).Lit(
      0 -> (new SubBundle).Lit(_.foo -> 42.U, _.bar -> 0xd.U),
      1 -> (new SubBundle).Lit(_.foo -> 7.U, _.bar -> 3.U)
    )
    out := bundle
  }

  "vec literals can contain bundles and should not be bulk connected" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new VecExample)
    chirrtl should include("""connect out[0].bar, UInt<4>(0hd)""")
    chirrtl should include("""connect out[0].foo, UInt<8>(0h2a)""")
    chirrtl should include("""connect out[1].bar, UInt<4>(0h3)""")
    chirrtl should include("""connect out[1].foo, UInt<8>(0h7)""")
  }

  "vec literals can have bundle children" in {
    val vec = Vec(2, new SubBundle).Lit(
      0 -> (new SubBundle).Lit(_.foo -> 0xab.U, _.bar -> 0xc.U),
      1 -> (new SubBundle).Lit(_.foo -> 0xde.U, _.bar -> 0xf.U)
    )
    vec.litValue.toString(16) should be("defabc")
  }

  "vec literals can have bundle children assembled incrementally" in {
    val bundle1 = (new SubBundle).Lit(_.foo -> 0xab.U, _.bar -> 0xc.U)
    val bundle2 = (new SubBundle).Lit(_.foo -> 0xde.U, _.bar -> 0xf.U)

    bundle1.litValue.toString(16) should be("abc")
    bundle2.litValue.toString(16) should be("def")

    val vec = Vec(2, new SubBundle).Lit(0 -> bundle1, 1 -> bundle2)

    vec.litValue.toString(16) should be("defabc")
  }

  "bundles can contain vec lits" in {
    val vec1 = Vec(3, UInt(4.W)).Lit(0 -> 0xa.U, 1 -> 0xb.U, 2 -> 0xc.U)
    val vec2 = Vec(2, UInt(4.W)).Lit(0 -> 0xd.U, 1 -> 0xe.U)
    val bundle = (new Bundle {
      val foo = Vec(3, UInt(4.W))
      val bar = Vec(2, UInt(4.W))
    }).Lit(_.foo -> vec1, _.bar -> vec2)
    bundle.litValue.toString(16) should be("cbaed")
  }

  "bundles can contain vec lits in-line" in {
    val bundle = (new Bundle {
      val foo = Vec(3, UInt(4.W))
      val bar = Vec(2, UInt(4.W))
    }).Lit(
      _.foo -> Vec(3, UInt(4.W)).Lit(0 -> 0xa.U, 1 -> 0xb.U, 2 -> 0xc.U),
      _.bar -> Vec(2, UInt(4.W)).Lit(0 -> 0xd.U, 1 -> 0xe.U)
    )
    bundle.litValue.toString(16) should be("cbaed")
  }

  "Vec.Lit is a trivial Vec literal factory" in {
    val vec = Vec.Lit(0xa.U, 0xb.U)
    vec(0).litValue should be(0xa)
    vec(1).litValue should be(0xb)
  }

  "Vec.Lit bases it's element width on the widest literal supplied" in {
    val vec = Vec.Lit(0xa.U, 0xbbbb.U)
    vec(0).litValue should be(0xa)
    vec(1).litValue should be(0xbbbb)
    vec.length should be(2)
    vec.getWidth should be(16 * 2)
    vec.litValue should be(BigInt("bbbb000a", 16))
  }

  "vec literals should materialize const wires" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new Module {
      val r = RegInit(Vec(2, UInt(4.W)).Lit(0 -> 1.U, 1 -> 2.U))
    })
    val wire = """wire.*: const UInt<4>\[2\]""".r
    (chirrtl should include).regex(wire)
  }

  "Empty vec literals should be supported" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val lit = Vec(0, UInt(8.W)).Lit()
      lit.litOption should equal(Some(0))
    })
    // It should also work when the element type is a Bundle
    class MyBundle extends Bundle {
      val a = UInt(8.W)
      val b = UInt(8.W)
    }
    ChiselStage.emitCHIRRTL(new RawModule {
      val lit = Vec(0, new MyBundle).Lit()
      lit.litOption should equal(Some(0))
    })
  }

  "Vec literals should use the width of the Vec element rather than the widths of the literals" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      // Whether the user specifies a width or not.
      val lit0 = (Vec(2, UInt(4.W))).Lit(0 -> 0x3.U, 1 -> 0x2.U(3.W))
      lit0(0).getWidth should be(4)
      lit0(1).getWidth should be(4)
      val uint0 = Cat(lit0(1), lit0(0))
      val lit1 = Vec.Lit(0x3.U, 0x2.U(4.W))
      lit1(0).getWidth should be(4)
      lit1(1).getWidth should be(4)
      val uint1 = Cat(lit1(1), lit1(0))
    })
    chirrtl should include("node uint0 = cat(UInt<4>(0h2), UInt<4>(0h3))")
    chirrtl should include("node uint1 = cat(UInt<4>(0h2), UInt<4>(0h3))")
  }

  "Calling .asUInt on a Vec literal should return a UInt literal and work outside of elaboration" in {
    val vlit0 = Vec(2, UInt(4.W)).Lit(0 -> 0x3.U, 1 -> 0x2.U(3.W))
    val ulit0 = vlit0.asUInt
    ulit0.litOption should be(Some(0x23))

    val vlit1 = Vec.Lit(0x3.U, 0x2.U(4.W))
    val ulit1 = vlit1.asUInt
    ulit1.litOption should be(Some(0x23))

    simulate {
      new Module {
        // Check that it gives the same value as the generated hardware
        val wire0 = WireInit(vlit0).asUInt
        chisel3.assert(ulit0.litValue.U === wire0)
        val wire1 = WireInit(vlit1).asUInt
        chisel3.assert(ulit1.litValue.U === wire1)

        stop()
      }
    }(RunUntilFinished(3))
  }

  "Calling .asUInt on a Vec literal with DontCare fields should NOT return a UInt literal" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val vlit = Vec(2, UInt(4.W)).Lit(1 -> 0x2.U(3.W))
      val ulit = vlit.asUInt
      ulit.litOption should be(None)
    })
  }

  "Casting a Vec literal to a complex type should maintain the literal value" in {
    class MyBundle extends Bundle {
      val foo = UInt(1.W)
      val bar = SInt(3.W)
    }
    val vlit = Vec.Lit(0xab.U, 0xcd.U)
    val olit = vlit.asTypeOf(Vec(4, new MyBundle))
    olit.litOption should be(Some(0xcdab))
    olit(0).litValue should be(0xb)
    olit(0).foo.litValue should be(1)
    olit(0).bar.litValue should be(3)
    olit(1).litValue should be(0xa)
    olit(1).foo.litValue should be(1)
    olit(1).bar.litValue should be(2)
    olit(2).litValue should be(0xd)
    olit(2).foo.litValue should be(1)
    olit(2).bar.litValue should be(-3)
    olit(3).litValue should be(0xc)
    olit(3).foo.litValue should be(1)
    olit(3).bar.litValue should be(-4)

    simulate {
      new Module {
        // Check that it gives the same value as the generated hardware.
        val wire = WireInit(vlit).asTypeOf(Vec(4, new MyBundle))
        // ScalaTest has its own multiversal === which overrules extension method.
        // Manually instantiate extension method to get around it.
        chisel3.assert(new Data.DataEquality(olit).===(wire))
        stop()
      }
    }(RunUntilFinished(3))
  }
}
