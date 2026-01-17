// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import circt.stage.ChiselStage
import chisel3.util.Valid
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class ValidSpec extends AnyFlatSpec with Matchers {
  "Valid.map" should "apply a function to the wrapped Data" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new Module {
      val in = IO(Flipped(Valid(UInt(8.W))))
      val out = IO(Valid(UInt(8.W)))
      out :#= in.map(_ + 1.U)
    })

    // Check for data assignment
    chirrtl should include("""node _out_map_bits_T = add(in.bits, UInt<1>(0h1))""")
    chirrtl should include("""node _out_map_bits = tail(_out_map_bits_T, 1)""")
    chirrtl should include("""connect _out_map.bits, _out_map_bits""")
    chirrtl should include("""connect out.bits, _out_map.bits""")

    // Check for valid assignment
    chirrtl should include("""connect _out_map.valid, in.valid""")
    chirrtl should include("""connect out.valid, _out_map.valid""")
  }

  "Valid.map" should "apply a function to a wrapped Bundle" in {
    class TestBundle extends Bundle {
      val foo = UInt(8.W)
      val bar = UInt(8.W)
      val fizz = Bool()
      val buzz = Bool()
    }

    // Add one to foo, subtract one from bar, set fizz to false and buzz to true
    def func(t: TestBundle): TestBundle = {
      val res = Wire(new TestBundle)

      res.foo := t.foo + 1.U
      res.bar := t.bar - 1.U
      res.fizz := false.B
      res.buzz := true.B

      res
    }

    val chirrtl = ChiselStage
      .emitCHIRRTL(new Module {
        val in = IO(Flipped(Valid(new TestBundle)))
        val out = IO(Valid(new TestBundle))
        out :#= in.map(func)
      })

    // Check for data assignment
    chirrtl should include("""wire _out_map_bits : { foo : UInt<8>, bar : UInt<8>, fizz : UInt<1>, buzz : UInt<1>}""")

    chirrtl should include("""node _out_map_bits_res_foo_T = add(in.bits.foo, UInt<1>(0h1)""")
    chirrtl should include("""node _out_map_bits_res_foo_T_1 = tail(_out_map_bits_res_foo_T, 1)""")
    chirrtl should include("""connect _out_map_bits.foo, _out_map_bits_res_foo_T_1""")

    chirrtl should include("""node _out_map_bits_res_bar_T = sub(in.bits.bar, UInt<1>(0h1)""")
    chirrtl should include("""node _out_map_bits_res_bar_T_1 = tail(_out_map_bits_res_bar_T, 1)""")
    chirrtl should include("""connect _out_map_bits.bar, _out_map_bits_res_bar_T_1""")

    chirrtl should include("""connect _out_map_bits.fizz, UInt<1>(0h0)""")
    chirrtl should include("""connect _out_map_bits.buzz, UInt<1>(0h1)""")

    chirrtl should include("""connect _out_map.bits, _out_map_bits""")
    for ((field, _) <- (new TestBundle).elements) {
      chirrtl should include(s"""connect out.bits.$field, _out_map.bits.$field""")
    }

    // Check for valid assignment
    chirrtl should include("""connect _out_map.valid, in.valid""")
    chirrtl should include("""connect out.valid, _out_map.valid""")
  }

  "Valid.map" should "apply a function to a wrapped Bundle and return a different typed Valid" in {
    class TestBundle extends Bundle {
      val foo = UInt(8.W)
      val bar = UInt(8.W)
    }

    val chirrtl = ChiselStage
      .emitCHIRRTL(new Module {
        val in = IO(Flipped(Valid(new TestBundle)))
        val out = IO(Valid(UInt(8.W)))
        out :#= in.map(bundle => bundle.foo & bundle.bar)
      })

    // Check that the _map wire wraps a UInt and not a TestBundle
    chirrtl should include("""wire _out_map : { valid : UInt<1>, bits : UInt<8>}""")

    // Check for data assignment
    chirrtl should include("""node _out_map_bits = and(in.bits.foo, in.bits.bar)""")
    chirrtl should include("""connect _out_map.bits, _out_map_bits""")
    chirrtl should include("""connect out.bits, _out_map.bits""")

    // Check for valid assignment
    chirrtl should include("""connect _out_map.valid, in.valid""")
    chirrtl should include("""connect out.valid, _out_map.valid""")
  }
}
