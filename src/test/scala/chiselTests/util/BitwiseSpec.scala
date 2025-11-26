package chiselTests.util

import chisel3._
import chisel3.util.{Fill, FillInterleaved, PopCount, Reverse}
import _root_.circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class FillInterleavedSpec extends AnyFlatSpec with Matchers {
  behavior.of("util.FillInterleaved")

  it should "have source locators when passed a UInt" in {
    class MyModule extends RawModule {
      val out = IO(Output(UInt()))
      out := FillInterleaved(2, "b1000".U)
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
    val cat = """cat.*BitwiseSpec\.scala""".r
    (chirrtl should include).regex(cat)
    val mux = """mux.*BitwiseSpec\.scala""".r
    (chirrtl should include).regex(mux)
    (chirrtl should not).include("Bitwise.scala")
  }

  it should "have source locators when passed a Seq[Bool]" in {
    class MyModule extends RawModule {
      val out = IO(Output(UInt()))
      out := FillInterleaved(2, Seq(true.B, false.B, false.B, false.B))
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
    val cat = """cat.*BitwiseSpec\.scala""".r
    (chirrtl should include).regex(cat)
    val mux = """mux.*BitwiseSpec\.scala""".r
    (chirrtl should include).regex(mux)
    (chirrtl should not).include("Bitwise.scala")
  }
}

class PopCountSpec extends AnyFlatSpec with Matchers {
  behavior.of("util.PopCount")

  it should "have source locators when passed a Iterable[Bool]" in {
    class MyModule extends RawModule {
      val out = IO(Output(UInt()))
      out := PopCount(Seq(true.B, false.B, false.B, false.B))
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
    val add = """add.*BitwiseSpec\.scala""".r
    (chirrtl should include).regex(add)
    val bits = """bits.*BitwiseSpec\.scala""".r
    (chirrtl should include).regex(bits)
    (chirrtl should not).include("Bitwise.scala")
  }

  it should "have source locators when passed a Bits" in {
    class MyModule extends RawModule {
      val out = IO(Output(UInt()))
      out := PopCount("b1000".U)
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
    val add = """add.*BitwiseSpec\.scala""".r
    (chirrtl should include).regex(add)
    val bits = """bits.*BitwiseSpec\.scala""".r
    (chirrtl should include).regex(bits)
    (chirrtl should not).include("Bitwise.scala")
  }
}

class FillSpec extends AnyFlatSpec with Matchers {
  behavior.of("util.Fill")
  it should "have source locators when passed a Bits" in {
    class MyModule extends RawModule {
      val out = IO(Output(UInt()))
      out := Fill(2, "b1000".U)
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
    val cat = """cat.*BitwiseSpec\.scala""".r
    (chirrtl should include).regex(cat)
    (chirrtl should not).include("Bitwise.scala")
  }
}

class ReverseSpec extends AnyFlatSpec with Matchers {
  behavior.of("util.Reverse")

  it should "have source locators when passed a UInt" in {
    class MyModule extends RawModule {
      val out = IO(Output(UInt()))
      out := Reverse("b1101".U)
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new MyModule)
    val cat = """cat.*BitwiseSpec\.scala""".r
    (chirrtl should include).regex(cat)
    (chirrtl should not).include("Bitwise.scala")
  }
}
