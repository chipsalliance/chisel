// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.ltl._
import chisel3.testing.scalatest.FileCheck
import _root_.circt.stage.ChiselStage

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import Sequence._

class FormalContractSpec extends AnyFlatSpec with Matchers with FileCheck {
  behavior.of("FormalContract")

  def check(gen: => RawModule)(fc: String): Unit =
    ChiselStage.emitCHIRRTL(gen).fileCheck("--strict-whitespace", "--match-full-lines")(fc)

  it should "support contracts with no operands" in {
    class Foo extends RawModule {
      val a = IO(Input(Bool()))
      FormalContract {
        EnsureProperty(a)
      }
    }
    check(new Foo)("""
      // CHECK-LABEL:  public module Foo : {{@.*}}
      //  CHECK-NEXT:    input a : UInt<1> {{@.*}}
      //       CHECK:    contract : {{@.*}}
      //  CHECK-NEXT:      intrinsic(circt_verif_ensure, a) {{@.*}}
      """)
  }

  it should "support contracts with a single operand" in {
    class Foo extends RawModule {
      val a = IO(Input(Bool()))
      val b = FormalContract(a) { x =>
        EnsureProperty(a)
        EnsureProperty(x)
      }
      layer.elideBlocks {
        AssumeProperty(b)
      }
    }
    check(new Foo)("""
      // CHECK-LABEL:  public module Foo : {{@.*}}
      //  CHECK-NEXT:    input a : UInt<1> {{@.*}}
      //       CHECK:    contract b = a : {{@.*}}
      //  CHECK-NEXT:      intrinsic(circt_verif_ensure, a) {{@.*}}
      //  CHECK-NEXT:      intrinsic(circt_verif_ensure, b) {{@.*}}
      //       CHECK:    intrinsic(circt_verif_assume, b) {{@.*}}
      """)
  }

  it should "support contracts with multiple operands" in {
    class Foo extends RawModule {
      val a = IO(Input(Bool()))
      val b = IO(Input(Bool()))
      val (c, d) = FormalContract(a, b) { case (x, y) =>
        EnsureProperty(a)
        EnsureProperty(b)
        EnsureProperty(x)
        EnsureProperty(y)
      }
      layer.elideBlocks {
        AssumeProperty(c)
        AssumeProperty(d)
      }
    }
    check(new Foo)("""
      // CHECK-LABEL:  public module Foo : {{@.*}}
      //  CHECK-NEXT:    input a : UInt<1> {{@.*}}
      //  CHECK-NEXT:    input b : UInt<1> {{@.*}}
      //       CHECK:    contract [[C:.+]], [[D:.+]] = a, b : {{@.*}}
      //  CHECK-NEXT:      intrinsic(circt_verif_ensure, a) {{@.*}}
      //  CHECK-NEXT:      intrinsic(circt_verif_ensure, b) {{@.*}}
      //  CHECK-NEXT:      intrinsic(circt_verif_ensure, [[C]]) {{@.*}}
      //  CHECK-NEXT:      intrinsic(circt_verif_ensure, [[D]]) {{@.*}}
      //       CHECK:    intrinsic(circt_verif_assume, [[C]]) {{@.*}}
      //  CHECK-NEXT:    intrinsic(circt_verif_assume, [[D]]) {{@.*}}
      """)
  }

  it should "support nested contracts" in {
    class Foo extends RawModule {
      val a = IO(Input(Bool()))
      val b = FormalContract(a) { x =>
        val c = FormalContract(x) { y =>
          EnsureProperty(y)
        }
      }
    }
    check(new Foo)("""
      // CHECK-LABEL:  public module Foo : {{@.*}}
      //  CHECK-NEXT:    input a : UInt<1> {{@.*}}
      //       CHECK:    contract [[B:.+]] = a : {{@.*}}
      //  CHECK-NEXT:      contract [[C:.+]] = [[B]] : {{@.*}}
      //  CHECK-NEXT:        intrinsic(circt_verif_ensure, [[C]]) {{@.*}}
      """)
  }
}
