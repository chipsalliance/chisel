// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.ltl._
import chisel3.testers.BasicTester
import chisel3.experimental.SourceLine
import _root_.circt.stage.ChiselStage

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import Sequence._

class FormalContractSpec extends AnyFlatSpec with Matchers with FileCheck {
  it should "support contracts with no operands" in {
    class Foo extends RawModule {
      val a = IO(Input(Bool()))
      FormalContract {
        AssertProperty(a)
      }
    }
    generateFirrtlAndFileCheck(new Foo)("""
      // CHECK-LABEL: module Foo
      // CHECK: input a : UInt<1>
      // CHECK: contract :
      // CHECK:   intrinsic(circt_verif_assert, a)
      """)
  }

  it should "support contracts with a single operand" in {
    class Foo extends RawModule {
      val a = IO(Input(Bool()))
      val b = FormalContract(a) { x =>
        AssertProperty(a)
        AssertProperty(x)
      }
      AssumeProperty(b)
    }
    generateFirrtlAndFileCheck(new Foo)("""
      // CHECK-LABEL: module Foo
      // CHECK: input a : UInt<1>
      // CHECK: contract b = a :
      // CHECK:   intrinsic(circt_verif_assert, a)
      // CHECK:   intrinsic(circt_verif_assert, b)
      // CHECK: intrinsic(circt_verif_assume, b)
      """)
  }

  it should "support contracts with multiple operands" in {
    class Foo extends RawModule {
      val a = IO(Input(Bool()))
      val b = IO(Input(Bool()))
      val (c, d) = FormalContract(a, b) { case (x, y) =>
        AssertProperty(a)
        AssertProperty(b)
        AssertProperty(x)
        AssertProperty(y)
      }
      AssumeProperty(c)
      AssumeProperty(d)
    }
    generateFirrtlAndFileCheck(new Foo)("""
      // CHECK-LABEL: module Foo
      // CHECK: input a : UInt<1>
      // CHECK: input b : UInt<1>
      // CHECK: contract [[C:.+]], [[D:.+]] = a, b :
      // CHECK:   intrinsic(circt_verif_assert, a)
      // CHECK:   intrinsic(circt_verif_assert, b)
      // CHECK:   intrinsic(circt_verif_assert, [[C]])
      // CHECK:   intrinsic(circt_verif_assert, [[D]])
      // CHECK: intrinsic(circt_verif_assume, [[C]])
      // CHECK: intrinsic(circt_verif_assume, [[D]])
      """)
  }

  it should "support nested contracts" in {
    class Foo extends RawModule {
      val a = IO(Input(Bool()))
      val b = FormalContract(a) { x =>
        val c = FormalContract(x) { y =>
          AssertProperty(y)
        }
      }
    }
    generateFirrtlAndFileCheck(new Foo)("""
      // CHECK-LABEL: module Foo
      // CHECK: input a : UInt<1>
      // CHECK: contract [[B:.+]] = a :
      // CHECK:   contract [[C:.+]] = [[B]] :
      // CHECK:     intrinsic(circt_verif_assert, [[C]])
      """)
  }
}
