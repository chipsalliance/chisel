// SPDX-License-Identifier: Apache-2.0

package firrtl.fuzzer

import org.scalacheck.{Gen, Prop, Properties}
import firrtl.{ChirrtlForm, CircuitState, CustomTransformException, LowFirrtlCompiler, Namespace}
import firrtl.passes.CheckWidths
import ExprGen._
import ScalaCheckGenMonad._

object FirrtlCompileProperties extends Properties("FirrtlCompile") {
  property("compile") = {
    val gen = Gen.sized { size =>
      val params = ExprGenParams(
        maxDepth = size / 3,
        maxWidth = math.min(size + 1, CheckWidths.MaxWidth),
        generators = Map(
           AddDoPrimGen -> 1,
           SubDoPrimGen -> 1,
           MulDoPrimGen -> 1,
           DivDoPrimGen -> 1,
           LtDoPrimGen -> 1,
           LeqDoPrimGen -> 1,
           GtDoPrimGen -> 1,
           GeqDoPrimGen -> 1,
           EqDoPrimGen -> 1,
           NeqDoPrimGen -> 1,
           PadDoPrimGen -> 1,
           ShlDoPrimGen -> 1,
           ShrDoPrimGen -> 1,
           DshlDoPrimGen -> 1,
           CvtDoPrimGen -> 1,
           NegDoPrimGen -> 1,
           NotDoPrimGen -> 1,
           AndDoPrimGen -> 1,
           OrDoPrimGen -> 1,
           XorDoPrimGen -> 1,
           AndrDoPrimGen -> 1,
           OrrDoPrimGen -> 1,
           XorrDoPrimGen -> 1,
           CatDoPrimGen -> 1,
           BitsDoPrimGen -> 1,
           HeadDoPrimGen -> 1,
           TailDoPrimGen -> 1,
           AsUIntDoPrimGen -> 1,
           AsSIntDoPrimGen -> 1,
        )
      )
      params.generateSingleExprCircuit[Gen]()
    }
    val lowFirrtlCompiler = new firrtl.LowFirrtlCompiler()
    Prop.forAll(gen) { circuit =>
      val state = CircuitState(circuit, ChirrtlForm, Seq())
      //val compiler = new LowFirrtlCompiler()
      val compiler = lowFirrtlCompiler
      try {
        val res = compiler.compile(state, Seq())
        true
      } catch {
        case e: CustomTransformException => false
        case any : Throwable => false
      }
    }
  }
}
