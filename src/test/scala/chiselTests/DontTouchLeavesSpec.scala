// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import circt.stage.ChiselStage
import chiselTests.experimental.hierarchy.Utils

import firrtl.transforms.DontTouchAnnotation

class HasDeadCodeChildLeaves(withDontTouchAgg: Boolean) extends Module {
  val io = IO(new Bundle {
    val a = Input(new Bundle {val a1 = UInt(32.W); val a2 = UInt(32.W)})
    val b = Output(new Bundle {val b1 = UInt(32.W); val b2 = UInt(32.W)})
  })

  io.b.b1 := io.a.a1
  io.b.b2 := DontCare
  dontTouch(io.a, withDontTouchAgg)
}

class HasDeadCodeLeaves(withDontTouchAgg: Boolean) extends Module {
  val io = IO(new Bundle {
    val a = Input(UInt(32.W))
    val b = Output(UInt(32.W))
  })
  val inst = Module(new HasDeadCodeChildLeaves(withDontTouchAgg))
  inst.io.a.a1 := io.a
  inst.io.a.a2 := io.a
  val tmp = inst.io.b.b1 + inst.io.b.b2
  dontTouch(tmp, withDontTouchAgg)
  io.b := tmp
}

class DontTouchLeavesSpec extends ChiselFlatSpec with Utils {
  "fields" should "be marked don't touch by default" in {
    val (_, annos) = getFirrtlAndAnnos(new HasDeadCodeLeaves(false))
    annos should contain(DontTouchAnnotation("~HasDeadCodeLeaves|HasDeadCodeChildLeaves>io.a.a1".rt))
    annos should not contain(DontTouchAnnotation("~HasDeadCodeLeaves|HasDeadCodeChildLeaves>io.a".rt))
  }
  "aggregates" should "be marked if marked markAgg is true" in {
    val (_, annos) = getFirrtlAndAnnos(new HasDeadCodeLeaves(true))
    annos should not contain(DontTouchAnnotation("~HasDeadCodeLeaves|HasDeadCodeChildLeaves>io.a.a1".rt))
    annos should contain(DontTouchAnnotation("~HasDeadCodeLeaves|HasDeadCodeChildLeaves>io.a".rt))
  }
}
