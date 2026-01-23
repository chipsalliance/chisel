// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.experimental.hierarchy._
import chisel3.experimental.{annotate, dedupGroup}
import chisel3.properties.Class
import chisel3.stage.{ChiselGeneratorAnnotation, CircuitSerializationAnnotation}
import chisel3.testing.scalatest.FileCheck
import chisel3.util.circt.PlusArgsValue
import chisel3.util.{Counter, Decoupled, Queue}
import circt.stage.ChiselStage
import firrtl.EmittedVerilogCircuitAnnotation
import firrtl.transforms.DedupGroupAnnotation
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class DedupIO extends Bundle {
  val in = Flipped(Decoupled(UInt(32.W)))
  val out = Decoupled(UInt(32.W))
}

class DedupQueues(n: Int) extends Module {
  require(n > 0)
  val io = IO(new DedupIO)
  val queues = Seq.fill(n)(Module(new Queue(UInt(32.W), 4)))
  var port = io.in
  for (q <- queues) {
    q.io.enq <> port
    port = q.io.deq
  }
  io.out <> port
}

/* This module creates a Queue in a nested function (such that it is not named via reflection). The
 * default naming for instances prior to #470 caused otherwise identical instantiations of this
 * module to have different instance names for the queues which prevented deduplication.
 * NestedDedup instantiates this module twice to ensure it is deduplicated properly.
 */
class DedupSubModule extends Module {
  val io = IO(new DedupIO)
  io.out <> Queue(io.in, 4)
}

class NestedDedup extends Module {
  val io = IO(new DedupIO)
  val inst0 = Module(new DedupSubModule)
  val inst1 = Module(new DedupSubModule)
  inst0.io.in <> io.in
  inst1.io.in <> inst0.io.out
  io.out <> inst1.io.out
}

object DedupConsts {
  val foo = 3.U
}

class SharedConstantValDedup extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(8.W))
    val out = Output(UInt(8.W))
  })
  io.out := io.in + DedupConsts.foo
}

class SharedConstantValDedupTop extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(8.W))
    val out = Output(UInt(8.W))
  })
  val inst0 = Module(new SharedConstantValDedup)
  val inst1 = Module(new SharedConstantValDedup)
  inst0.io.in := io.in
  inst1.io.in := io.in
  io.out := inst0.io.out + inst1.io.out
}

class SharedConstantValDedupTopDesiredName extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(8.W))
    val out = Output(UInt(8.W))
  })
  val inst0 = Module(new SharedConstantValDedup {
    override def desiredName = "foo"
  })
  val inst1 = Module(new SharedConstantValDedup {
    override def desiredName = "bar"
  })
  inst0.io.in := io.in
  inst1.io.in := io.in
  io.out := inst0.io.out + inst1.io.out
}

class ModuleWithIntrinsic extends Module {
  val plusarg = PlusArgsValue(Bool(), "plusarg=%d")
}

class ModuleWithClass extends Module {
  val cls = Definition(new Class)
}

class DedupSpec extends AnyFlatSpec with Matchers with FileCheck {
  implicit class VerilogHelpers(verilog: String) {
    private val ModuleRegex = """\s*module\s+(\w+)\b.*""".r
    def countModules: Int =
      (verilog.split("\n").collect { case ModuleRegex(name) => name }).filterNot(_.contains("ram_4x32")).size
  }

  "Deduplication" should "occur" in {
    ChiselStage.emitSystemVerilog(new DedupQueues(4)).countModules should be(2)
  }

  it should "properly dedup modules with deduped submodules" in {
    ChiselStage.emitSystemVerilog(new NestedDedup).countModules should be(3)
  }

  it should "dedup modules that share a literal" in {
    ChiselStage.emitSystemVerilog(new SharedConstantValDedupTop).countModules should be(2)
  }

  it should "not dedup modules that are in different dedup groups" in {
    ChiselStage.emitSystemVerilog {
      val top = new SharedConstantValDedupTop
      dedupGroup(top.inst0, "inst0")
      dedupGroup(top.inst1, "inst1")
      top
    }.countModules should be(3)
  }

  it should "work natively for desiredNames with ChiselStage (the class)" in {
    val verilog = new ChiselStage()
      .execute(
        Array("--target", "systemverilog"),
        Seq(ChiselGeneratorAnnotation(() => new SharedConstantValDedupTopDesiredName))
      )
      .collectFirst { case EmittedVerilogCircuitAnnotation(a) =>
        a.value
      }
      .get

    verilog.countModules should be(3)
  }

  it should "work natively for desiredNames with ChiselStage$ (the object)" in {
    val verilog = ChiselStage.emitSystemVerilog {
      val top = new SharedConstantValDedupTopDesiredName
      top
    }

    verilog.countModules should be(3)
  }

  it should "error on conflicting dedup groups" in {
    a[Exception] should be thrownBy {
      ChiselStage.emitSystemVerilog {
        val top = new SharedConstantValDedupTop
        dedupGroup(top.inst0, "inst0")
        dedupGroup(top.inst0, "anothergroup")
        dedupGroup(top.inst1, "inst1")
        dedupGroup(top.inst1, "anothergroup")
        top
      }
    }
  }

  it should "not add DedupGroupAnnotation to intrinsics" in {
    // TODO: This test _should_ be able to use `ChiselStage$` methods, but there
    // are problems with annotations and D/I.
    //
    // See: https://github.com/chipsalliance/chisel/issues/4730
    val annotations = new ChiselStage()
      .execute(
        Array("--target", "chirrtl"),
        Seq(ChiselGeneratorAnnotation(() => new ModuleWithIntrinsic))
      )
    val chirrtl = annotations.collectFirst { case a: CircuitSerializationAnnotation => a }.get
      .emitLazily(annotations.collect { case a: DedupGroupAnnotation => a })
      .mkString

    chirrtl.fileCheck()(
      """|CHECK:      "class":"firrtl.transforms.DedupGroupAnnotation"
         |CHECK-NEXT: "target":"~|ModuleWithIntrinsic"
         |CHECK-NEXT: "group":"ModuleWithIntrinsic"
         |""".stripMargin
    )
  }

  it should "not add DedupGroupAnnotation to classes" in {
    // TODO: This test _should_ be able to use `ChiselStage$` methods, but there
    // are problems with annotations and D/I.
    //
    // See: https://github.com/chipsalliance/chisel/issues/4730
    val annotations = new ChiselStage()
      .execute(
        Array("--target", "chirrtl"),
        Seq(ChiselGeneratorAnnotation(() => new ModuleWithClass))
      )
    val chirrtl = annotations.collectFirst { case a: CircuitSerializationAnnotation => a }.get
      .emitLazily(annotations.collect { case a: DedupGroupAnnotation => a })
      .mkString

    chirrtl.fileCheck()(
      """|CHECK:      "class":"firrtl.transforms.DedupGroupAnnotation"
         |CHECK-NEXT: "target":"~|ModuleWithClass"
         |CHECK-NEXT: "group":"ModuleWithClass"
         |""".stripMargin
    )
  }
}
