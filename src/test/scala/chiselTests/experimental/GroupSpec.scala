// See LICENSE for license details.

package chiselTests.experimental

import chiselTests.ChiselFlatSpec
import chisel3._
import chisel3.stage.{ChiselGeneratorAnnotation, ChiselMain}
import chisel3.util.experimental.group
import firrtl.analyses.InstanceGraph
import firrtl.options.{Dependency, TargetDirAnnotation}
import firrtl.stage.CompilerAnnotation
import firrtl.{EmittedFirrtlCircuitAnnotation, FirrtlExecutionSuccess, LowFirrtlCompiler, LowFirrtlEmitter, ir => fir}

import scala.collection.mutable

class GroupSpec extends ChiselFlatSpec {

  class MyModule extends Module {
    val io = IO(new Bundle{
      val a = Input(Bool())
      val b = Output(Bool())
    })
    val reg1 = RegInit(0.U)
    reg1 := io.a
    val reg2 = RegNext(reg1)
    io.b := reg2
    group(Seq(reg1, reg2), "DosRegisters", "doubleReg")
  }

  def collectInstances(c: fir.Circuit, top: Option[String] = None): Seq[String] = new InstanceGraph(c)
    .fullHierarchy.values.flatten.toSeq
    .map( v => (top.getOrElse(v.head.name) +: v.tail.map(_.name)).mkString(".") )

  def collectRegisters(m: fir.DefModule): Seq[String] = {
    val regs = mutable.ArrayBuffer[String]()
    def onStmt(s: fir.Statement): fir.Statement = s.mapStmt(onStmt) match {
      case r: fir.DefRegister => regs += r.name; r
      case other => other
    }
    m.mapStmt(onStmt)
    regs
  }

  "Module Grouping" should "compile to low FIRRTL" in {
    val firrtlCircuit = (ChiselMain.stage.run(
      Seq(
        CompilerAnnotation(new LowFirrtlCompiler()),
        TargetDirAnnotation("test_run_dir"),
        ChiselGeneratorAnnotation( () => new MyModule)
      )
    ) collectFirst {
      case firrtl.stage.FirrtlCircuitAnnotation(circuit) => circuit
    }).get

    firrtlCircuit.modules.collect {
      case m: fir.Module if m.name == "MyModule" =>
        Nil should be (collectRegisters(m))
      case m: fir.Module if m.name == "DosRegisters" =>
        Seq("reg1", "reg2") should be (collectRegisters(m))
    }
    val instances = collectInstances(firrtlCircuit, Some("MyModule")).toSet
    Set("MyModule", "MyModule.doubleReg") should be (instances)
  }
}
