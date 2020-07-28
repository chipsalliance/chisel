// See LICENSE for license details.

package chiselTests.experimental

import chiselTests.ChiselFlatSpec
import chisel3._
import chisel3.RawModule
import chisel3.stage.{ChiselGeneratorAnnotation, ChiselStage}
import chisel3.util.experimental.group
import firrtl.analyses.InstanceGraph
import firrtl.options.TargetDirAnnotation
import firrtl.stage.CompilerAnnotation
import firrtl.{LowFirrtlCompiler, ir => fir}

import scala.collection.mutable

class GroupSpec extends ChiselFlatSpec {

  def collectInstances(c: fir.Circuit, top: Option[String] = None): Seq[String] = new InstanceGraph(c)
    .fullHierarchy.values.flatten.toSeq
    .map( v => (top.getOrElse(v.head.name) +: v.tail.map(_.name)).mkString(".") )

  def collectDeclarations(m: fir.DefModule): Set[String] = {
    val decs = mutable.HashSet[String]()
    def onStmt(s: fir.Statement): fir.Statement = s.mapStmt(onStmt) match {
      case d: fir.IsDeclaration => decs += d.name; d
      case other => other
    }
    m.mapStmt(onStmt)
    decs.toSet
  }

  def lower[T <: RawModule](gen: () => T): fir.Circuit = {
    (new ChiselStage)
      .execute(Array("--compiler", "low",
                     "--target-dir", "test_run_dir"),
               Seq(ChiselGeneratorAnnotation(gen)))
      .collectFirst {
        case firrtl.stage.FirrtlCircuitAnnotation(circuit) => circuit
      }.get
  }

  "Module Grouping" should "compile to low FIRRTL" in {
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

    val firrtlCircuit = lower(() => new MyModule)
    firrtlCircuit.modules.collect {
      case m: fir.Module if m.name == "MyModule" =>
        Set("doubleReg") should be (collectDeclarations(m))
      case m: fir.Module if m.name == "DosRegisters" =>
        Set("reg1", "reg2") should be (collectDeclarations(m))
    }
    val instances = collectInstances(firrtlCircuit, Some("MyModule")).toSet
    Set("MyModule", "MyModule.doubleReg") should be (instances)
  }

  "Module Grouping" should "not include intermediate registers" in {
    class MyModule extends Module {
      val io = IO(new Bundle{
        val a = Input(Bool())
        val b = Output(Bool())
      })
      val reg1 = RegInit(0.U)
      reg1 := io.a
      val reg2 = RegNext(reg1)
      val reg3 = RegNext(reg2)
      io.b := reg3
      group(Seq(reg1, reg3), "DosRegisters", "doubleReg")
    }

    val firrtlCircuit = lower(() => new MyModule)
    firrtlCircuit.modules.collect {
      case m: fir.Module if m.name == "MyModule" =>
        Set("reg2", "doubleReg") should be (collectDeclarations(m))
      case m: fir.Module if m.name == "DosRegisters" =>
        Set("reg1", "reg3") should be (collectDeclarations(m))
    }
    val instances = collectInstances(firrtlCircuit, Some("MyModule")).toSet
    Set("MyModule", "MyModule.doubleReg") should be (instances)
  }

  "Module Grouping" should "include intermediate wires" in {
    class MyModule extends Module {
      val io = IO(new Bundle{
        val a = Input(Bool())
        val b = Output(Bool())
      })
      val reg1 = RegInit(0.U)
      reg1 := io.a
      val wire = WireInit(reg1)
      val reg3 = RegNext(wire)
      io.b := reg3
      group(Seq(reg1, reg3), "DosRegisters", "doubleReg")
    }

    val firrtlCircuit = lower(() => new MyModule)
    firrtlCircuit.modules.collect {
      case m: fir.Module if m.name == "MyModule" =>
        Set("doubleReg") should be (collectDeclarations(m))
      case m: fir.Module if m.name == "DosRegisters" =>
        Set("reg1", "reg3", "wire") should be (collectDeclarations(m))
    }
    val instances = collectInstances(firrtlCircuit, Some("MyModule")).toSet
    Set("MyModule", "MyModule.doubleReg") should be (instances)
  }
}
