package chiselTests.experimental

import chiselTests.ChiselRunners
import org.scalatest.{FreeSpec, Matchers}
import chisel3._
import chisel3.util.experimental.group
import firrtl.analyses.InstanceGraph
import firrtl.transforms.GroupAnnotation
import firrtl.{FirrtlExecutionSuccess, ir => fir}

import scala.collection.mutable

class GroupSpec extends FreeSpec with ChiselRunners with Matchers {

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

  "Module Grouping" - {
    "should compile to low FIRRTL" - {
      Driver.execute(Array("-X", "low", "--target-dir", "test_run_dir"), () => new MyModule) match {
        case ChiselExecutionSuccess(Some(chiselCircuit), _, Some(firrtlResult: FirrtlExecutionSuccess)) =>
          "emitting one GroupAnnotation at the CHIRRTL level" in {
            chiselCircuit.annotations.map(_.toFirrtl).collect{ case a: GroupAnnotation => a }.size should be (1)
          }
          "low FIRRTL should contain only instance z" in {
            firrtlResult.circuitState.circuit.modules.collect {
              case m: fir.Module if m.name == "MyModule" =>
                Nil should be (collectRegisters(m))
              case m: fir.Module if m.name == "DosRegisters" =>
                Seq("reg1", "reg2") should be (collectRegisters(m))
            }
            val instances = collectInstances(firrtlResult.circuitState.circuit, Some("MyModule")).toSet
            Set("MyModule", "MyModule.doubleReg") should be (instances)
          }
      }
    }
  }
}
