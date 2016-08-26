// See LICENSE for license details.

package chisel3.iotesters

import chisel3.{Module, Data, Element, Bundle, Vec}
import chisel3.internal.InstanceId
import chisel3.internal.firrtl.Circuit
import scala.sys.process._
import scala.collection.mutable.{ArrayBuffer, HashMap}

// TODO: FIRRTL will eventually return valid names
private[iotesters] object validName {
  def apply(name: String) = (if (firrtl.Utils.v_keywords contains name) name + "$"
    else name) replace (".", "_") replace ("[", "_") replace ("]", "")
}

private[iotesters] object getDataNames {
  def apply(name: String, data: Data): Seq[(Data, String)] = data match {
    case b: Element => Seq(b -> name)
    case b: Bundle => b.elements.toSeq flatMap {case (n, e) => apply(s"${name}_$n", e)}
    case v: Vec[_] => v.zipWithIndex flatMap {case (e, i) => apply(s"${name}_$i", e)}
  }
  def apply(dut: Module, separator: String = "."): Seq[(Data, String)] =
    apply(dut.io.pathName replace (".", separator), dut.io)
}

private[iotesters] object getPorts {
  def apply(dut: Module, separator: String = ".") =
    getDataNames(dut, separator) partition (_._1.dir == chisel3.INPUT)
}

private[iotesters] object flatten {
  def apply(data: Data): Seq[Element] = data match {
    case b: Element => Seq(b)
    case b: Bundle => b.elements.toSeq flatMap (x => apply(x._2))
    case v: Vec[_] => v.toSeq flatMap apply
  }
}

private[iotesters] object getTopModule {
  def apply(circuit: Circuit) = {
    (circuit.components find (_.name == circuit.name)).get.id
  }
}

/* TODO: Chisel should provide nodes of the circuit? */
private[iotesters] object getChiselNodes {
  import chisel3.internal.firrtl._
  def apply(circuit: Circuit): Seq[InstanceId] = {
    circuit.components flatMap (_.commands flatMap {
      case x: DefReg => flatten(x.id)
      case x: DefRegInit => flatten(x.id)
      case mem: DefMemory => mem.t match {
        case _: Element => Seq(mem.id)
        case _ => Nil // Do not supoort aggregate type memories
      }
      case mem: DefSeqMemory => mem.t match {
        case _: Element => Seq(mem.id)
        case _ => Nil // Do not supoort aggregate type memories
      }
      case _ => Nil
    }) filterNot (x => (x.instanceName slice (0, 2)) == "T_") 
  }
}

private[iotesters] object bigIntToStr {
  def apply(x: BigInt, base: Int) = base match {
    case 2  if x < 0 => s"-0b${(-x).toString(base)}"
    case 16 if x < 0 => s"-0x${(-x).toString(base)}"
    case 2  => s"0b${x.toString(base)}"
    case 16 => s"0x${x.toString(base)}"
    case _ => x.toString(base)
  }
}

private[iotesters] object verilogToVCS {
  def apply(
    topModule: String,
    dir: java.io.File,
    vcsHarness: java.io.File
                ): ProcessBuilder = {
    val ccFlags = Seq("-I$VCS_HOME/include", "-I$dir", "-fPIC", "-std=c++11")
    val vcsFlags = Seq("-full64",
      "-quiet",
      "-timescale=1ns/1ps",
      "-debug_pp",
      s"-Mdir=$topModule.csrc",
      "+v2k", "+vpi",
      "+vcs+lic+wait",
      "+vcs+initreg+random",
      "+define+CLOCK_PERIOD=1",
      "-P", "vpi.tab",
      "-cpp", "g++", "-O2", "-LDFLAGS", "-lstdc++",
      "-CFLAGS", "\"%s\"".format(ccFlags mkString " "))
    val cmd = Seq("cd", dir.toString, "&&", "vcs") ++ vcsFlags ++ Seq(
      "-o", topModule, s"${topModule}.v", vcsHarness.toString, "vpi.cpp") mkString " "
    println(s"$cmd")
    Seq("bash", "-c", cmd)
  }
}

private[iotesters] case class BackendException(b: String)
  extends Exception(s"Unknown backend: $b. Backend shoule be firrtl, verilator, vcs, or glsim")

private[iotesters] case class TestApplicationException(exitVal: Int, lastMessage: String)
  extends RuntimeException(lastMessage)

private[iotesters] object TesterProcess {
  val processes = ArrayBuffer[Process]()

  def apply(cmd: Seq[String], logs: ArrayBuffer[String]) = synchronized {
    require(new java.io.File(cmd.head).exists, s"${cmd.head} doesn't exists")
    val processBuilder = Process(cmd mkString " ")
    val processLogger = ProcessLogger(println, logs += _) // don't log stdout
    val process = processBuilder run processLogger
    processes += process
    process
  }

  def finish(p: Process) = synchronized {
    processes -= p
    // p.destroy
  }

  def killall = synchronized {
    processes foreach (_.destroy)
    processes.clear
  }
}
