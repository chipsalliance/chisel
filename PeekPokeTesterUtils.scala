// See LICENSE for license details.

package chisel3.iotesters

import chisel3._
import scala.sys.process._
import scala.collection.mutable.{ArrayBuffer, HashMap}

private[iotesters] object getDataNames {
  def apply(name: String, data: Data): Seq[(Data, String)] = data match {
    case b: Element => Seq(b -> name)
    case b: Bundle => b.elements.toSeq flatMap {case (n, e) => apply(s"${name}_$n", e)}
    case v: Vec[_] => v.zipWithIndex flatMap {case (e, i) => apply(s"${name}_$i", e)}
  }
  def apply(dut: Module): Seq[(Data, String)] = apply("io", dut.io)
}

private[iotesters] object getPorts {
  def apply(dut: Module) = getDataNames(dut).unzip._1 partition (_.dir == INPUT)
}

private[iotesters] object validName {
  def apply(name: String) =
    if (firrtl.Utils.v_keywords contains name) name + "$" else name
}

private[iotesters] object CircuitGraph {
  import internal.HasId
  import internal.firrtl._
  private val _modParent = HashMap[Module, Module]()
  private val _nodeParent = HashMap[HasId, Module]()
  private val _modToName = HashMap[Module, String]()
  private val _nodeToName = HashMap[HasId, String]()
  private val _nodes = ArrayBuffer[HasId]()

  private def construct(modN: String, components: Seq[Component]): Module = {
    val component = (components find (_.name == modN)).get
    val mod = component.id

    getDataNames(mod) foreach {case (port, name) =>
      // _nodes += port
      _nodeParent(port) = mod
      _nodeToName(port) = validName(name)
    }

    component.commands foreach {
      case inst: DefInstance =>
        val child = construct(validName(inst.id.name), components)
        _modParent(child) = mod
        _modToName(child) = inst.name
      case reg: DefReg if reg.name.slice(0, 2) != "T_" =>
        getDataNames(reg.name, reg.id) foreach { case (data, name) =>
          _nodes += data
          _nodeParent(data) = mod
          _nodeToName(data) = validName(name)
        }
      case reg: DefRegInit if reg.name.slice(0, 2) != "T_" =>
        getDataNames(reg.name, reg.id) foreach { case (data, name) =>
          _nodes += data
          _nodeParent(data) = mod
          _nodeToName(data) = validName(name)
        }
      case wire: DefWire if wire.name.slice(0, 2) != "T_" =>
        getDataNames(wire.name, wire.id) foreach { case (data, name) =>
          // _nodes += data
          _nodeParent(data) = mod
          _nodeToName(data) = validName(name)
        }
      case prim: DefPrim[_] if prim.name.slice(0, 2) != "T_" =>
        getDataNames(prim.name, prim.id) foreach { case (data, name) =>
          // _nodes += data
          _nodeParent(data) = mod
          _nodeToName(data) = validName(name)
        }
      case mem: DefMemory if mem.name.slice(0, 2) != "T_" => mem.t match {
        case _: Bits =>
          _nodes += mem.id
          _nodeParent(mem.id) = mod
          _nodeToName(mem.id) = validName(mem.name)
        case _ => // Do not supoort aggregate type memories
      }
      case mem: DefSeqMemory if mem.name.slice(0, 2) != "T_" => mem.t match {
        case _: Bits =>
          _nodes += mem.id
          _nodeParent(mem.id) = mod
          _nodeToName(mem.id) = validName(mem.name)
        case _ => // Do not supoort aggregate type memories
      }
      case _ =>
    }
    mod
  }

  def construct(circuit: Circuit): Module =
    construct(circuit.name, circuit.components)
  
  def nodes = _nodes.toList

  def getName(node: HasId) = _nodeToName(node)

  def getPathName(mod: Module, seperator: String): String = {
    val modName = _modToName getOrElse (mod, mod.name)
    (_modParent get mod) match {
      case None    => modName
      case Some(p) => s"${getPathName(p, seperator)}$seperator$modName"
    }
  }

  def getPathName(node: HasId, seperator: String): String = {
    (_nodeParent get node) match {
      case None    => getName(node)
      case Some(p) => s"${getPathName(p, seperator)}$seperator${getName(node)}"
    }
  }

  def getParentPathName(node: HasId, seperator: String): String = {
    (_nodeParent get node) match {
      case None    => ""
      case Some(p) => getPathName(p, seperator)
    }
  }

  def clear {
    _modParent.clear
    _nodeParent.clear
    _modToName.clear
    _nodeToName.clear
    _nodes.clear
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

private[iotesters] case class TestApplicationException(exitVal: Int, lastMessage: String) extends RuntimeException(lastMessage)
