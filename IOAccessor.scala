// See LICENSE for license details.

package chisel3.iotesters

import chisel3._
import chisel3.util._

import scala.collection.mutable
import scala.util.matching.Regex

/**
 * named access and type information about the IO bundle of a module
 * used for building testing harnesses
 */
class IOAccessor(val device_io: Record, verbose: Boolean = true) {
  val ports_referenced = new mutable.HashSet[Data]

  val dut_inputs                 = new mutable.HashSet[Data]()
  val dut_outputs                = new mutable.HashSet[Data]()

  val name_to_decoupled_port = new mutable.HashMap[String, DecoupledIO[Data]]()
  val name_to_valid_port     = new mutable.HashMap[String, ValidIO[Data]]()

  val port_to_name = {
    val port_to_name_accumulator = new mutable.HashMap[Data, String]()

    def checkDecoupledOrValid(port: Data, name: String): Unit = {
      port match {
        case decoupled_port : DecoupledIO[Data] =>
          name_to_decoupled_port(name) = decoupled_port
        case valid_port : ValidIO[Data] =>
          name_to_valid_port(name) = valid_port
        case _ =>
      }
    }

    def add_to_ports_by_direction(port: Data): Unit = {
      port match {
        case e: Element =>
          e.dir match {
            case INPUT => dut_inputs += port
            case OUTPUT => dut_outputs += port
            case _ =>
          }
        case _ =>
      }
    }

    def parseBundle(b: Record, name: String = ""): Unit = {
      for ((n, e) <- b.elements) {
        val new_name = name + (if(name.length > 0 ) "." else "" ) + n
        port_to_name_accumulator(e) = new_name
        add_to_ports_by_direction(e)

        e match {
          case bb: Record     => parseBundle(bb, new_name)
          case vv: Vec[_]  => parseVecs(vv, new_name)
          case ee: Element    =>
          case _              =>
            throw new Exception(s"bad bundle member $new_name $e")
        }
        checkDecoupledOrValid(e, new_name)
      }
    }
    def parseVecs[T<:Data](b: Vec[T], name: String = ""): Unit = {
      for ((e, i) <- b.zipWithIndex) {
        val new_name = name + s"($i)"
        port_to_name_accumulator(e) = new_name
        add_to_ports_by_direction(e)

        e match {
          case bb: Record     => parseBundle(bb, new_name)
          case vv: Vec[_]  => parseVecs(vv, new_name)
          case ee: Element    =>
          case _              =>
            throw new Exception(s"bad bundle member $new_name $e")
        }
        checkDecoupledOrValid(e, new_name)
      }
    }

    parseBundle(device_io)
    port_to_name_accumulator
  }
  val name_to_port = port_to_name.map(_.swap)

  //noinspection ScalaStyle
  def showPorts(pattern : Regex): Unit = {
    def orderPorts(a: Data, b: Data) : Boolean = {
      port_to_name(a) < port_to_name(b)
    }
    def showDecoupledCode(port_name:String): String = {
      if(name_to_decoupled_port.contains(port_name)) "D"
      else if(name_to_valid_port.contains(port_name)) "V"
      else if(findParentDecoupledPortName(port_name).nonEmpty) "D."
      else if(findParentValidPortName(port_name).nonEmpty) "V."
      else ""

    }
    def showDecoupledParent(port_name:String): String = {
      findParentDecoupledPortName(port_name) match {
        case Some(decoupled_name) => s"$decoupled_name"
        case _                    => findParentValidPortName(port_name).getOrElse("")
      }
    }
    def show_dir(dir: Direction) = dir match {
      case INPUT  => "I"
      case OUTPUT => "O"
      case _      => "-"
    }

    println("=" * 80)
    println("Device under test: io bundle")
    println("%3s  %3s  %-4s  %4s   %-25s %s".format(
            "#", "Dir", "D/V", "Used", "Name", "Parent"
    ))
    println("-" * 80)

    for((port,index) <- port_to_name.keys.toList.sortWith(orderPorts).zipWithIndex) {
      val dir = port match {
        case e: Element => show_dir(e.dir)
        case _ => "-"
      }
      val port_name = port_to_name(port)
      println("%3d  %3s   %-4s%4s    %-25s %s".format(
        index,
        dir,
        showDecoupledCode(port_name),
        if(ports_referenced.contains(port)) "y" else "",
        port_name,
        showDecoupledParent(port_name)
      ))
    }
    if(verbose) {
      println("=" * 80)
    }
  }

  def findParentDecoupledPortName(name: String): Option[String] = {
    val possible_parents = name_to_decoupled_port.keys.toList.filter(s => name.startsWith(s))
    if(possible_parents.isEmpty) return None
    possible_parents.sorted.lastOption
  }
  def findParentValidPortName(name: String): Option[String] = {
    val possible_parents = name_to_valid_port.keys.toList.filter(s => name.startsWith(s))
    if(possible_parents.isEmpty) return None
    possible_parents.sorted.lastOption
  }

  def contains(port: Data) : Boolean = {
    ports_referenced.contains(port)
  }
}
