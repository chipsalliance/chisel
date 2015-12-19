package Chisel.testers

import Chisel._

import scala.collection.mutable

/**
 * named access and type information about the IO bundle of a module
 * used for building testing harnesses
 */
class IOAccessor(val device_io: Bundle, verbose: Boolean = true) {
  val dut_inputs  = device_io.flatten.filter( port => port.dir == INPUT)
  val dut_outputs = device_io.flatten.filter( port => port.dir == OUTPUT)
  val ports_referenced = new mutable.HashSet[Data]

  val decoupled_ports = new mutable.ArrayBuffer[Data]()
  val name_to_decoupled_port = new mutable.HashMap[String, Data]()
  val port_to_name = {
    val port_to_name_accumulator = new mutable.HashMap[Data, String]()

    if(verbose) {
      println("=" * 80)
      println("Device under test: io bundle")
      println("%10s %10s %s".format("direction", "referenced", "name"))
      println("-" * 80)
    }

    def parse_bundle(b: Bundle, name: String = ""): Unit = {
      for ((n, e) <- b.elements) {
        val new_name = name + (if(name.length > 0 ) "." else "" ) + n
        port_to_name_accumulator(e) = new_name
        if(verbose) {
          println("%10s %5s      %s".format(e.dir, "-", new_name))
        }
        e match {
          case bb: Bundle  => parse_bundle(bb, new_name)
          case vv: Vec[_]  => parse_vecs(vv, new_name)
          case ee: Element => {}
          case _           => {
            throw new Exception(s"bad bundle member ${new_name} $e")
          }
        }

        if(e.isInstanceOf[DecoupledIO[_]]) {
          decoupled_ports += e
          name_to_decoupled_port(name) = e
        }
      }
    }
    def parse_vecs[T<:Data](b: Vec[T], name: String = ""): Unit = {
      for ((e, i) <- b.zipWithIndex) {
        val new_name = name + s"($i)"

        e match {
          case bb: Bundle  => parse_bundle(bb, new_name)
          case vv: Vec[_]  => parse_vecs(vv, new_name)
          case ee: Element => {}
          case _           => {
            throw new Exception(s"bad bundle member ${new_name} $e")
          }
        }
      }
    }

    parse_bundle(device_io)
    if(verbose) {
      println("=" * 80)
    }
    port_to_name_accumulator
  }

  /**
   * return the name of a parent in the io hierarchy which is of type decoupled
   *
   * @param name
   * @return
   */
  def find_decoupled_parent_ref(name: String) : Option[String] = {
    None
  }

  def register(port: Data): Unit = {
    ports_referenced += port
  }

  def contains(port: Data) : Boolean = {
    ports_referenced.contains(port)
  }
}
