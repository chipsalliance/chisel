package chisel3.util.circt

/** from circt port to chisel port definition ABI. */
object Port {
  sealed trait Direction
  object In extends Direction
  object Out extends Direction
  object Inout extends Direction
}

case class Port(name: String, direction: Port.Direction, width: Int)
