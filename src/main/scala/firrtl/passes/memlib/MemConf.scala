// See LICENSE for license details.

package firrtl.passes
package memlib


sealed abstract class MemPort(val name: String) { override def toString = name }

case object ReadPort extends MemPort("read")
case object WritePort extends MemPort("write")
case object MaskedWritePort extends MemPort("mwrite")
case object ReadWritePort extends MemPort("rw")
case object MaskedReadWritePort extends MemPort("mrw")

object MemPort {

  val all = Set(ReadPort, WritePort, MaskedWritePort, ReadWritePort, MaskedReadWritePort)

  def apply(s: String): Option[MemPort] = MemPort.all.find(_.name == s)

  def fromString(s: String): Map[MemPort, Int] = {
    s.split(",").toSeq.map(MemPort.apply).map(_ match {
      case Some(x) => x
      case _ => throw new Exception(s"Error parsing MemPort string : ${s}")
    }).groupBy(identity).mapValues(_.size)
  }
}

case class MemConf(
  name: String,
  depth: BigInt,
  width: Int,
  ports: Map[MemPort, Int],
  maskGranularity: Option[Int]
) {

  private def portsStr = ports.map { case (port, num) => Seq.fill(num)(port.name).mkString(",") } mkString (",")
  private def maskGranStr = maskGranularity.map((p) => s"mask_gran $p").getOrElse("")

  // Assert that all of the entries in the port map are greater than zero to make it easier to compare two of these case classes
  // (otherwise an entry of XYZPort -> 0 would not be equivalent to another with no XYZPort despite being semantically the same)
  ports.foreach { case (k, v) => require(v > 0, "Cannot have negative or zero entry in the port map") }

  override def toString = s"name ${name} depth ${depth} width ${width} ports ${portsStr} ${maskGranStr} \n"
}

object MemConf {

  val regex = raw"\s*name\s+(\w+)\s+depth\s+(\d+)\s+width\s+(\d+)\s+ports\s+([^\s]+)\s+(?:mask_gran\s+(\d+))?\s*".r

  def fromString(s: String): Seq[MemConf] = {
    s.split("\n").toSeq.map(_ match {
      case MemConf.regex(name, depth, width, ports, maskGran) => Some(MemConf(name, depth.toInt, width.toInt, MemPort.fromString(ports), Option(maskGran).map(_.toInt)))
      case "" => None
      case _ => throw new Exception(s"Error parsing MemConf string : ${s}")
    }).flatten
  }

  def apply(name: String, depth: BigInt, width: Int, readPorts: Int, writePorts: Int, readWritePorts: Int, maskGranularity: Option[Int]): MemConf = {
    val ports: Map[MemPort, Int] = (if (maskGranularity.isEmpty) {
      (if (writePorts == 0) Map.empty[MemPort, Int] else Map(WritePort -> writePorts)) ++
      (if (readWritePorts == 0) Map.empty[MemPort, Int] else Map(ReadWritePort -> readWritePorts))
    } else {
      (if (writePorts == 0) Map.empty[MemPort, Int] else Map(MaskedWritePort -> writePorts)) ++
      (if (readWritePorts == 0) Map.empty[MemPort, Int] else Map(MaskedReadWritePort -> readWritePorts))
    }) ++ (if (readPorts == 0) Map.empty[MemPort, Int] else Map(ReadPort -> readPorts))
    return new MemConf(name, depth, width, ports, maskGranularity)
  }
}
