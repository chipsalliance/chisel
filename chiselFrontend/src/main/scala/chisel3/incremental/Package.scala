// See LICENSE for license details.

package chisel3.incremental

trait Package {
  val majorVersion: Option[Int]
  val minorVersion: Option[Int]
  val uniqueVersion: Option[Long]
}

case class PackageSelection(packge: Option[(String, Option[(Int, Option[(Int, Option[Long])])])]) {
  val name: Option[String] = packge.map(_._1)
  val majorVersion: Option[Int] = packge.flatMap(_._2.map(_._1))
  val minorVersion: Option[Int] = packge.flatMap(_._2.flatMap(_._2.map(_._1)))
  val uniqueVersion: Option[Long] = packge.flatMap(_._2.flatMap(_._2.flatMap(_._2)))
}

object Package {
  type PackageSelection = Option[(String, Option[(Int, Option[(Int, Option[Int])])])]

  // <name>[.<majorVersion>[.<minorVersion>[.<uniqueVersion>]]]
  //def deserialize(str: String): Package = {
  //  str.split('.') match {
  //    case Array(name, major, minor, unique) => Package(name, Some(major.toInt), Some(minor.toInt), Some(unique.toLong))
  //  }
  //}
  implicit val packageSorting = new Ordering[Package] {
    override def compare(x: Package, y: Package): Int = ???
      /*
    {
      (x.majorVersion,y.majorVersion) match {
        case (Some(xa), Some(ya)) if xa == ya =>
      }
    }

       */
  }
}
