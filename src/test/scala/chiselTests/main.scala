package chiselTests
import Chisel._

object MiniChisel {
  def main(gargs: Array[String]): Unit = {

    require(gargs.length > 0, "Need a test name argument")
    val name   = gargs(0)
    require(name(0) != '-', "Need a test name as the first argument")
    val args   = gargs.slice(1, gargs.length) 
    name match {
      case "BundleWire" => chiselMain(args, () => Module(new BundleWire(16)))
      case "ComplexAssign" => chiselMain(args, () => Module(new ComplexAssign(32)))
      case "GCD" => chiselMain(args, () => Module(new GCD()))
      case "MulLookup" => chiselMain(args, () => Module(new MulLookup(2)))
      case "Stack" => chiselMain(args, () => Module(new Stack(16)))
      case "Tbl" => chiselMain(args, () => Module(new Tbl(8, 42)))
      case _ => println(" skipping " + name)
    }
  }
}
