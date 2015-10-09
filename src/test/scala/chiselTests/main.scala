package chiselTests
import Chisel._

object MiniChisel {
  def main(gargs: Array[String]): Unit = {

    require(gargs.length > 0 && gargs(0)(0) != '-', "Need a test name argument")
    val name   = gargs(0)
    val nameIndex = 0
    val optionIndex = gargs.indexWhere { arg => arg(0) == '-' }
    val numberOfOptions = if (optionIndex == -1) {
      0
    } else {
      gargs.length - optionIndex
    }
    val numberOfNames = gargs.length - numberOfOptions
    val testNames: Array[String]  =  if (name == "all") {
      Array[String]("BundleWire", "ComplexAssign", "GCD", "MulLookup", "Stack", "Tbl")
    } else {
      gargs.slice(nameIndex, nameIndex + numberOfNames)  
    }
    val args = gargs.slice(optionIndex, optionIndex + numberOfOptions) 
    for (testName <- testNames) {
      testName match {
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
}
