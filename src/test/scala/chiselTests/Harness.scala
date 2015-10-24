package chiselTests
import Chisel.testers.TesterDriver
import org.scalatest._
import org.scalatest.prop._

class HarnessSpec extends ChiselPropSpec {
  //val tmp = System.getProperty("java.io.tmpdir")
  //val t = java.io.File.createTempFile("pre", "post")
  property("Test making verilog harnesses and executing") {
    TesterDriver.test()
  }
}
 
