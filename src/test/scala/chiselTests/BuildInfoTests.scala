package chiselTests

import org.scalatest.FlatSpec

//import Chisel._

class BuildInfoTests extends FlatSpec {
  behavior of "BuildInfoTests"

  it should "provide correct BuildInfo" in {
    println(Chisel.Driver.chiselVersionString)
  }
}
