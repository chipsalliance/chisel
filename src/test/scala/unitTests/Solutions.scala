package unitTests

import Chisel.testers.{UnitTestRunners, UnitTester}

/**
 * Created by chick on 12/13/15.
 */
object Solutions extends UnitTestRunners {
  def main(args: Array[String]) {
    execute( { new DecoupledRealGCDTester })
//    execute( { new RealGCDTests })
//    execute( { new AdderTests })
  }
}
