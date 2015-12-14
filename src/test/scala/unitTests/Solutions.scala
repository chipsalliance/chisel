package unitTests

import Chisel.testers.{UnitTestRunners, UnitTester}
import unitTests.{Adder, AdderTests}

/**
 * Created by chick on 12/13/15.
 */
object Solutions extends UnitTestRunners {
  def main(args: Array[String]) {
    execute( { new AdderTests })
  }
}
