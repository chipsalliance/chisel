// See LICENSE for license details.

package firrtlTests

class MemSpec extends FirrtlPropSpec {

  property("Zero-ported mems should be supported!") {
    runFirrtlTest("ZeroPortMem", "/features")
  }

  property("Mems with zero-width elements should be supported!") {
    runFirrtlTest("ZeroWidthMem", "/features")
  }
}

