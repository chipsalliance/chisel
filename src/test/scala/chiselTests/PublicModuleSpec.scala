// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import circt.stage.ChiselStage

class PublicModuleSpec extends ChiselFlatSpec with MatchesAndOmits {

  class Qux extends RawModule

  class Baz extends RawModule with Public {
    val qux = Module(new Qux)
    override def isPublic = false
  }

  class Bar extends RawModule with Public {
    val baz = Module(new Baz)
  }

  class Foo extends RawModule {
    val bar = Module(new Bar)
  }

  "The main module" should "be marked public" in {

    matchesAndOmits(ChiselStage.emitCHIRRTL(new Foo))(
      "module Qux",
      "module Baz",
      "public module Bar",
      "public module Foo"
    )(
      "public module Qux",
      "public module Baz"
    )

  }

}
