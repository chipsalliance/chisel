// See LICENSE for license details.
package chiselTests


import chisel3._
import chisel3.stage.{ChiselGeneratorAnnotation, ChiselStage}
import chisel3.util.TransitName

import firrtl.FirrtlExecutionSuccess
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class TransitNameSpec extends AnyFlatSpec with Matchers {

  class MyModule extends RawModule {
    val io = IO(new Bundle{})
    override val desiredName: String = "MyModule"
  }

  /** A top-level module that instantiates three copies of MyModule */
  class Top extends RawModule {

    /* Assign the IO of a new MyModule instance to value "foo". The instance will be named "MyModule". */
    val foo = Module(new MyModule).io

    /* Assign the IO of a new MyModule instance to value "bar". The instance will be named "bar". */
    val bar = {
      val x = Module(new MyModule)
      TransitName(x.io, x) // TransitName returns the first argument
    }

    /* Assign the IO of a new MyModule instance to value "baz". The instance will be named "baz_generated". */
    val baz = {
      val x = Module(new MyModule)
      TransitName.withSuffix("_generated")(x.io, x) // TransitName returns the first argument
    }

  }

  it should "transit a name" in {

    val firrtl = (new ChiselStage)
      .emitFirrtl(new Top, Array("--target-dir", "test_run_dir/TransitNameSpec"))

    info("""output FIRRTL includes "inst MyModule"""")
    firrtl should include ("inst MyModule of MyModule")

    info("""output FIRRTL includes "inst bar"""")
    firrtl should include ("inst bar of MyModule")

    info("""output FIRRTL includes "inst baz_generated"""")
    firrtl should include ("inst baz_generated of MyModule")
  }

}
