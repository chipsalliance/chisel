// See LICENSE for license details.
package chiselTests

import org.scalatest.{FlatSpec, Matchers}

import chisel3._
import chisel3.experimental.RawModule
import chisel3.util.TransitName

import firrtl.FirrtlExecutionSuccess

class TransitNameSpec extends FlatSpec with Matchers {

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

    Driver.execute(Array("-X", "high", "--target-dir", "test_run_dir/TransitNameSpec"), () => new Top) match {
      case ChiselExecutionSuccess(_,_,Some(FirrtlExecutionSuccess(_,a))) =>
        info("""output FIRRTL includes "inst MyModule"""")
        a should include ("inst MyModule of MyModule")

        info("""output FIRRTL includes "inst bar"""")
        a should include ("inst bar of MyModule")

        info("""output FIRRTL includes "inst baz_generated"""")
        a should include ("inst baz_generated of MyModule")
      case _ => fail
    }
  }

}
