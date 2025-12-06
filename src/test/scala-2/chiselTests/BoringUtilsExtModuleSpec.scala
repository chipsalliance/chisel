// SPDX-License-Identifier: Apache-2.0

package chiselTests

import circt.stage.ChiselStage
import chisel3._
import chisel3.util.experimental.BoringUtils
import chisel3.testing.scalatest.FileCheck
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** Tests for BoringUtils.tapAndRead with ExtModule targets
  *
  * This test suite checks if there's a bug when using BoringUtils.tapAndRead
  * with ExtModule instances instead of BlackBox instances.
  */
class BoringUtilsExtModuleSpec extends AnyFlatSpec with Matchers with FileCheck {

  behavior.of("BoringUtils.tapAndRead with ExtModule")

  it should "work when tapping a signal from an ExtModule in a child module" in {
    trait BazIO extends experimental.BaseModule {
      val out = FlatIO{
        new Bundle {
          val clock = Output(Clock())
        }
      }
    }

    class Baz extends ExtModule with BazIO

    class Foo extends RawModule {
      val baz = Module((new Baz).asInstanceOf[experimental.BaseModule with BazIO]).out.clock
    }

    class Top extends RawModule {

      private val foo = Module(new Foo)

      private val bar = Module(new Bar(foo))
    }

    class Bar(foo: Foo) extends RawModule {
      val a = dontTouch(WireInit(BoringUtils.tapAndRead(foo.baz)))
    }

    println(ChiselStage.emitCHIRRTL(new Top))
  }

}
