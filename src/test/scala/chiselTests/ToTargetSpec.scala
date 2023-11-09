// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.properties.{Path, Property}
import circt.stage.ChiselStage

class RelativeInnerModule extends RawModule {
  val wire = Wire(Bool())
}

class RelativeMiddleModule extends RawModule {
  val inner = Module(new RelativeInnerModule())
}

class RelativeOuterRootModule extends RawModule {
  val middle = Module(new RelativeMiddleModule())

  atModuleBodyEnd {
    val reference = middle.inner.wire.toRelativeTarget(Some(this))
    val referenceOut = IO(Output(Property[Path]()))
    referenceOut := Property(Path(reference))
  }
}

class RelativeCurrentModule extends RawModule {
  val wire = Wire(Bool())

  val child = Module(new RawModule {
    override def desiredName = "Child"
    val io = IO(Output(Bool()))
  })

  val io = IO(Output(Bool()))

  atModuleBodyEnd {
    val reference1 = wire.toRelativeTarget(Some(this))
    val referenceOut1 = IO(Output(Property[Path]()))
    referenceOut1 := Property(Path(reference1))

    val reference2 = child.io.toRelativeTarget(Some(this))
    val referenceOut2 = IO(Output(Property[Path]()))
    referenceOut2 := Property(Path(reference2))

    val reference3 = io.toRelativeTarget(Some(this))
    val referenceOut3 = IO(Output(Property[Path]()))
    referenceOut3 := Property(Path(reference3))
  }
}

class RelativeOuterMiddleModule extends RawModule {
  val middle = Module(new RelativeMiddleModule())
  val reference = middle.inner.wire.toRelativeTarget(Some(middle))
  val referenceOut = IO(Output(Property[Path]()))
  referenceOut := Property(Path(reference))
}

class RelativeOuterLocalModule extends RawModule {
  val inner = Module(new RelativeInnerModule())
  val reference = inner.wire.toRelativeTarget(Some(inner))
  val referenceOut = IO(Output(Property[Path]()))
  referenceOut := Property(Path(reference))
}

class RelativeDefaultModule extends RawModule {
  val middle = Module(new RelativeMiddleModule())

  atModuleBodyEnd {
    val reference = middle.inner.wire.toRelativeTarget(None)
    val referenceOut = IO(Output(Property[Path]()))
    referenceOut := Property(Path(reference))
  }
}

class RelativeSiblingsModule extends RawModule {
  val middle1 = Module(new RelativeMiddleModule())
  val middle2 = Module(new RelativeMiddleModule())

  atModuleBodyEnd {
    val reference = middle1.inner.wire.toRelativeTarget(Some(middle2))
  }
}

class ToTargetSpec extends ChiselFlatSpec with Utils {

  var m: InstanceNameModule = _
  ChiselStage.emitCHIRRTL { m = new InstanceNameModule; m }

  val mn = "InstanceNameModule"
  val top = s"~$mn|$mn"

  behavior.of(".toTarget")

  val deprecationMsg = "Accessing the .instanceName or .toTarget of non-hardware Data is deprecated"

  it should "work with module IO" in {
    val io = m.io.toTarget.toString
    assert(io == s"$top>io")
  }

  it should "not work for literals" in {
    a[ChiselException] shouldBe thrownBy {
      m.x.toTarget.toString
    }
  }

  it should "NOT work for non-hardware values" in {
    a[ChiselException] shouldBe thrownBy { m.y.toTarget }
    a[ChiselException] shouldBe thrownBy { m.z.toTarget }
  }

  it should "NOT work for non-hardware bundle elements" in {
    a[ChiselException] shouldBe thrownBy { m.z.foo.toTarget }
  }

  it should "work with modules" in {
    val q = m.q.toTarget.toString
    assert(q == s"~$mn|Queue4_UInt32")
  }

  it should "error on non-hardware types and provide information" in {
    class Example extends Module {
      val tpe = UInt(8.W)

      val in = IO(Input(tpe))
      val out = IO(Output(tpe))
      out := in
    }

    val e = the[ChiselException] thrownBy extractCause[ChiselException] {
      var e: Example = null
      circt.stage.ChiselStage.emitCHIRRTL { e = new Example; e }
      e.tpe.toTarget
    }
    e.getMessage should include(
      "You cannot access the .instanceName or .toTarget of non-hardware Data: 'tpe', in module 'Example'"
    )
  }

  behavior.of(".toRelativeTarget")

  it should "work relative to modules being elaborated within atModuleBodyEnd" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RelativeOuterRootModule)

    chirrtl should include(
      "~RelativeOuterRootModule|RelativeOuterRootModule/middle:RelativeMiddleModule/inner:RelativeInnerModule>wire"
    )
  }

  it should "work relative to modules being elaborated for HasIds within the module" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RelativeCurrentModule)

    chirrtl should include("~RelativeCurrentModule|RelativeCurrentModule>wire")
    chirrtl should include("~RelativeCurrentModule|RelativeCurrentModule/child:Child>io")
    chirrtl should include("~RelativeCurrentModule|RelativeCurrentModule>io")
  }

  it should "work relative to non top-level modules that have been elaborated" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RelativeOuterMiddleModule)

    chirrtl should include("~RelativeOuterMiddleModule|RelativeMiddleModule/inner:RelativeInnerModule>wire")
  }

  it should "work relative to non top-level modules for components local to the root" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RelativeOuterLocalModule)

    chirrtl should include("~RelativeOuterLocalModule|RelativeInnerModule>wire")
  }

  it should "default to the root module in the requested hierarchy" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RelativeDefaultModule)

    chirrtl should include(
      "~RelativeDefaultModule|RelativeDefaultModule/middle:RelativeMiddleModule/inner:RelativeInnerModule>wire"
    )
  }

  it should "raise an exception when the requested root is not an ancestor" in {
    val e = the[ChiselException] thrownBy {
      ChiselStage.emitCHIRRTL(new RelativeSiblingsModule)
    }

    (e.getMessage should include).regex("Requested .toRelativeTarget relative to .+, but it is not an ancestor")
  }
}
