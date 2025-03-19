// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.experimental.BundleLiterals._
import chisel3.experimental.SourceInfo
import chisel3.experimental.VecLiterals._
import chisel3.experimental.dataview._
import chisel3.probe._
import chisel3.properties._
import chisel3.util.Counter
import circt.stage.ChiselStage
import org.scalactic.source.Position
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

object ReadOnlySpec {
  class SimpleBundle extends Bundle {
    val a = UInt(8.W)
  }
  object SimpleBundle {
    implicit val toOther: DataView[SimpleBundle, OtherBundle] = DataView(
      _ => new OtherBundle,
      _.a -> _.foo
    )
    implicit val toBigger: DataView[(SimpleBundle, UInt), BiggerBundle] = DataView(
      _ => new BiggerBundle,
      _._1.a -> _.fizz,
      _._2 -> _.buzz
    )
  }
  class OtherBundle extends Bundle {
    val foo = UInt(8.W)
  }
  class BiggerBundle extends Bundle {
    val fizz = UInt(8.W)
    val buzz = UInt(8.W)
  }

  class BidirectionalBundle extends Bundle {
    val foo = UInt(8.W)
    val bar = Flipped(UInt(8.W))
  }
  object BidirectionalBundle {
    implicit val fromTuple: DataView[(UInt, UInt), BidirectionalBundle] = DataView(
      _ => new BidirectionalBundle,
      _._1 -> _.foo,
      _._2 -> _.bar
    )
  }

  class ProbeBundle extends Bundle {
    val probe = Probe(UInt(8.W))
    val rwProbe = RWProbe(UInt(8.W))
    val field = UInt(8.W)
  }
  object ProbeBundle {
    implicit val fromTuple: DataView[(UInt, UInt, UInt), ProbeBundle] = DataView(
      _ => new ProbeBundle,
      _._1 -> _.probe,
      _._2 -> _.rwProbe,
      _._3 -> _.field
    )
  }

  class PropertyBundle extends Bundle {
    val field = UInt(8.W)
    val prop = Property[String]()
  }
  object PropertyBundle {
    implicit val fromTuple: DataView[(UInt, Property[String]), PropertyBundle] = DataView(
      _ => new PropertyBundle,
      _._1 -> _.field,
      _._2 -> _.prop
    )
  }
}

class ReadOnlySpec extends AnyFlatSpec with Matchers {
  import ReadOnlySpec._

  behavior.of("readOnly")

  def check(m: => RawModule)(implicit pos: Position): Unit = {
    val e = the[ChiselException] thrownBy {
      ChiselStage.elaborate(m, Array("--throw-on-first-error"))
    }
    e.getMessage should include("Cannot connect to read-only value")
  }

  // Probes don't support '<>'
  def rightToLeftConnectOpsNoLegacyBulkConnect(implicit info: SourceInfo): Seq[(Data, Data) => Unit] = Seq(
    _ := _,
    _ :<= _,
    _ :#= _,
    _ :<>= _
  )

  def rightToLeftConnectOps(implicit info: SourceInfo): Seq[(Data, Data) => Unit] =
    rightToLeftConnectOpsNoLegacyBulkConnect :+ (_ <> _)

  def leftToRightConnectOps(implicit info: SourceInfo): Seq[(Data, Data) => Unit] = Seq(
    _ :>= _,
    _ :<>= _,
    _ <> _
  )

  it should "be an error to connect to a read-only UInt" in {
    for (op <- rightToLeftConnectOps) {
      check(new RawModule {
        val foo = IO(UInt(8.W))
        op(foo.readOnly, 1.U)
      })
    }
  }

  it should "be an error to connect to a field of a read-only Bundle" in {
    for (op <- rightToLeftConnectOps) {
      check(new RawModule {
        val foo = IO(new SimpleBundle)
        val bar = foo.readOnly
        op(bar.a, 1.U)
      })
    }
  }

  it should "be an error to connect to a read-only Bundle" in {
    for (op <- rightToLeftConnectOps) {
      check(new RawModule {
        val in = IO(Input(new SimpleBundle))
        val foo = IO(new SimpleBundle)
        op(foo.readOnly, in)
      })
    }
  }

  it should "be an error to connect to a read-only view of a UInt" in {
    for (op <- rightToLeftConnectOps) {
      check(new RawModule {
        val foo = IO(UInt(8.W))
        val x = foo.viewAs[UInt].readOnly
        op(x, 1.U)
      })
    }
  }

  it should "be an error to connect to a field of a read-only view of a Bundle" in {
    for (op <- rightToLeftConnectOps) {
      check(new RawModule {
        val out = IO(new SimpleBundle)
        val x = out.viewAs[OtherBundle].readOnly
        op(x.foo, 1.U)
      })
    }
  }

  it should "be an error to connect to a read-only view of a Bundle" in {
    for (op <- rightToLeftConnectOps) {
      check(new RawModule {
        val in = IO(Input(new OtherBundle))
        val out = IO(new SimpleBundle)
        val x = out.viewAs[OtherBundle].readOnly
        op(x, in)
      })
    }
  }

  it should "be an error to connect to a field of a view of a read-only Bundle" in {
    for (op <- rightToLeftConnectOps) {
      check(new RawModule {
        val out = IO(new SimpleBundle)
        val x = out.readOnly.viewAs[OtherBundle]
        op(x.foo, 1.U)
      })
    }
  }

  it should "be an error to connect to a view of a read-only Bundle" in {
    for (op <- rightToLeftConnectOps) {
      check(new RawModule {
        val in = IO(Input(new OtherBundle))
        val out = Wire(new SimpleBundle).readOnly
        val x = out.viewAs[OtherBundle]
        op(x, in)
      })
    }
  }

  it should "be an error to connect to a field of read-only non-identity view" in {
    for (op <- rightToLeftConnectOps) {
      check(new RawModule {
        val w = IO(new SimpleBundle)
        val x = IO(UInt(8.W))
        val y = (w, x).viewAs[BiggerBundle].readOnly
        op(y.fizz, 1.U)
      })
    }
  }

  it should "be an error to connect to a read-only non-identity view" in {
    for (op <- rightToLeftConnectOps) {
      check(new RawModule {
        val in = IO(Input(new BiggerBundle))
        val w = Wire(new SimpleBundle)
        val x = Wire(UInt(8.W))
        val y = (w, x).viewAs[BiggerBundle].readOnly
        op(y, in)
      })
    }
  }

  it should "be an error to connect to a field of a non-identity view targeting a read-only Bundle" in {
    for (op <- rightToLeftConnectOps) {
      check(new RawModule {
        val w = IO(new SimpleBundle).readOnly
        val x = IO(UInt(8.W))
        val y = (w, x).viewAs[BiggerBundle]
        op(y.fizz, 1.U)
      })
    }
  }

  it should "NOT be an error to connect to a field of a non-identity view targeting a normal Bundle (where another targeted field is read-only)" in {
    for (op <- rightToLeftConnectOps) {
      val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
        val w = IO(new SimpleBundle)
        val x = IO(UInt(8.W))
        val y = (w, x.readOnly).viewAs[BiggerBundle]
        op(y.fizz, 1.U)
      })
      chirrtl should include("connect w.a, UInt<1>(0h1)")
    }
  }

  it should "be an error to connect to a non-identity view targeting a read-only Bundle (where another targeted field is normal)" in {
    for (op <- rightToLeftConnectOps) {
      check(new RawModule {
        val in = IO(Input(new BiggerBundle))
        val w = Wire(new SimpleBundle).readOnly
        val x = Wire(UInt(8.W))
        val y = (w, x).viewAs[BiggerBundle]
        op(y, in)
      })
    }
  }

  it should "be an error to bidirectionally connect to a read-only Bundle on the RHS" in {
    for (op <- leftToRightConnectOps) {
      check(new RawModule {
        val in = IO(Flipped(new BidirectionalBundle))
        val w = Wire(new BidirectionalBundle)
        op(w, in.readOnly)
      })
    }
  }

  it should "be an error to bidirectionally connect to a view containing a read-only field on the RHS" in {
    for (op <- leftToRightConnectOps) {
      check(new RawModule {
        val x, y = Wire(UInt(8.W))
        val z = (x, y.readOnly).viewAs[BidirectionalBundle]
        val out = IO(new BidirectionalBundle)
        op(out, z)
      })
      // But note that it's fine if x (not flipped) is readOnly.
      ChiselStage.elaborate(new RawModule {
        val x, y = Wire(UInt(8.W))
        val z = (x.readOnly, y).viewAs[BidirectionalBundle]
        val out = IO(new BidirectionalBundle)
        op(out, z)
      })
    }
  }

  it should "be an error to define to a read-only probe" in {
    check(new RawModule {
      val p = IO(Output(Probe(UInt(8.W))))
      define(p.readOnly, ProbeValue(8.U(8.W)))
    })
    check(new RawModule {
      val p = IO(Output(RWProbe(UInt(8.W))))
      val w = Wire(UInt(8.W))
      define(p.readOnly, RWProbeValue(w))
    })
  }

  it should "be an error to connect to a read-only Bundle containing a probe" in {
    for (op <- rightToLeftConnectOpsNoLegacyBulkConnect) {
      check(new RawModule {
        val w1 = Wire(new ProbeBundle)
        val w2 = Wire(new ProbeBundle).readOnly
        op(w2, w1)
      })
    }
  }

  it should "be an error to define to a probe field of a read-only Bundle" in {
    check(new RawModule {
      val w = Wire(new ProbeBundle).readOnly
      define(w.probe, ProbeValue(8.U(8.W)))
    })
    // Check the RWProbe too.
    check(new RawModule {
      val w = Wire(new ProbeBundle).readOnly
      val w2 = Wire(UInt(8.W))
      define(w.rwProbe, RWProbeValue(w2))
    })
  }

  it should "be an error to connect to a view containing a read-only probe" in {
    for (op <- rightToLeftConnectOpsNoLegacyBulkConnect) {
      check(new RawModule {
        val a = IO(Output(Probe(UInt(8.W))))
        val b = IO(Output(RWProbe(UInt(8.W))))
        val c = IO(Output(UInt(8.W)))
        val d = (a.readOnly, b, c).viewAs[ProbeBundle]
        val w = Wire(new ProbeBundle)
        op(d, w)
      })
    }
    // Check the RWProbe too.
    for (op <- rightToLeftConnectOpsNoLegacyBulkConnect) {
      check(new RawModule {
        val a = IO(Output(Probe(UInt(8.W))))
        val b = IO(Output(RWProbe(UInt(8.W))))
        val c = IO(Output(UInt(8.W)))
        val d = (a, b.readOnly, c).viewAs[ProbeBundle]
        val w = Wire(new ProbeBundle)
        op(d, w)
      })
    }
  }

  it should "be an error to connect to a read-only property" in {
    check(new RawModule {
      val p = IO(Output(Property[String]()))
      p.readOnly := Property("foo")
    })
  }

  it should "be an error to connect to a read-only Bundle containing a property" in {
    for (op <- rightToLeftConnectOps) {
      check(new RawModule {
        val w1 = Wire(new PropertyBundle)
        val w2 = IO(new PropertyBundle)
        op(w2.readOnly, w1)
      })
    }
  }

  it should "be an error to connect to a property field of a read-only Bundle" in {
    check(new RawModule {
      val w = IO(new PropertyBundle).readOnly
      w.prop := Property("foo")
    })
  }

  it should "be an error to connect to a view containing a read-only property" in {
    for (op <- rightToLeftConnectOps) {
      check(new RawModule {
        val a = IO(Output(UInt(8.W)))
        val b = IO(Output(Property[String]()))
        val x = (a, b.readOnly).viewAs[PropertyBundle]
        val w = Wire(new PropertyBundle)
        op(x, w)
      })
    }
  }

  it should "NOT create a view for literals" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val a = 123.U
      val b = -27.S
      val c = (new BiggerBundle).Lit(_.fizz -> 123.U, _.buzz -> 456.U)
      val d = Vec.Lit(0.U, 23.U)
      // Use referential equality to check they are the same objects (no view introduced)
      assert(a.readOnly eq a)
      assert(b.readOnly eq b)
      assert(c.readOnly eq c)
      assert(d.readOnly eq d)
    })
  }

  it should "NOT create a view for op results" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val a = IO(Input(UInt(8.W)))
      val b = IO(Input(new BiggerBundle))

      val x = a + 1.U
      val y = b.asUInt
      // Use referential equality to check they are the same objects (no view introduced)
      assert(x.readOnly eq x)
      assert(y.readOnly eq y)
    })
  }
}
