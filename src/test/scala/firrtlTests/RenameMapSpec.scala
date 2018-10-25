// See LICENSE for license details.

package firrtlTests

import firrtl.RenameMap
import firrtl.FIRRTLException
import firrtl.RenameMap.{CircularRenameException, IllegalRenameException}
import firrtl.annotations._

class RenameMapSpec extends FirrtlFlatSpec {
  val cir   = CircuitTarget("Top")
  val cir2  = CircuitTarget("Pot")
  val cir3  = CircuitTarget("Cir3")
  val modA  = cir.module("A")
  val modA2 = cir2.module("A")
  val modB = cir.module("B")
  val foo = modA.ref("foo")
  val foo2 = modA2.ref("foo")
  val bar = modA.ref("bar")
  val fizz = modA.ref("fizz")
  val fooB = modB.ref("foo")
  val barB = modB.ref("bar")

  val tmb = cir.module("Top").instOf("mid", "Middle").instOf("bot", "Bottom")
  val tm2b = cir.module("Top").instOf("mid", "Middle2").instOf("bot", "Bottom")
  val middle = cir.module("Middle")
  val middle2 = cir.module("Middle2")

  behavior of "RenameMap"

  it should "return None if it does not rename something" in {
    val renames = RenameMap()
    renames.get(modA) should be (None)
    renames.get(foo) should be (None)
  }

  it should "return a Seq of renamed things if it does rename something" in {
    val renames = RenameMap()
    renames.record(foo, bar)
    renames.get(foo) should be (Some(Seq(bar)))
  }

  it should "allow something to be renamed to multiple things" in {
    val renames = RenameMap()
    renames.record(foo, bar)
    renames.record(foo, fizz)
    renames.get(foo) should be (Some(Seq(bar, fizz)))
  }

  it should "allow something to be renamed to nothing (ie. deleted)" in {
    val renames = RenameMap()
    renames.record(foo, Seq())
    renames.get(foo) should be (Some(Seq()))
  }

  it should "return None if something is renamed to itself" in {
    val renames = RenameMap()
    renames.record(foo, foo)
    renames.get(foo) should be (None)
  }

  it should "allow targets to change module" in {
    val renames = RenameMap()
    renames.record(foo, fooB)
    renames.get(foo) should be (Some(Seq(fooB)))
  }

  it should "rename targets if their module is renamed" in {
    val renames = RenameMap()
    renames.record(modA, modB)
    renames.get(foo) should be (Some(Seq(fooB)))
    renames.get(bar) should be (Some(Seq(barB)))
  }

  it should "rename renamed targets if the module of the target is renamed" in {
    val renames = RenameMap()
    renames.record(modA, modB)
    renames.record(foo, bar)
    renames.get(foo) should be (Some(Seq(barB)))
  }

  it should "rename modules if their circuit is renamed" in {
    val renames = RenameMap()
    renames.record(cir, cir2)
    renames.get(modA) should be (Some(Seq(modA2)))
  }

  it should "rename targets if their circuit is renamed" in {
    val renames = RenameMap()
    renames.record(cir, cir2)
    renames.get(foo) should be (Some(Seq(foo2)))
  }

  val TopCircuit = cir
  val Top = cir.module("Top")
  val Top_m = Top.instOf("m", "Middle")
  val Top_m_l = Top_m.instOf("l", "Leaf")
  val Top_m_l_a = Top_m_l.ref("a")
  val Top_m_la = Top_m.ref("l").field("a")
  val Middle = cir.module("Middle")
  val Middle2 = cir.module("Middle2")
  val Middle_la = Middle.ref("l").field("a")
  val Middle_l_a = Middle.instOf("l", "Leaf").ref("a")

  it should "rename targets if modules in the path are renamed" in {
    val renames = RenameMap()
    renames.record(Middle, Middle2)
    renames.get(Top_m) should be (Some(Seq(Top.instOf("m", "Middle2"))))
  }

  it should "rename targets if instance and module in the path are renamed" in {
    val renames = RenameMap()
    renames.record(Middle, Middle2)
    renames.record(Top.ref("m"), Top.ref("m2"))
    renames.get(Top_m) should be (Some(Seq(Top.instOf("m2", "Middle2"))))
  }

  it should "rename targets if instance in the path are renamed" in {
    val renames = RenameMap()
    renames.record(Top.ref("m"), Top.ref("m2"))
    renames.get(Top_m) should be (Some(Seq(Top.instOf("m2", "Middle"))))
  }

  it should "rename targets if instance and ofmodule in the path are renamed" in {
    val renames = RenameMap()
    val Top_m2 = Top.instOf("m2", "Middle2")
    renames.record(Top_m, Top_m2)
    renames.get(Top_m) should be (Some(Seq(Top_m2)))
  }

  it should "properly do nothing if no remaps" in {
    val renames = RenameMap()
    renames.get(Top_m_l_a) should be (None)
  }

  it should "properly rename if leaf is inlined" in {
    val renames = RenameMap()
    renames.record(Middle_l_a, Middle_la)
    renames.get(Top_m_l_a) should be (Some(Seq(Top_m_la)))
  }

  it should "properly rename if middle is inlined" in {
    val renames = RenameMap()
    renames.record(Top_m.ref("l"), Top.ref("m_l"))
    renames.get(Top_m_l_a) should be (Some(Seq(Top.instOf("m_l", "Leaf").ref("a"))))
  }

  it should "properly rename if leaf and middle are inlined" in {
    val renames = RenameMap()
    val inlined = Top.ref("m_l_a")
    renames.record(Top_m_l_a, inlined)
    renames.record(Top_m_l, Nil)
    renames.record(Top_m, Nil)
    renames.get(Top_m_l_a) should be (Some(Seq(inlined)))
  }

  it should "quickly rename a target with a long path" in {
    (0 until 50 by 10).foreach { endIdx =>
      val renames = RenameMap()
      renames.record(TopCircuit.module("Y0"), TopCircuit.module("X0"))
      val deepTarget = (0 until endIdx).foldLeft(Top: IsModule) { (t, idx) =>
        t.instOf("a", "A" + idx)
      }.ref("ref")
      val (millis, rename) = firrtl.Utils.time(renames.get(deepTarget))
      println(s"${(deepTarget.tokens.size - 1) / 2} -> $millis")
      //rename should be(None)
    }
  }

  it should "rename with multiple renames" in {
    val renames = RenameMap()
    val Middle2 = cir.module("Middle2")
    renames.record(Middle, Middle2)
    renames.record(Middle.ref("l"), Middle.ref("lx"))
    renames.get(Middle.ref("l")) should be (Some(Seq(Middle2.ref("lx"))))
  }

  it should "rename with fields" in {
    val Middle_o = Middle.ref("o")
    val Middle_i = Middle.ref("i")
    val Middle_o_f = Middle.ref("o").field("f")
    val Middle_i_f = Middle.ref("i").field("f")
    val renames = RenameMap()
    renames.record(Middle_o, Middle_i)
    renames.get(Middle_o_f) should be (Some(Seq(Middle_i_f)))
  }

  it should "rename instances with same ofModule" in {
    val Middle_o = Middle.ref("o")
    val Middle_i = Middle.ref("i")
    val renames = RenameMap()
    renames.record(Middle_o, Middle_i)
    renames.get(Middle.instOf("o", "O")) should be (Some(Seq(Middle.instOf("i", "O"))))
  }

  it should "detect circular renames" in {
    case class BadRename(from: IsMember, tos: Seq[IsMember])
    val badRenames =
      Seq(
        BadRename(foo, Seq(foo.field("bar"))),
        BadRename(modA, Seq(foo))
        //BadRename(cir, Seq(foo)),
        //BadRename(cir, Seq(modA))
      )
    // Run all BadRename tests
    for (BadRename(from, tos) <- badRenames) {
      val fromN = from
      val tosN = tos.mkString(", ")
      //it should s"error if a $fromN is renamed to $tosN" in {
      val renames = RenameMap()
      for (to <- tos) {
        a [IllegalArgumentException] shouldBe thrownBy {
          renames.record(from, to)
        }
      }
      //}
    }
  }

  it should "be able to rename weird stuff" in {
    // Renaming `from` to each of the `tos` at the same time should be ok
    case class BadRename(from: CompleteTarget, tos: Seq[CompleteTarget])
    val badRenames =
      Seq(//BadRename(foo, Seq(cir)),
        BadRename(foo, Seq(modB)),
        BadRename(modA, Seq(fooB)),
        //BadRename(modA, Seq(cir)),
        //BadRename(cir, Seq(foo)),
        //BadRename(cir, Seq(modA)),
        BadRename(cir, Seq(cir2, cir3))
      )
    // Run all BadRename tests
    for (BadRename(from, tos) <- badRenames) {
      val fromN = from
      val tosN = tos.mkString(", ")
      //it should s"error if a $fromN is renamed to $tosN" in {
        val renames = RenameMap()
        for (to <- tos) {
          (from, to) match {
            case (f: CircuitTarget, t: CircuitTarget) => renames.record(f, t)
            case (f: IsMember, t: IsMember) => renames.record(f, t)
          }
        }
        //a [FIRRTLException] shouldBe thrownBy {
        renames.get(from)
        //}
      //}
    }
  }

  it should "error if a circular rename occurs" in {
    val renames = RenameMap()
    val top = CircuitTarget("Top")
    renames.record(top.module("A"), top.module("B").instOf("c", "C"))
    renames.record(top.module("B"), top.module("A").instOf("c", "C"))
    a [CircularRenameException] shouldBe thrownBy {
      renames.get(top.module("A"))
    }
  }

  it should "not error if a swapping rename occurs" in {
    val renames = RenameMap()
    val top = CircuitTarget("Top")
    renames.record(top.module("A"), top.module("B"))
    renames.record(top.module("B"), top.module("A"))
    renames.get(top.module("A")) should be (Some(Seq(top.module("B"))))
    renames.get(top.module("B")) should be (Some(Seq(top.module("A"))))
  }

  it should "error if a reference is renamed to a module, and then we try to rename the reference's field" in {
    val renames = RenameMap()
    val top = CircuitTarget("Top")
    renames.record(top.module("A").ref("ref"), top.module("B"))
    renames.get(top.module("A").ref("ref")) should be(Some(Seq(top.module("B"))))
    a [IllegalRenameException] shouldBe thrownBy {
      renames.get(top.module("A").ref("ref").field("field"))
    }
    a [IllegalRenameException] shouldBe thrownBy {
      renames.get(top.module("A").instOf("ref", "R"))
    }
  }

  it should "error if we rename an instance's ofModule into a non-module" in {
    val renames = RenameMap()
    val top = CircuitTarget("Top")

    renames.record(top.module("C"), top.module("D").ref("x"))
    a [IllegalRenameException] shouldBe thrownBy {
      renames.get(top.module("A").instOf("c", "C"))
    }
  }

  it should "error if path is renamed into a non-path" ignore {
    val renames = RenameMap()
    val top = CircuitTarget("Top")

    renames.record(top.module("E").instOf("f", "F"), top.module("E").ref("g"))

    a [IllegalRenameException] shouldBe thrownBy {
      println(renames.get(top.module("E").instOf("f", "F").ref("g")))
    }
  }
}
