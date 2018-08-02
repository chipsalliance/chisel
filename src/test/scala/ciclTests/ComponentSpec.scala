package ciclTests

import chiselTests.ChiselPropSpec
import chisel3.libs._


class ComponentSpec extends ChiselPropSpec {
  def check(comp: Component): Unit = {
    val named = Component.convertComponent2Named(comp)
    println(named)
    val comp2 = Component.named2component(named)
    assert(comp == comp2)
  }
  property("Should convert to/from Named") {
    check(Component(None, None, Nil, None))
    check(Component(Some("Top"), None, Nil, None))
    check(Component(None, Some("Top"), Nil, None))
    check(Component(Some("Top"), Some("Other"), Nil, None))
    val i1 = Seq(Instance("i1"), OfModule("I"))
    val i2 = Seq(Instance("i2"), OfModule("I"))
    val x = Seq(Instance("x"), OfModule("X"))
    check(Component(None, None, i1 ++ x ++ Seq(Clock), None))
    check(Component(None, None, i1 ++ x ++ Seq(Ref("ref")), None))
    check(Component(Some("Top"), Some("Other"), i1 ++ x ++ Seq(Clock), None))
    check(Component(Some("Top"), Some("Other"), i1 ++ x ++ Seq(Ref("ref")), None))
  }
  property("Should enable creating from API") {
    val top = Component(Some("Top"), Some("Top"), Nil, None)
    val x_reg0_data = top.inst("x").of("X").ref("reg0").field("data")
    top.inst("x").inst("x")
    top.ref("y")
    println(x_reg0_data)
  }
}
