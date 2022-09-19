package chisel3.experimental.hierarchy.core
import Contextual.log

trait Derivation {
  implicit val mg = Lookupable.mg
  def compute[H](h: Hierarchy[H]): Option[Any]
}

trait DefinitiveDerivation extends Derivation {
  final def compute[H](h: Hierarchy[H]): Option[Any] = compute
  def compute: Option[Any]
}

trait ContextualDerivation extends Derivation

case class ContextualToContextualDerivation[V](p: ContextualProxy[V], f: ParameterFunction)
    extends ContextualDerivation {
  def compute[H](h: Hierarchy[H]): Option[Any] = {
    implicit val mg = Lookupable.mg
    log(s"entering c2c, context=${h.debug}, input=${p.debug}")
    val pViewedFromH = h._lookup{_ => 
      val y = p.toContextual
      log(s"In Lookup.that: ${y.debug}")
      y
    }
    //require(false, "never finished C2C")
    log(s"middle c2c: context=${h.debug}, input=${p.debug}, contextP=${pViewedFromH.debug}")
    val ret = pViewedFromH.proxy.compute(h).map(f.applyAny)
    log(s"returning c2c: context=${h.debug}, input=${p.debug}, output=$ret")
    ret
  }
}

case class DefinitiveToDefinitiveDerivation[I, O](p: DefinitiveProxy[I], f: ParameterFunction)
    extends DefinitiveDerivation {
  def compute: Option[Any] = {
    p.compute.map(f.applyAny)
  }
}

case class ContextualToDefinitiveDerivation[I, O](p: ContextualProxy[I], f: CombinerFunction)
    extends DefinitiveDerivation {
  def compute: Option[Any] = {
    if (p.isResolved) Some(f.applyAny(p.values)) else None
  }
}
