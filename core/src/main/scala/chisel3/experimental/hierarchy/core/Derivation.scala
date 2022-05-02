package chisel3.experimental.hierarchy.core

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
    val pViewedFromH = h._lookup(_ => p.toContextual)
    pViewedFromH.proxy.compute(h).map(f.applyAny)
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
