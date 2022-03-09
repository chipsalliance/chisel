package chisel3.experimental.hierarchy.core

import java.util.IdentityHashMap

// Wrapper Class
final case class Contextual[+V](value: V)

final case class Edit[V](c: Contextual[V], f: V => V)

//final case class Context(map: IdentityHashMap[Contextual[Any], Edit[Any]])
final case class AllEdits(ls: List[Edit[Any]])
object AllEdits {
  def empty = AllEdits(Nil)//new IdentityHashMap[Contextual[Any], Edit[Any]]()
}

// Typeclass Trait
trait Contextualizer[V]  {
  type R
  def apply[P](value: V, hierarchy: Hierarchy[P]): R
}

// Default Typeclass Implementations
object Contextualizer {
  //implicit def isLookupable[L <: IsLookupable] = new Contextualizer[L] {
  //  type R = L
  //  def apply[P](v: L, hierarchy: Hierarchy[P]): L = v
  //}
  //implicit def contextual[V] = new Contextualizer[Contextual[V]] {
  //  type R = V
  //  def apply[P](v: Contextual[V], hierarchy: Hierarchy[P]): V = {
  //    hierarchy.proxy match {
  //      case Proto(p: IsContext, _) => p.edit(v)
  //      case StandIn(i: IsContext) => i.edit(v)
  //      case other => v.value
  //    }
  //  }
  //}
}
