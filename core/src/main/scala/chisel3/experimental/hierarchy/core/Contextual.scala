package chisel3.experimental.hierarchy.core

import java.util.IdentityHashMap

trait Contextual[V, +P] {
  def proto: Contextual[V, P]
  def protoParent: P
  def compute[X](hierarchy: Hierarchy[X]): Option[V]
  //def edit(f: V => V): EditValue[V, P]
  //def broaden(h: Hierarchy[P]): BroadenedValue[V, P]
  //def narrow(h: Hierarchy[P]): NarrowedValue[V, P]
  //def merge(other: Contextual[V, P]): MergedValue[V, P]
}

trait PrimitiveContextual[V, +P] extends Contextual[V, P] {
  override def proto = this
}
trait ComputedContextual[V, +P] extends Contextual[V, P] {
  def protoParent: P = proto.protoParent
}
final class EmptyValue[V, +P](val protoParent: P) extends PrimitiveContextual[V, P] {
  def compute[X](hierarchy: Hierarchy[X]): Option[V] = None
}
final class DefaultValue[V, +P](val value: V, val protoParent: P) extends PrimitiveContextual[V, P] {
  def compute[X](hierarchy: Hierarchy[X]): Option[V] = {
    require(protoParent == hierarchy.proto)
    Some(value)
  }
}

final class EditValue[V, +P](val f: V => V, value: Contextual[V, P]) extends ComputedContextual[V, P] {
  def proto = this
  def compute[X](hierarchy: Hierarchy[X]): Option[V] = value.compute(hierarchy).map(f)
}

final class BroadenedValue[V, +P](val context: Hierarchy[P], value: Contextual[V, P]) extends ComputedContextual[V, P] {
  require(context.proto == value.protoParent)
  def proto = value.proto
  def compute[X](hierarchy: Hierarchy[X]): Option[V] = {
    require(hierarchy.proto == context.proto)
    if(context.isNarrowerOrEquivalentTo(hierarchy)) value.compute(hierarchy) else None
  }
}

final class NarrowedValue[V, +P](val context: Hierarchy[P], value: Contextual[V, P]) extends ComputedContextual[V, P] {
  require(context.proto == value.protoParent)
  def proto = value.proto
  def compute[X](hierarchy: Hierarchy[X]): Option[V] = {
    require(hierarchy.proto == context.proto)
    if(hierarchy.isNarrowerOrEquivalentTo(context)) value.compute(hierarchy) else None
  }
}

final class MergedValue[V, +P](val contextuals: Set[Contextual[_, _]]) extends ComputedContextual[V, P] {
  def proto = contextuals.head.proto.asInstanceOf[Contextual[V, P]]
  require(contextuals.forall(c => c.proto == proto), s"Cannot create a MergedValue with contextuals which don't share a proto")

  def compute[X](hierarchy: Hierarchy[X]): Option[V] = {
    val newValues = contextuals.flatMap(_.compute(hierarchy).map(_.asInstanceOf[V]))
    require(newValues.size <= 1, s"Not enough context to determine proper contextual value")
    newValues.headOption
  }
  def merge[X](other: Contextual[V, X]): MergedValue[V, P] = {
    require(other.proto == proto && other.protoParent == protoParent)
    other match {
      case o: MergedValue[V, P] => new MergedValue(contextuals ++ o.contextuals)
      case other: Contextual[V, P] => new MergedValue(contextuals + other)
    }
  }
}

object Contextual {
  def apply[V, P](value: V)(implicit contextualInstancer: ContextualInstancer[V, P]): Contextual[V, P] = contextualInstancer(value)
}

/*
Sleepless night thoughts

1. Think of chaining contextual functions, rather that representing the solution
   within the contextual. this is important because you never know the answer; it
   always depends on who is asking. thus, you chain it so that once you ahve a
   definite query, you can recurse asking/looking up the "narrowerProxy" or predecessing
   contextuals
2. I'm missing a base case in my recursion, where 'this' is passed as an argument to a child.
   If we are querying a base module which has no parent, we need to try looking it up in
   the lineage of the hierarchy
3. If we address this, it may help with the merging function, where we don't really need to
   add a hierarchy layer, but instead we can just keep the raw contextual. more thought needed
   here.
4. Support contextual.edit first, to demonstrate chaining of contextuals.
5. My previous thought of holding contextual values paired with hierarchy is analogous
   to my previous error of solving the recursion incorrectly when looking up a non-local instance.
6. Perhaps contexts are the right datastructure to pair hierarchy with values in merged contextuals.
 */
