package chisel3.experimental.hierarchy.core

import java.util.IdentityHashMap

sealed trait Contextual[V] {
  implicit val mg = new chisel3.internal.MacroGenerated {}
  def proto: Contextual[V]
  def protoParent: Any
  def compute[X, Y](hierarchy: Hierarchy[X], context: Context[Y]): Option[V]

  def edit(f: V => V): EditValue[V] = new EditValue(f, this)
  //def broaden[P](h: [P], value: Contextual[V]): BroaderValue[V] = new BroaderValue(h, value, this)
  //def narrow[P](h: Hierarchy[P]): NarrowedValue[V] = new NarrowedValue(h, this)
  //def merge(other: Contextual[V]): MergedValue[V] = {
  //  require(other.proto == proto && other.protoParent == protoParent)
  //  (this, other) match {
  //    case (t: MergedValue[V], o: MergedValue[V]) => new MergedValue(t.contextuals ++ o.contextuals)
  //    case (t: MergedValue[V], o)                 => new MergedValue(t.contextuals + other)
  //    case (t,                 o: MergedValue[V]) => new MergedValue(o.contextuals + t)
  //    case (t,                 o)                 => new MergedValue(Set(t, o))
  //  }
  //}
  def serialize(n: Int): String
  def carriage(n: Int): String = "\n" + indent(n)
  def indent(n: Int): String = ("  " * n)
  def kind: String
  override def toString: String = kind + "@" + Integer.toHexString(hashCode());
}

sealed trait PrimitiveContextual[V] extends Contextual[V] {
  def proto = this
}

sealed trait ComputedContextual[V] extends Contextual[V] {
  def protoParent: Any = proto.protoParent
}
final class EmptyValue[V](val protoParent: Any) extends PrimitiveContextual[V] {
  def compute[X, Y](hierarchy: Hierarchy[X], context: Context[Y]): Option[V] = {
    require(protoParent == hierarchy.proto)
    context.getter(this) match {
       case x if x == this =>
          //println(hierarchy, context)
          None
       case other =>
          val x = other.compute(hierarchy, hierarchy.toContext)
          //println(x, other)
          x
    }
  }
  def kind: String = "<.>"
  def serialize(n: Int): String = indent(n) + super.toString
  override def toString: String = serialize(0)
}
final class DefaultValue[V](val value: V, predecessor: Contextual[V]) extends PrimitiveContextual[V] {
  def protoParent: Any = predecessor.protoParent
  def kind: String = s"<$value>"
  def compute[X, Y](hierarchy: Hierarchy[X], context: Context[Y]): Option[V] = {
    require(protoParent == hierarchy.proto)
    //println(s"Computing $this")
    val ret = context.getter(this) match {
       case x if x == this =>
          //println(s"Here: $x, $value")
          Some(value)
       case other => 
          //println(s"There: ${hierarchy.proxy.getClass.getSimpleName}, $other, $this")
          //???
          other.compute(hierarchy, hierarchy.toContext)
    }
    //println(s"Returning $ret from $this")
    ret
  }
  def serialize(n: Int): String = indent(n) + super.toString + s"(\n${predecessor.serialize(n+ 1)}" + carriage(n) + ")"
  override def toString: String = serialize(0)
}

final class EditValue[V](val f: V => V, predecessor: Contextual[V]) extends PrimitiveContextual[V] {
  def protoParent: Any = predecessor.protoParent
  def kind: String = s"<$f>"
  def compute[X, Y](hierarchy: Hierarchy[X], context: Context[Y]): Option[V] = {
    context.getter(this) match {
       case x if x == this =>
         //println(s"This: $this")
         val ret = predecessor.compute(hierarchy, hierarchy.toContext).map(f)
         //println(s"RET: $ret")
         ret
       case other =>
         //???
         other.compute(hierarchy, hierarchy.toContext)
    }
   }
  def serialize(n: Int): String = indent(n) + super.toString + s"\n${predecessor.serialize(n+ 1)}" + carriage(n) + ")"
  override def toString: String = serialize(0)
}

final class ContextualEdit[V](val f: V => V, val des: String, predecessor: Contextual[V]) extends ComputedContextual[V] {
  def proto = predecessor.proto
  def kind: String = s"<e%$des%>"
  def compute[X, Y](hierarchy: Hierarchy[X], context: Context[Y]): Option[V] = {
    //println(s"Computing $this")
    val ret = predecessor.compute(hierarchy, context).map(f)
    //println(s"Returning $ret from $this")
    ret
  }
  def serialize(n: Int): String = indent(n) + super.toString + s"(\n${predecessor.serialize(n+1)}" + carriage(n) + ")"
  override def toString: String = serialize(0)
}

final class ContextualValue[V](val v: V, val des: String, predecessor: Contextual[V]) extends ComputedContextual[V] {
  def proto = predecessor.proto
  def kind: String = s"<v%$des%>"
  def compute[X, Y](hierarchy: Hierarchy[X], context: Context[Y]): Option[V] = {
    //println(s"Computing $this")
    val ret = Some(v)
    //println(s"Returning $ret from $this")
    ret
  }
  def serialize(n: Int): String = indent(n) + super.toString + s"(\n${predecessor.serialize(n+1)}" + carriage(n) + ")"
  override def toString: String = serialize(0)
}

final class BroaderValue[V](val myProxy: Proxy[Any], valueProxy: Proxy[Any], value: Contextual[V], predecessor: Contextual[V]) extends ComputedContextual[V] {
  require(myProxy.proto == predecessor.protoParent)
  //println(s"Building BroaderValue: $this")
  def proto = predecessor.proto
  def kind: String = s"<%B%>"
  def compute[X, Y](hierarchy: Hierarchy[X], context: Context[Y]): Option[V] = {
    require(hierarchy.proto == myProxy.proto)
    //println(s"Computing BroaderValue: ${hierarchy.proxy.getClass.getSimpleName}, ${myProxy.getClass.getSimpleName}")
    if(myProxy.toHierarchy.isNarrowerOrEquivalentTo(hierarchy)) {
      //println(s"Too")
      value.compute(hierarchy, valueProxy.toContext)
    } else {
      //println(s"Fro")
      predecessor.compute(hierarchy, valueProxy.toContext)
    }
  }
  def serialize(n: Int): String = indent(n) + super.toString + s"(${carriage(n+1)}${myProxy.getClass.getSimpleName()}, ${value.serialize(n+1)}, \n${predecessor.serialize(n + 1)}" + carriage(n) + ")"
  override def toString: String = serialize(0)
}

//final class NarrowedValue[V](val context: Hierarchy[Any], predecessor: Contextual[V]) extends ComputedContextual[V] {
//  require(context.proto == predecessor.protoParent)
//  def proto = predecessor.proto
//  def compute[X](hierarchy: Hierarchy[X]): Option[V] = {
//    require(hierarchy.proto == context.proto)
//    if(hierarchy.isNarrowerOrEquivalentTo(context)) predecessor.compute(hierarchy) else None
//  }
//}
//
//final class MergedValue[V](val contextuals: Set[Contextual[_]]) extends ComputedContextual[V] {
//  def proto = contextuals.head.proto.asInstanceOf[Contextual[V]]
//  require(contextuals.forall(c => c.proto == proto), s"Cannot create a MergedValue with contextuals which don't share a proto")
//
//  def compute[X](hierarchy: Hierarchy[X]): Option[V] = {
//    val newValues = contextuals.flatMap(_.compute(hierarchy).map(_.asInstanceOf[V]))
//    require(newValues.size <= 1, s"Not enough context to determine proper contextual value")
//    newValues.headOption
//  }
//}
//
object Contextual {
  def apply[V](value: V)(implicit contextualInstancer: ContextualInstancer[V]): Contextual[V] = contextualInstancer(value)
  def empty[V](implicit contextualInstancer: ContextualInstancer[V]): Contextual[V] = contextualInstancer.empty[V]
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
