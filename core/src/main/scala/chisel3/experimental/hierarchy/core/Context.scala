// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy.core
import scala.collection.mutable
import java.util.IdentityHashMap

/** Represents a user-specified path in a hierarchy, which is then used to set and get Contextual values
  *
  * Created when specifying the contextual values during Instance creation, and used when a contextual is
  *   looked up from a hierarchy.
  * Stored in InstanceProxy, so that when a value is looked up from that InstanceProxy, corresponding Contexts
  *   can also be analyzed.
  */
final case class Context[+P](proxy: Proxy[P]) {

  /** @return Underlying proxy at this context. Used when creating children contexts. */

  /** @return Return the Hierarchy representation of this Context. */
  private[chisel3] def toHierarchy: Hierarchy[P] = proxy match {
    case d: DefinitionProxy[P] => new Definition(d)
    case i: InstanceProxy[P]   => new Instance(i)
  }


  /** Computes the new contextual given the original, proto Contextual (key) and the current contextual (c)
    *
    * We need two values here because the edits are stored according to the identity of the original, proto contextual value.
    * However, contextual values could be set in multiple contexts which grow in context. For example, a default
    * contextual value can be given, but when we have nested instances, each proto in that hierarchy path can give an
    * edit function. Thus, the contextual passed to the edit function may not be the same object as the original contextual.
    *
    * @param key original proto's contextual
    * @param c current contextual containing a value to be edited
    * @return a new contextual with the edited value
    */

  // Caching returned values from lookup'ed values
  //private[chisel3] val cache = new IdentityHashMap[Any, Any]()
  //private[chisel3] val getterCache = new IdentityHashMap[Any, Any]()

  /** Lookup function called by the macro-generated extension methods on Context[P]
    *
    * Returns the context version of the value looked up in the proto
    *
    * @param that lookup function which, given the proto, returns the desired proto's @public value
    * @param lookupable provides the setter function for type B
    * @param macroGenerated a layer of protection to avoid users calling the function directly
    * @return Context-version of the value looked up in the proto
    */
  def _lookup[B, C](
    that: P => B
  )(
    implicit lookupable: Lookupable[B],
    macroGenerated:      chisel3.internal.MacroGenerated
  ): lookupable.S = {
    val protoValue = that(this.proxy.proto)
    proxy.retrieveMeAsContext(protoValue).orElse(proxy.retrieveMe(protoValue)).orElse {
      val retValue = lookupable.setter(protoValue, this)
      proxy.cacheMe(protoValue, retValue)
      Some(retValue)
    }.get.asInstanceOf[lookupable.S]
  }

  /** Lookup function called by Lookupable's of Contextuals, which need to compute a contextual value
    *
    * @param that lookup function which, given the proto, returns the desired proto's @public value
    * @param lookupable provides the getter function for type B
    * @return Computed value for the looked up value at this context
    */
  def getter[B, C](
    protoValue: B
  )(
    implicit lookupable: Lookupable[B]
  ): lookupable.G = {
    proxy.retrieveMeAsContext(protoValue).orElse(proxy.retrieveMe(protoValue)).orElse {
      val retValue = lookupable.getter(protoValue, this)
      proxy.cacheMe(protoValue, retValue)
      Some(retValue)
    }.get.asInstanceOf[lookupable.G]
  }
}

/** The returned value from Contextual._lookup, when looking up a Contextual.
  * This enables users to specify a Contextual's value from a given context.
  *
  * @param contextual The proto's version of this contextual
  * @param context The context from which you are setting the value
  */
final case class ContextualSetter[T](contextual: Contextual[T], context: Context[Any]) {
  def value: T = contextual.compute(context.toHierarchy, context).get
  def value_=(newValue: T): Unit = {
    context.proxy.addValue(contextual, newValue, newValue.toString)
  }
  def edit(f: T => T): Unit = editWithString(f, f.toString)
  def editWithString(f: T => T, description: String): Unit = {
    context.proxy.addDerivation(contextual, f.asInstanceOf[Any => Any], description)
  }
}
