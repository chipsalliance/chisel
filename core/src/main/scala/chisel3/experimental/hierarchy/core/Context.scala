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
trait Context[+P] {
  /** @return Underlying proxy at this context. Used when creating children contexts.  */
  private[chisel3] def proxy: Proxy[P]

  /** All Contexts have a pointer to their RootContext, so that when a value is set, we can return the entire context
    *  starting from the root can be provided.
    * @return root of this context
    */
  private[chisel3] def root:   RootContext[_]

  /** @return Return the Hierarchy representation of this Context.  */
  private[chisel3] def toHierarchy: Hierarchy[P] = proxy match {
    case d: DefinitionProxy[P] => Definition(d)
    case i: InstanceProxy[P]   => Instance(i)
  }

  // All stored contextual edit functions for contextuals defined within the proxy's proto
  private[chisel3] val edits = new IdentityHashMap[Contextual[Any], Any => Any]()

  // Store an edit function for a given contextual, at this context
  // Can only set this function once per contextual, at this context
  private[chisel3] def addEdit[T](contextual: Contextual[T], edit: T => T): Unit = {
    require(!edits.containsKey(contextual), s"Cannot set $contextual more than once!")
    edits.put(contextual, edit.asInstanceOf[Any => Any])
  }

  // Return an edit function for a given contextual, at this context
  // If no function is defined, return an identity function
  private[chisel3] def getEdit[T](contextual: Contextual[T]): Edit[T] = {
    if (edits.containsKey(contextual)) Edit(contextual, edits.get(contextual).asInstanceOf[T => T])
    else {
      Edit(contextual, { t: T => t })
    }
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
  private[chisel3] def compute[T](key: Contextual[T], c: Contextual[T]): Contextual[T] = new Contextual(
    getEdit(key).edit(c.value)
  )

  // Caching returned values from lookup'ed values
  private[chisel3] val cache = new IdentityHashMap[Any, Any]()
  private[chisel3] val getterCache = new IdentityHashMap[Any, Any]()


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
    if (cache.containsKey(protoValue)) cache.get(protoValue).asInstanceOf[lookupable.S]
    else {
      val retValue = lookupable.setter(protoValue, this)
      cache.put(protoValue, retValue)
      retValue
    }
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
    if (getterCache.containsKey(protoValue)) getterCache.get(protoValue).asInstanceOf[lookupable.G]
    else {
      val retValue = lookupable.getter(protoValue, this)
      getterCache.put(protoValue, retValue)
      retValue
    }
  }
}

/** The top-most context in a context path. This is a separate type so functions can require a root */
final case class RootContext[+P](proxy: Proxy[P]) extends Context[P] {
  def root = this
}

/** The a nested context */
final case class NestedContext[+P](proxy: Proxy[P], root: RootContext[_]) extends Context[P] {}

/** The returned value from Contextual._lookup, when looking up a Contextual.
  * This enables users to specify a Contextual's value from a given context.
  *
  * @param contextual The proto's version of this contextual
  * @param context The context from which you are setting the value
  */
final case class ContextualSetter[T](contextual: Contextual[T], context: Context[_]) {
  def value: T = contextual.value
  def value_=(newValue: T): RootContext[_] = edit({ _: T => newValue })
  def edit(f: T => T): RootContext[_] = {
    context.addEdit(contextual, f)
    context.root
  }
}

/** An edit of a contextual.
  *
  * @param contextual the key, or the proto's version of this contextual
  * @param edit the function which edits a contextual's value
  */
final case class Edit[T](contextual: Contextual[T], edit: T => T)
