// See LICENSE for license details.

// This file contains part of the implementation of the naming static annotation system.

package chisel3.internal.naming

import scala.collection.mutable.Stack
import scala.collection.mutable.ListBuffer

import scala.collection.JavaConversions._

import java.util.IdentityHashMap

/** Recursive Function Namer overview
  *
  * In every function, creates a NamingContext object, which associates all vals with a string name
  * suffix, for example:
  *   val myValName = SomeStatement()
  * produces the entry in items:
  *   {ref of SomeStatement(), "myValName"}
  *
  * This is achieved with a macro transforming:
  *   val myValName = SomeStatement()
  * statements into a naming call:
  *   val myValName = context.name(SomeStatement(), "myValName")
  *
  * The context is created from a global dynamic context stack at the beginning of each function.
  * At the end of each function call, the completed context is added to its parent context and
  * associated with the return value (whose name at an enclosing function call will form the prefix
  * for all named objects).
  *
  * When the naming context prefix is given, it will name all of its items with the prefix and the
  * associated suffix name. Then, it will check its descendants for sub-contexts with references
  * matching the item reference, and if there is a match, it will (recursively) give the
  * sub-context a prefix of its current prefix plus the item reference suffix.
  *
  * Note that for Modules, the macro will insert a naming context prefix call with an empty prefix,
  * starting the recursive naming process.
  */

/** Base class for naming contexts, providing the basic API consisting of naming calls and
  * ability to take descendant naming contexts.
  */
class NamingContext {
  val descendants = new IdentityHashMap[AnyRef, ListBuffer[NamingContext]]()
  val anonymousDescendants = ListBuffer[NamingContext]()
  val items = ListBuffer[(AnyRef, String)]()
  var closed = false  // a sanity check to ensure no more name() calls are done after name_prefix

  /** Adds a NamingContext object as a descendant - where its contained objects will have names
    * prefixed with the name given to the reference object, if the reference object is named in the
    * scope of this context.
    */
  def add_descendant(ref: AnyRef, descendant: NamingContext) {
    if (!descendants.containsKey(ref)) {
      descendants.put(ref, ListBuffer[NamingContext]())
    }
    descendants.get(ref) += descendant
  }

  def add_anonymous_descendant(descendant: NamingContext) {
    anonymousDescendants += descendant
  }

  /** Suggest a name (that will be propagated to FIRRTL) for an object, then returns the object
    * itself (so this can be inserted transparently anywhere).
    * Is a no-op (so safe) when applied on objects that aren't named, including non-Chisel data
    * types.
    */
  def name[T](obj: T, name: String): T = {
    assert(!closed, "Can't name elements after name_prefix called")
    obj match {
      case ref: AnyRef => items += ((ref, name))
      case _ =>
    }
    obj
  }

  /** Gives this context a naming prefix (which may be empty, "", for a top-level Module context)
    * so that actual naming calls (HasId.suggestName) can happen.
    * Recursively names descendants, for those whose return value have an associated name.
    */
  def name_prefix(prefix: String) {
    closed = true
    for ((ref, suffix) <- items) {
      // First name the top-level object
      ref match {
        case nameable: chisel3.internal.HasId => nameable.suggestName(prefix + suffix)
        case _ =>
      }

      // Then recurse into descendant contexts
      if (descendants.containsKey(ref)) {
        for (descendant <- descendants.get(ref)) {
          descendant.name_prefix(prefix + suffix + "_")
        }
        descendants.remove(ref)
      }
    }

    for (descendant <- descendants.values().flatten) {
      // Where we have a broken naming link, just ignore the missing parts
      descendant.name_prefix(prefix)
    }
    for (descendant <- anonymousDescendants) {
      descendant.name_prefix(prefix)
    }
  }
}

/** Class for the (global) naming stack object, which provides a way to push and pop naming
  * contexts as functions are called / finished.
  */
class NamingStack {
  val naming_stack = Stack[NamingContext]()

  /** Creates a new naming context, where all items in the context will have their names prefixed
    * with some yet-to-be-determined prefix from object names in an enclosing scope.
    */
  def push_context(): NamingContext = {
    val context = new NamingContext
    naming_stack.push(context)
    context
  }

  /** Called at the end of a function, popping the current naming context, adding it to the
    * enclosing context's descendants, and passing through the prefix naming reference.
    * Every instance of push_context() must have a matching pop_context().
    *
    * Will assert out if the context being popped isn't the topmost on the stack.
    */
  def pop_return_context[T <: Any](prefix_ref: T, until: NamingContext): T = {
    assert(naming_stack.top == until)
    naming_stack.pop()
    if (!naming_stack.isEmpty) {
      prefix_ref match {
        case prefix_ref: AnyRef => naming_stack.top.add_descendant(prefix_ref, until)
        case _ => naming_stack.top.add_anonymous_descendant(until)
      }

    }
    prefix_ref
  }

  /** Same as pop_return_context, but for cases where there is no return value (like Module scope).
   */
  def pop_context(until: NamingContext) {
    assert(naming_stack.top == until)
    naming_stack.pop()
  }
}
