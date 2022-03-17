package chisel3.experimental.hierarchy.core

import java.util.IdentityHashMap

final class Contextual[+V](val value: V)

object Contextual {
  def apply[V](value: V): Contextual[V] = new Contextual(value)
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
