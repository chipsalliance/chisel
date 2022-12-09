// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy.core

/** A User-extendable trait to mark metadata-containers, e.g. parameter case classes, as valid to return unchanged
  * from an instance.
  *
  * This should only be true of the metadata returned is identical for ALL instances!
  *
  * @example For instances of the same proto, metadata or other construction parameters
  *   may be useful to access outside of the instance construction. For parameters that are
  *   the same for all instances, we should mark it as IsLookupable
  * {{{
  * case class Params(debugMessage: String) extends IsLookupable
  * class MyModule(p: Params) extends Module {
  *   printf(p.debugMessage)
  * }
  * val myParams = Params("Hello World")
  * val definition = Definition(new MyModule(myParams))
  * val i0 = Instance(definition)
  * val i1 = Instance(definition)
  * require(i0.p == i1.p) // p is only accessable because it extends IsLookupable
  * }}}
  */
trait IsLookupable
