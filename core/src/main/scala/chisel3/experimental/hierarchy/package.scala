package chisel3.experimental

package object hierarchy {

  /** Classes or traits which will be used with the [[Definition]] + [[Instance]] api should be marked
    * with the [[instantiable]] annotation at the class/trait definition.
    *
    * @example {{{
    * @instantiable
    * class MyModule extends Module {
    *   ...
    * }
    *
    * val d = Definition(new MyModule)
    * val i0 = Instance(d)
    * val i1 = Instance(d)
    * }}}
    */
  class instantiable extends chisel3.internal.instantiable

  /** Classes marked with [[instantiable]] can have their vals marked with the [[public]] annotation to
    * enable accessing these values from a [[Definition]] or [[Instance]] of the class.
    *
    * Only vals of the the following types can be marked [[public]]:
    *   1. IsInstantiable
    *   2. IsLookupable
    *   3. Data
    *   4. BaseModule
    *   5. Iterable/Option containing a type that meets these requirements
    *   6. Basic type like String, Int, BigInt etc.
    *
    * @example {{{
    * @instantiable
    * class MyModule extends Module {
    *   @public val in = IO(Input(UInt(3.W)))
    *   @public val out = IO(Output(UInt(3.W)))
    *   ..
    * }
    *
    * val d = Definition(new MyModule)
    * val i0 = Instance(d)
    * val i1 = Instance(d)
    *
    * i1.in := i0.out
    * }}}
    */
  class public extends chisel3.internal.public

  type Instance[P] = core.Instance[P]
  val Instance = core.Instance
  type Definition[P] = core.Definition[P]
  val Definition = core.Definition
  type Hierarchy[P] = core.Hierarchy[P]
  val Hierarchy = core.Hierarchy
  type IsInstantiable = core.IsInstantiable
  type IsLookupable = core.IsLookupable
}
