package chisel3.experimental

package object hierarchy {
  type Instance[P] = core.Instance[P]
  val Instance = core.Instance
  type Definition[P] = core.Definition[P]
  val Definition = core.Definition
  type Hierarchy[P] = core.Hierarchy[P]
  val Hierarchy = core.Hierarchy
  type IsInstantiable = core.IsInstantiable
  type IsLookupable = core.IsLookupable
}
