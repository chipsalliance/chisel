package chisel3.experimental.hierarchy


/** Represents a clone of an underlying object. This is used to support CloneModuleAsRecord and Instance/Definition.
  *
  * @note We don't actually "clone" anything in the traditional sense but is a placeholder so we lazily clone internal state
  */
trait IsClone[+T <: IsInstantiable] {
  // Underlying object of which this is a clone of
  // TODO: this may need to change to represent Instance of something deserialized
  def getProto: T
  def contexts: Contexts

  /** Determines whether another object is a clone of the same underlying proto
    *
    * @param a
    */
  def hasSameProto(a: Any): Boolean = {
    val aProto = a match {
      case x: IsClone[_] => x.getProto
      case o => o
    }
    this == aProto || getProto == aProto
  }
  def setParent[P <: IsInstantiable](p: P): Unit = ???
  def getParent[P <: IsInstantiable](p: P): Unit = ???
  //def withContext(pf: PartialFunction[Any, Any]): IsClone[T]
}
