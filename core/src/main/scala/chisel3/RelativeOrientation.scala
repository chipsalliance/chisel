package chisel3

import chisel3.experimental.DataMirror


// Indicates whether the active side is aligned or flipped relative to the active side's root
sealed trait RelativeOrientation { 
  def data: Data
  def invert: RelativeOrientation
  def coerced: Boolean
  def coerce: RelativeOrientation
  def alignsWith(o: RelativeOrientation): Boolean = o.coerce.swap(DontCare) == this.coerce.swap(DontCare) // Clear out coerce and data in comparison
  def childOrientations: Seq[RelativeOrientation] = data match {
    case a: Aggregate => a.getElements.map(e => RelativeOrientation.deriveOrientation(e, this))
    case o => Nil
  }
  def swap(d: Data): RelativeOrientation
}
object RelativeOrientation {
  def apply(d: Data): RelativeOrientation = AlignedWithRoot(d, isCoercing(d))
  def isCoercing(d: Data): Boolean = {
    def recUp(x: Data): Boolean = x.binding match {
      case _ if isCoercing(x)           => true
      case None                         => false
      case Some(t: internal.TopBinding) => false
      case Some(internal.ChildBinding(p))        => recUp(p)
      case other                        => throw new Exception(s"Unexpected $other! $x, $d")
    }
    def isCoercing(d: Data): Boolean = {
      val s = DataMirror.specifiedDirectionOf(d)
      (s == SpecifiedDirection.Input) || (s == SpecifiedDirection.Output)
    }
    val ret = recUp(d)
    //println(s"isCoercing: $d gives $ret")
    ret
  }

  /** Determines the aligned/flipped of subMember with respect to activeRoot
    *
    * Due to Chisel/chisel3 differences, its a little complicated to calculate the RelativeOrientation, as the information
    *   is captured with both ActualDirection and SpecifiedDirection. Fortunately, all this complexity is captured in this
    *   one function.
    *
    * References activeRoot, defined earlier in the function
    *
    * @param subMember a subfield/subindex of activeRoot (or sub-sub, or sub-sub-sub etc)
    * @param orientation aligned/flipped of d's direct parent aggregate with respect to activeRoot
    * @return orientation aligned/flipped of d with respect to activeRoot
    */
  def deriveOrientation(subMember: Data, orientation: RelativeOrientation): RelativeOrientation = {
    //TODO(azidar): write exhaustive tests to demonstrate Chisel and chisel3 type/direction declarations compose
    val x = (DataMirror.specifiedDirectionOf(subMember)) match {
      case (SpecifiedDirection.Unspecified) => orientation.swap(subMember)
      case (SpecifiedDirection.Flip)        => orientation.invert.swap(subMember)
      case (SpecifiedDirection.Output)      => orientation.coerce.swap(subMember)
      case (SpecifiedDirection.Input)       => orientation.invert.coerce.swap(subMember)
      case other                            => throw new Exception(s"Unexpected internal error! $other")
    }
    //println(s"$subMember has $x")
    x
  }
}
case class AlignedWithRoot(data: Data, coerced: Boolean) extends RelativeOrientation {
  def invert = if(coerced) this else FlippedWithRoot(data, coerced)
  def coerce = this.copy(data, true)
  def swap(d: Data): RelativeOrientation = this.copy(data = d)
}
case class FlippedWithRoot(data: Data, coerced: Boolean) extends RelativeOrientation {
  def invert = if(coerced) this else AlignedWithRoot(data, coerced)
  def coerce = this.copy(data, true)
  def swap(d: Data): RelativeOrientation = this.copy(data = d)
}
case object EmptyOrientation extends RelativeOrientation {
  def data = DontCare
  def invert = this
  def coerced = false
  def coerce = this
  def swap(d: Data): RelativeOrientation = this
}
