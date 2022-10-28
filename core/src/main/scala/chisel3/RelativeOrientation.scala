package chisel3

import chisel3.experimental.DataMirror


// Indicates whether the active side is aligned or flipped relative to the active side's root
sealed trait RelativeOrientation { 
  def data: Data
  def waivers: Set[Data]
  def invert: RelativeOrientation
  def coerced: Boolean
  def coerce: RelativeOrientation
  def alignsWith(o: RelativeOrientation): Boolean = o.alignment == this.alignment
  def childOrientations: Seq[RelativeOrientation] = data match {
    case a: Aggregate => a.getElements.map(e => RelativeOrientation.deriveOrientation(e, this))
    case o => Nil
  }
  def alignment: String
  def swap(d: Data): RelativeOrientation
  def isWaived: Boolean = waivers.contains(data)
  def isAgg: Boolean = data.isInstanceOf[Aggregate]
  def isConsumer: Boolean
  def errorWord(op: DirectionalConnectionFunctions.ConnectionOperator): String = (isConsumer, op.assignToConsumer, op.assignToProducer, alignment) match {
    case (true,  true,   _,    "aligned") => "unassigned"
    case (false, _,      true, "flipped") => "unassigned"
    case (true,  _,      true, "flipped") => "dangling"
    case (false, true,   _,    "aligned") => "dangling"
    case other => "unmatched"
  }
}
object RelativeOrientation {
  def apply(d: Data, waivers: Set[Data], isConsumer: Boolean): RelativeOrientation = AlignedWithRoot(d, isCoercing(d), waivers, isConsumer)
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

  def deriveOrientation(subMember: Data, orientation: RelativeOrientation): RelativeOrientation = {
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




  implicit val RelativeOrientationMatchingZipOfChildren = new DataMirror.HasMatchingZipOfChildren[RelativeOrientation] {
    def matchingZipOfChildren(left: Option[RelativeOrientation], right: Option[RelativeOrientation]): Seq[(Option[RelativeOrientation], Option[RelativeOrientation])] = {
      Data.DataMatchingZipOfChildren.matchingZipOfChildren(left.map(_.data), right.map(_.data)).map {
        case (Some(l), None)    => (Some(deriveOrientation(l, left.get)), None)
        case (Some(l), Some(r)) => (Some(deriveOrientation(l, left.get)), Some(deriveOrientation(r, right.get)))
        case (None, Some(r))    => (None, Some(deriveOrientation(r, right.get)))
      }
    }
  }
}

sealed trait NonEmptyOrientation extends RelativeOrientation
case class AlignedWithRoot(data: Data, coerced: Boolean, waivers: Set[Data], isConsumer: Boolean) extends NonEmptyOrientation {
  def invert = if(coerced) this else FlippedWithRoot(data, coerced, waivers, isConsumer)
  def coerce = this.copy(data, true)
  def swap(d: Data): RelativeOrientation = this.copy(data = d)
  def alignment: String = "aligned"
}
case class FlippedWithRoot(data: Data, coerced: Boolean, waivers: Set[Data], isConsumer: Boolean) extends NonEmptyOrientation {
  def invert = if(coerced) this else AlignedWithRoot(data, coerced, waivers, isConsumer)
  def coerce = this.copy(data, true)
  def swap(d: Data): RelativeOrientation = this.copy(data = d)
  def alignment: String = "flipped"
}
case object EmptyOrientation extends RelativeOrientation {
  def data = DontCare
  def waivers = Set.empty
  def invert = this
  def coerced = false
  def coerce = this
  def swap(d: Data): RelativeOrientation = this
  def alignment: String = "none"
  def isConsumer = ??? // should never call this
}
