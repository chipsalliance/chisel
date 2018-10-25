// See LICENSE for license details.

package firrtl.annotations

/** Building block to represent a [[Target]] of a FIRRTL component */
sealed trait TargetToken {
  def keyword: String
  def value: Any

  /** Returns whether this token is one of the type of tokens whose keyword is passed as an argument
    * @param keywords
    * @return
    */
  def is(keywords: String*): Boolean = {
    keywords.map { kw =>
      require(TargetToken.keyword2targettoken.keySet.contains(kw),
        s"Keyword $kw must be in set ${TargetToken.keyword2targettoken.keys}")
      val lastClass = this.getClass
      lastClass == TargetToken.keyword2targettoken(kw)("0").getClass
    }.reduce(_ || _)
  }
}

/** Object containing all [[TargetToken]] subclasses */
case object TargetToken {
  case class Instance(value: String)  extends TargetToken { override def keyword: String = "inst" }
  case class OfModule(value: String)  extends TargetToken { override def keyword: String = "of" }
  case class Ref(value: String)       extends TargetToken { override def keyword: String = "ref" }
  case class Index(value: Int)        extends TargetToken { override def keyword: String = "[]" }
  case class Field(value: String)     extends TargetToken { override def keyword: String = "." }
  case object Clock                   extends TargetToken { override def keyword: String = "clock"; val value = "" }
  case object Init                    extends TargetToken { override def keyword: String = "init";  val value = "" }
  case object Reset                   extends TargetToken { override def keyword: String = "reset"; val value = "" }

  val keyword2targettoken = Map(
    "inst" -> ((value: String) => Instance(value)),
    "of" -> ((value: String) => OfModule(value)),
    "ref" -> ((value: String) => Ref(value)),
    "[]" -> ((value: String) => Index(value.toInt)),
    "." -> ((value: String) => Field(value)),
    "clock" -> ((value: String) => Clock),
    "init" -> ((value: String) => Init),
    "reset" -> ((value: String) => Reset)
  )
}

