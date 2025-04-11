// SPDX-License-Identifier: Apache-2.0

package firrtl.annotations

import firrtl._
import ir.{DefInstance, DefModule}

/** Building block to represent a [[Target]] of a FIRRTL component */
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
sealed trait TargetToken {
  def keyword: String
  def value:   Any

  /** Returns whether this token is one of the type of tokens whose keyword is passed as an argument
    * @param keywords
    * @return
    */
  def is(keywords: String*): Boolean = {
    keywords.map { kw =>
      require(
        TargetToken.keyword2targettoken.keySet.contains(kw),
        s"Keyword $kw must be in set ${TargetToken.keyword2targettoken.keys}"
      )
      val lastClass = this.getClass
      lastClass == TargetToken.keyword2targettoken(kw)("0").getClass
    }.reduce(_ || _)
  }
}

/** Object containing all [[TargetToken]] subclasses */
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case object TargetToken {
  case class Instance(value: String) extends TargetToken { override def keyword: String = "inst" }
  case class OfModule(value: String) extends TargetToken { override def keyword: String = "of" }
  case class Ref(value: String) extends TargetToken { override def keyword: String = "ref" }
  case class Index(value: Int) extends TargetToken { override def keyword: String = "[]" }
  case class DynamicIndex(value: Seq[TargetToken]) extends TargetToken { override def keyword: String = "[>]" }
  case class Field(value: String) extends TargetToken { override def keyword: String = "." }

  implicit class fromStringToTargetToken(s: String) {
    def Instance: Instance = new TargetToken.Instance(s)
    def OfModule: OfModule = new TargetToken.OfModule(s)
    def Ref:      Ref = new TargetToken.Ref(s)
    def Field:    Field = new TargetToken.Field(s)
  }

  implicit class fromIntToTargetToken(i: Int) {
    def Index: Index = new TargetToken.Index(i)
  }

  implicit class fromDefModuleToTargetToken(m: DefModule) {
    def OfModule: OfModule = new TargetToken.OfModule(m.name)
  }

  implicit class fromDefInstanceToTargetToken(i: DefInstance) {
    def Instance: Instance = new TargetToken.Instance(i.name)
    def OfModule: OfModule = new TargetToken.OfModule(i.module)
    def toTokens: (Instance, OfModule) = (new TargetToken.Instance(i.name), new TargetToken.OfModule(i.module))
  }

  val keyword2targettoken = Map(
    "inst" -> ((value: String) => Instance(value)),
    "of" -> ((value: String) => OfModule(value)),
    "ref" -> ((value: String) => Ref(value)),
    "[]" -> ((value: String) => Index(value.toInt)),
    "[>]" -> ((value: String) => DynamicIndex(Seq())),
    "." -> ((value: String) => Field(value))
  )
}
