package chisel3.libs

import firrtl.annotations.{CircuitName, ComponentName, ModuleName, Named}

trait SubComponent {
  def keyword: String
  def value: Any
  override def toString = s"/$keyword@$value"
}
case class Instance(value: String)  extends SubComponent { override def keyword: String = "inst" }
case class OfModule(value: String)  extends SubComponent { override def keyword: String = "of" }
case class Ref(value: String)       extends SubComponent { override def keyword: String = "ref" }
case class Index(value: Int)        extends SubComponent { override def keyword: String = "[]" }
case class Field(value: String)     extends SubComponent { override def keyword: String = "." }
case class Arg(value: Int)          extends SubComponent { override def keyword: String = "arg" }
case class Anonymous(value: String) extends SubComponent { override def keyword: String = "" }
case class Bit(value: Int)          extends SubComponent { override def keyword: String = "bit" }
case object Clock                   extends SubComponent { override def keyword: String = "clock"; val value = "" }
case object Init                    extends SubComponent { override def keyword: String = "init";  val value = "" }
case object Reset                   extends SubComponent { override def keyword: String = "reset"; val value = "" }

case object SubComponent {
  //implicit def string2int(s: String): Int = s.toInt
  val keyword2subcomponent = Map(
    "inst" -> ((value: String) => Instance(value)),
    "of" -> ((value: String) => OfModule(value)),
    "ref" -> ((value: String) => Ref(value)),
    "[]" -> ((value: String) => Index(value.toInt)),
    "." -> ((value: String) => Field(value)),
    "arg" -> ((value: String) => Arg(value.toInt)),
    "" -> ((value: String) => Anonymous(value)),
    "bit" -> ((value: String) => Bit(value.toInt)),
    "clock" -> ((value: String) => Clock),
    "init" -> ((value: String) => Init),
    "reset" -> ((value: String) => Reset)
  )
}

case class Component(circuit: Option[String],
                     encapsulatingModule: Option[String],
                     reference: Seq[SubComponent],
                     tag: Option[Int] ) {
  def requireLast(default: Boolean, keywords: String*): Unit = {
    val isOne = if(reference.isEmpty) default else {
      keywords.map { kw =>
        val lastClass = reference.last.getClass
        lastClass == SubComponent.keyword2subcomponent(kw)("0").getClass()
      }.reduce(_ || _)
    }
    require(isOne, s"Last of $reference is not one of $keywords")
  }
  def ref(value: String): Component = {
    requireLast(true, "inst", "of")
    this.copy(reference = reference :+ Ref(value))
  }
  def inst(value: String): Component = {
    requireLast(true, "inst", "of")
    this.copy(reference = reference :+ Instance(value))
  }
  def of(value: String): Component = {
    requireLast(false, "inst")
    this.copy(reference = reference :+ OfModule(value))
  }
  def field(name: String): Component = this.copy(reference = reference :+ Field(name))
  def index(value: Int): Component = this.copy(reference = reference :+ Index(value))
  def bit(value: Int): Component = this.copy(reference = reference :+ Bit(value))
  def arg(index: Int): Component = {
    assert(reference.last.isInstanceOf[Anonymous])
    this.copy(reference = reference :+ Arg(index))
  }
  def clock: Component = this.copy(reference = reference :+ Clock)
  def init: Component = this.copy(reference = reference :+ Init)
  def reset: Component = this.copy(reference = reference :+ Reset)
  val circuitName: String = circuit.getOrElse(Component.emptyString)
  val moduleName: String = encapsulatingModule.getOrElse(Component.emptyString)
  override def toString(): String = {
    s"$$$tag$$ ($circuitName,$moduleName) ${reference.map(_.toString).mkString("")}"
  }
  def getComponentName: ComponentName = {
    val refs = reference.filter {
      case _: OfModule => false
      case _ => true
    }.map(_.value)
    ComponentName(refs.mkString("."), ModuleName(moduleName, CircuitName(circuitName)))
  }
}

object Component {

  private[libs] val counter = new java.util.concurrent.atomic.AtomicInteger(0)
  implicit def convertComponent2Named(c: Component): Named = {
    val cn = CircuitName(c.circuitName)
    val mn = ModuleName(c.moduleName, cn)
    (c.circuit, c.encapsulatingModule, c.reference) match {
      case (_, None, Nil) => cn
      case (_, _, Nil) => mn
      case (_, _, _) => ComponentName("c" + c.reference.mkString(""), mn)
    }
  }
  implicit def convertComponent2ComponentName(c: Component): ComponentName = {
    val cn = CircuitName(c.circuitName)
    val mn = ModuleName(c.moduleName, cn)
    ComponentName("c" + c.reference.mkString(""), mn)
  }
  implicit def convertComponent2ModuleName(c: Component): ModuleName = {
    val cn = CircuitName(c.circuitName)
    val mn = ModuleName(c.moduleName, cn)
    mn
  }
  implicit def convertComponent2CircuitName(c: Component): CircuitName = {
    CircuitName(c.circuitName)
  }

  val emptyString: String = "E@"

  implicit def string2opt(s: String): Option[String] = if(s == emptyString) None else Some(s)

  implicit def named2component(n: Named): Component = n match {
    case CircuitName(x) => Component(x, None, Nil, None)
    case ModuleName(m, CircuitName(c)) => Component(c, m, Nil, None)
    case ComponentName(name, ModuleName(m, CircuitName(c))) =>
      val subcomps = tokenize(name.tail)
      Component(c, m, subcomps, None)
  }

  def tokenize(s: String): Seq[SubComponent] = if(!s.isEmpty && s.head == '/') {
    val endKeywordIndex = s.indexWhere(c => c == '@', 1)
    val keyword = s.slice(1, endKeywordIndex)
    val endValueIndex = s.indexWhere(c => c == '/', endKeywordIndex + 1) match {
      case -1 => s.length
      case i => i
    }
    val value = s.slice(endKeywordIndex + 1, endValueIndex)
    SubComponent.keyword2subcomponent(keyword)(value) +: tokenize(s.substring(endValueIndex))
  } else Nil
}

object Referable {
  def apply(name: String, encapsulatingModule: String): Component = {
    Component(None, Some(encapsulatingModule), Seq(Ref(name)), Some(Component.counter.incrementAndGet()))
  }
}

object Irreferable {
  def apply(value: String, encapsulatingModule: String): Component = {
    Component(None, Some(encapsulatingModule), Seq(Anonymous(value)), Some(Component.counter.incrementAndGet()))
  }
}
