package chisel3.experimental.hierarchy.core
import java.util.IdentityHashMap

// Wrapper Class
sealed trait Proxy[+P] {
  def proto: P
  def compute[T](key: Contextual[T], contextual: Contextual[T]): Contextual[T]
  def lenses: Seq[Lense[P]]
  def lineageOpt: Option[Proxy[Any]]

  private[chisel3] def open[V](contextual: Contextual[V]): V = ???

  def toDefinition: Definition[P]
}

//trait Mocker[U, C] {
//  def apply(newValue: HierarchicalProxy[U, C], parent: C): HierarchicalProxy[U, C]
//}
//object Mocker {
////  implicit def instantiable[U <: IsInstantiable, C] = new Mocker[U, C] {
////    def apply(newValue: HierarchicalProxy[U, C], parent: C): HierarchicalProxy[U, C] = ???
////  }
//}

trait Proxifier[C] {

  def apply[P](protoValue: C, proxy: Proxy[P]): Proxy[C]
}
sealed trait InstanceProxy[+P] extends Proxy[P] {
  def genesis: Proxy[P]
  def compute[T](key: Contextual[T], contextual: Contextual[T]): Contextual[T] = {
    val genesisContextual = genesis.compute(key, contextual)
    lenses.foldLeft(genesisContextual) { case (c, lense) => lense.compute(key, c) }
  }
  def proto = genesis.proto
  def lineageOfType[C](pf: PartialFunction[Any, C]): Option[C] = lineageOpt match {
    case Some(a) if pf.isDefinedAt(a) => pf.lift(a)
    case Some(i: InstanceProxy[Any]) => i.lineageOfType[C](pf)
    case other => None
  }
  def toInstance = new Instance(this)
  def toDefinition: Definition[P] = genesis.toDefinition
  def localProxy: InstanceProxy[P] = genesis match {
    case d: DefinitionProxy[P] => this
    case i: InstanceProxy[P] => i.localProxy
  }
}

trait Clone[+P] extends InstanceProxy[P] {
  def lineageOpt: Option[Proxy[Any]] = None
  def genesis: DefinitionProxy[P]
}
trait Transparent[+P] extends InstanceProxy[P] {
  def lineageOpt: Option[Proxy[Any]] = None
  def genesis: DefinitionProxy[P]
}
trait Mock[+P] extends InstanceProxy[P] {
  def lineage: Proxy[Any]
  def lineageOpt: Option[Proxy[Any]] = Some(lineage)
  def genesis: InstanceProxy[P]
}
trait DefinitionProxy[+P] extends Proxy[P] {
  def lenses: Seq[Lense[P]] = Nil
  def compute[T](key: Contextual[T], contextual: Contextual[T]): Contextual[T] = {
    lenses.foldLeft(contextual) { case (c, lense) => lense.compute(key, c) }
  }
  def lineageOpt: Option[Proxy[Any]] = None
  def toDefinition = new Definition(this)
}

final case class InstantiableDefinition[P](proto: P) extends DefinitionProxy[P]
final case class InstantiableTransparent[P](genesis: InstantiableDefinition[P], lenses: Seq[Lense[P]]) extends InstanceProxy[P] {
  val lineageOpt = None
}
final case class InstantiableMock[P](genesis: InstanceProxy[P], lineage: Proxy[Any], lenses: Seq[Lense[P]]) extends InstanceProxy[P] {
  val lineageOpt = Some(lineage)
}



//sealed trait LocalProxy[+T] extends Proxy[T]
//
//// Used for when proxy implementation is pure
//final case class Proto[+T](proto: T) extends LocalProxy[T] {
//  def localProxy: LocalProxy[T] = this
//}
//
//// Used for when proxy implementation is not pure, and thus requires a mock up
//final case class Clone[+T](standIn: IsLocalStandIn[T]) extends LocalProxy[T] {
//  def proto = standIn.proto
//  def localProxy: LocalProxy[T] = this
//}
//
//final case class Lineage[+T](standIn: IsLineageStandIn[T]) extends Proxy[T] {
//  def proto = standIn.proto
//  def localProxy: LocalProxy[T] = standIn.genesisProxy match {
//    case l: LocalProxy[T] => l
//    case other: Proxy[T] => other.localProxy
//  }
//}

//sealed trait IsStandIn[+P] {
//  def proto: P
//  //def toInstance: Instance[P]
//  //def toDefinition: Definition[P] = toInstance.toDefinition
//
//  private val sockets = new java.util.IdentityHashMap[Contextual[Any], Any => Any]()
//  def addEdit[T](contextual: Contextual[T], edit: T => T): Unit = {
//    require(!sockets.containsKey(contextual), s"Cannot set the same Contextual twice, using the same lense! $this, $contextual")
//    sockets.put(contextual, edit.asInstanceOf[Any => Any])
//  }
//  def edit[T](contextual: Contextual[T]): T = {
//    def applyMyEdits[T](value: T): T = {
//      if(sockets.containsKey(contextual)) sockets.get(contextual)(value).asInstanceOf[T] else value
//    }
//    genesisProxy match {
//      case Proto(_) => applyMyEdits(contextual.value)
//      case StandIn(standIn: IsStandIn[_]) => applyMyEdits(standIn.edit(contextual))
//    }
//  }
//}

//trait IsLineageStandIn[+P] extends IsStandIn {
//  //def toInstance: core.Instance[T] = new core.Instance(Lineage(this))
//}
//
//trait IsLocalStandIn[+P] extends IsStandIn {
//  //def genesisProxy: LocalProxy[P]
//}

// Default implementation for IsInstantiable, as it does not add context
//case class StandInInstantiable[P <: IsInstantiable, C](genesisProxy: Proxy[P], parent: Option[StandIn[IsStandIn[C]]]) extends IsStandIn[P] {
//  def toInstance:   Instance[P] = new Instance(StandIn(this))
//  def toDefinition: Definition[P] = toInstance.toDefinition
//}


// Typeclass Trait
//trait LocalProxifier[V]  {
//  def apply[P](value: V): LocalProxifier[V]
//}
//trait Lineager[V]  {
//  def apply[P](value: V): Lineager[V]
//}


// Typeclass Default Implementations
object Proxifier {
  //implicit def proxifierInstance[P] = new Proxifier[Instance[P]] {
  //  def apply[P](value: Instance[P]) = value.proxy
  //}
  //implicit def proxifierDefinition[P] = new Proxifier[Definition[P]] {
  //  def apply[P](value: Definition[P]) = value.proxy
  //}
  //implicit def proxifierLense[P] = new Proxifier[Lense[P]] {
  //  def apply[P](value: Lense[P]) = value.proxy
  //}
  //implicit def isIsInstantiable[L <: IsInstantiable, C <: IsContext](implicit contexter: Contexter[L, C]) =
  //  new Proxifier[L] {
  //    type U = L
  //    def apply[P](value: L, hierarchy: Hierarchy[P]) = StandIn(InstantiableStandIn(value, contexter(value, hierarchy).context))
  //  }
}
