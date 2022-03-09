package chisel3.experimental.hierarchy.core
import java.util.IdentityHashMap

// Wrapper Class
sealed trait Proxy[+P] {
  def proto: P
  private[chisel3] val cache = new IdentityHashMap[Any, Any]()
  def lookup[U, C](value: Proxy[U])(
      mockerU: Mocker[U, C],
      mockerC: Mocker[C, C],
      proxifier: Proxifier[C]
  ): Proxy[U] = (this, value, this.proto == value.proto) match {
    case (t: Transparent[P, C],   v,                        _)     => println("0");v
    case (t: Proxy[P],            v: InstantiableProxy[U, C],  _)     => println("0.5");InstantiableProxy(v.proto, t)
    case (t: DefinitionProxy[P],  v: DefinitionProxy[U],    true)  => println("1");v
    case (t: DefinitionProxy[P],  v: DefinitionProxy[U],    false) => println("2");v
    case (t: DefinitionProxy[P],  v: HierarchicalProxy[U, C],   true)  => println("3");t.asInstanceOf[DefinitionProxy[U]]
    case (t: DefinitionProxy[P],  v: HierarchicalProxy[U, C],   false) => println("4");
      val newParent = this.blookup(v.parent)(mockerC, mockerC, proxifier, proxifier).asInstanceOf[C]
      if(newParent == v.parent) v else mockerU(v, newParent)
    case (t: HierarchicalProxy[P, C], v: DefinitionProxy[U], true)  => println("5");
      t.asInstanceOf[HierarchicalProxy[U, C]]
    case (t: HierarchicalProxy[P, C], v: DefinitionProxy[U], false) => println("6");v
    case (t: HierarchicalProxy[P, C], v: HierarchicalProxy[U, C],   true)  => println("7");
      val newGenesis = if(t.genesis == v) t.asInstanceOf[HierarchicalProxy[U, C]] else t.genesis.ilookup(v)(mockerU, mockerC, proxifier)
      mockerU(newGenesis, t.asInstanceOf[C])
    case (t: HierarchicalProxy[P, C], v: HierarchicalProxy[U, C],   false) => println("8");
      val newParent  = this.blookup(v.parent)(mockerC, mockerC, proxifier, proxifier).asInstanceOf[C]
      val newGenesis = t.genesis.ilookup(v)(mockerU, mockerC, proxifier)
      if(newGenesis == v && newParent == v.parent) v else mockerU(newGenesis, newParent)
    case (t: InstantiableProxy[P, C], v, _) => println("9");t.parent.asInstanceOf[Proxy[C]].lookup(v)(mockerU, mockerC, proxifier)
  }
  def ilookup[U, C](value: Proxy[U])(
      mockerU: Mocker[U, C],
      mockerC: Mocker[C, C],
      proxifier: Proxifier[C]
  ):  HierarchicalProxy[U, C] = {
    lookup(value)(mockerU, mockerC, proxifier).asInstanceOf[HierarchicalProxy[U, C]]
  }
  def blookup[U, C](value: U)(
      mockerU: Mocker[U, C],
      mockerC: Mocker[C, C],
      proxifierU: Proxifier[U],
      proxifierC: Proxifier[C]
  ): Proxy[U] = {
    //println(s"  value=$value")
    val p = proxifierU(value, this)
    //println(s"  p=$p")
    val ret = lookup(p)(mockerU, mockerC, proxifierC)
    //println(s"  blookupRet=$ret")
    ret
  }

  def toDefinition: Definition[P]
}

trait Mocker[U, C] {
  def apply(newValue: HierarchicalProxy[U, C], parent: C): HierarchicalProxy[U, C]
}
object Mocker {
//  implicit def instantiable[U <: IsInstantiable, C] = new Mocker[U, C] {
//    def apply(newValue: HierarchicalProxy[U, C], parent: C): HierarchicalProxy[U, C] = ???
//  }
}

trait Proxifier[C] {
  def apply[P](protoValue: C, proxy: Proxy[P]): Proxy[C]
}
sealed trait InstanceProxy[+P, +C] extends Proxy[P] {
  def proto: P
  def parent: C
  def toInstance = new Instance(this)
  def toDefinition: Definition[P]
}

sealed trait HierarchicalProxy[+P, +C] extends InstanceProxy[P, C] {
  def genesis: Proxy[P]
  def proto = genesis.proto
  def toDefinition: Definition[P] = genesis.toDefinition
}

sealed trait LocalHierarchicalProxy[+P, +C] extends HierarchicalProxy[P, C]

trait Clone[+P, +C] extends LocalHierarchicalProxy[P, C] {
  def genesis: DefinitionProxy[P]
}
trait Transparent[+P, +C] extends LocalHierarchicalProxy[P, C] {
  def genesis: DefinitionProxy[P]
}
trait Mock[+P, +C] extends HierarchicalProxy[P, C] {
  def genesis: HierarchicalProxy[P, C]
  def local: LocalHierarchicalProxy[P, C] = genesis match {
    case l: LocalHierarchicalProxy[P, C] => l
    case m: Mock[P, C] => m.local
  }
}
trait DefinitionProxy[+P] extends Proxy[P] {
  def toDefinition = new Definition(this)
}

final case class InstantiableDefinition[P](proto: P) extends DefinitionProxy[P]
final case class InstantiableProxy[P, C](proto: P, parent: C) extends InstanceProxy[P, C] {
  def toDefinition: Definition[P] = InstantiableDefinition(proto).toDefinition
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
