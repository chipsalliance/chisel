// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy.core
import java.util.IdentityHashMap
import scala.collection.mutable
import chisel3.internal.sourceinfo.SourceInfo
import Contextual.log

/** Representation of a hierarchical version of an object
  * Has two subclasses, a DefinitionProxy and InstanceProxy, which are wrapped with Definition(..) and Instance(..)
  */
trait Proxy[+P] {

  /** @return Original object that we are proxy'ing */
  def proto: P

  def debug: String
  def parentColumn: List[String]
  def suffixMatrix: List[List[String]]
  def displaySuffixMatrix: String = {
    def align(str: String, l: Int) = str + (" "*(str.length - l))
    val lengths = suffixMatrix.map {
      pc => pc.foldLeft(0){ (sz: Int, d: String) => sz.max(d.length) }
    }
    val colSize = suffixMatrix.head.length
    (0 until colSize).map { colIndex =>
      suffixMatrix.zip(lengths).map { case (pc, l) => align(pc(colIndex), l) }.mkString(" -S> ")
    }.mkString("\n")
  }

  private[chisel3] val cache = new IdentityHashMap[Any, Any]()
  private[chisel3] val visitTracker = new IdentityHashMap[Any, Int]()

  private[chisel3] val protoToName = new IdentityHashMap[Any, String]()
  private[chisel3] val nameToProto = new IdentityHashMap[Any, String]()

  def identity = this.hashCode

  def mock[X](args: Any*)(buildMock: => X): X = {
    val key = args
    if (cache.containsKey(key)) cache.get(key).asInstanceOf[X] else {
      val ret = buildMock
      cache.put(key, ret)
      ret
    }
  }

  def getIdentity(value: Any): Any = value match {
    case w: Wrapper[_] => w.proxy
    case w: Contextual[_] => w.proxy
    case other => other
  }
  def cacheMe[V, R](protoValue: V, retValue: R): Unit = {
    cache.put(getIdentity(protoValue), getIdentity(retValue))
  }
  def retrieveMe[V](protoValue: V): Option[Any] = {
    val key = getIdentity(protoValue)
    if (cache.containsKey(key)) {
      cache.get(key) match {
        case p: Proxy[_] => Some(p.toWrapper)
        case other => Some(other)
      }
    } else None
  }
  def visit[V](protoValue: V): Unit = {
    val key = getIdentity(protoValue)
    val nVisits = if(visitTracker.containsKey(key)) visitTracker.get(key) + 1 else 1
    visitTracker.put(key, nVisits)
  }
  def nVisits[V](protoValue: V): Int = {
    val key = getIdentity(protoValue)
    visitTracker.getOrDefault(key, 0)
  }

  /** @return a Definition wrapping this Proxy */
  //def toSetter: Setter[P]
  def toWrapper: Wrapper[P]
}

trait HierarchicalProxy[+P] extends Proxy[P] {
  //override def toSetter: HierarchySetter[P]
  def underlying: Underlying[P]
  final def proto: P = underlying.proto
  def toHierarchy: Hierarchy[P]
  def toRoot:      Root[P]
  def parentOpt:   Option[Proxy[Any]]
  def parentDebug = parentOpt.map(_.debug).getOrElse("~")
  def parentColumn: List[String] = List(this.debug) ++ parentOpt.map(_.parentColumn).getOrElse(List.empty[String])
  def toWrapper: Hierarchy[P] = toHierarchy
  def isResolved: Boolean
}

/** Proxy representing an Instance version of an object */
sealed trait InstanceProxy[+P] extends HierarchicalProxy[P] {

  /** The proxy which refers to the same proto, but whose root is one hierarchical step closer to the proto
    *
    * Example 0: if this Proxy refers to ~Top|Foo/bar:Bar, then suffixProxy refers to ~Top|Bar
    * Example 1: if this Proxy refers to ~Top|Top/foo:Foo/bar:Bar, then suffixProxy refers to ~Top|Foo/bar:Bar
    *
    * @return the suffixProxy proxy of this proxy
    */
  def suffixProxy: HierarchicalProxy[P]
  def suffixMatrix: List[List[String]] = List(parentColumn) ++ suffixProxy.suffixMatrix
  def isResolved: Boolean = suffixProxy.isResolved

  /** @return the InstanceProxy closest to the proto in the chain of suffixProxy proxy's */
  def localProxy: InstanceProxy[P] = suffixProxy match {
    case d: DefinitionProxy[P] => this
    case i: InstanceProxy[P]   => i.localProxy
  }

  override def underlying = suffixProxy.underlying

  /** @return an Instance wrapping this Proxy */
  def toInstance = new Instance(this)

  override def toRoot: Root[P] = suffixProxy.toRoot

  override def toHierarchy: Hierarchy[P] = toInstance

  //override def toSetter: HierarchySetter[P] = HierarchySetter(this)
}

/** InstanceProxy representing a new Instance which was instantiated in a proto.
  *
  * Note: Clone's parentOpt is always empty
  * Note: Clone's suffixProxy is always a DefinitionProxy
  *
  * E.g. it is the underlying proxy of Instance when a user writes:
  *   val inst = Instance(..)
  */
trait Clone[+P] extends InstanceProxy[P] {
  override def parentOpt: Option[Proxy[Any]] = None

  override def suffixProxy: RootProxy[P]
}

/** InstanceProxy representing an Instance which was created from converting an existing object into an Instance.
  *
  * Note: Transparent's parentOpt is always empty
  * Note: Transparent's suffixProxy is always a DefinitionProxy
  *
  * E.g. it is the underlying proxy of Instance when a user writes:
  *   val inst = myObject.toInstance
  */
trait Transparent[+P] extends InstanceProxy[P] {
  override def parentOpt: Option[Proxy[Any]] = None

  override def suffixProxy: RootProxy[P]
}

/** InstanceProxy representing an Instance which was created when a proto is looked up from a Hierarchy or Context
  *
  * Note: Mock's always have a non-empty parentOpt.
  * Note: Mock's suffixProxy is always an InstanceProxy
  *
  * E.g. it is the underlying proxy of child when a user writes:
  *   val parent = Instance(..)
  *   val child = parent.child
  */
trait Mock[+P] extends InstanceProxy[P] {

  /** @return Lineage of this Mock, e.g. the parent proxy from whom this Mock was looked up from */
  def parent: Proxy[Any]

  override def parentOpt: Option[Proxy[Any]] = Some(parent)

  override def suffixProxy: InstanceProxy[P]
}

trait RootProxy[+P] extends HierarchicalProxy[P] {
  def toHierarchy: Root[P]
  def toRoot:       Root[P] = toHierarchy
  def toDefinition: Definition[P] = toHierarchy.toDefinition
  def predecessorOption: Option[RootProxy[P]]
  def builder:           Option[Implementation]
  def parentOpt: Option[Proxy[Any]] = None
  def suffixMatrix: List[List[String]] = List(parentColumn)
}

trait DefinitionProxy[+P] extends RootProxy[P] {
  //def toInterfaceProxy: InterfaceProxy[P]
  def builder: Option[Implementation]

  var isResolved = false
  def predecessorOption = None
  override def toHierarchy: Definition[P] = new Definition(this)
}

//sealed trait ParameterProxy[P] extends Proxy[P] {
//  def parentOpt: Option[Proxy[Any]]
//  def compute[H](h: Hierarchy[H]): Option[P]
//  def hasDerivation: Boolean
//  def derivation: Option[Derivation]
//  def isResolved: Boolean
//}

sealed trait DefinitiveProxy[P] extends Proxy[P] {
  def compute: Option[P]
  var derivation: Option[DefinitiveDerivation]
  def isResolved:             Boolean = compute.nonEmpty
  private[chisel3] val namer: mutable.ArrayBuffer[Any => Unit] = mutable.ArrayBuffer[Any => Unit]()
  val sourceInfo: SourceInfo

  final def compute[H](h: Hierarchy[H]): Option[P] = compute
  def parentOpt: Option[Proxy[Any]] = None
  def parentDebug = parentOpt.map(_.debug).getOrElse("~")
  def parentColumn: List[String] = List(this.debug) ++ parentOpt.map(_.parentColumn).getOrElse(List.empty[String])
  def suffixMatrix: List[List[String]] = List(parentColumn)
  def hasDerivation = derivation.nonEmpty
  def toWrapper = toDefinitive
  def toDefinitive = Definitive(this, sourceInfo)
}

case class DefinitiveValue[P](value: P, sourceInfo: SourceInfo) extends DefinitiveProxy[P] {
  var derivation: Option[DefinitiveDerivation] = None
  def compute:    Option[P] = Some(value)
  override def hasDerivation = true
  def proto = value
  def debug = sourceInfo.makeMessage(x => s"$x $parentDebug/Definitive($value, $sourceInfo)")
}

trait DefinitiveProtoProxy[P] extends DefinitiveProxy[P] {
  var derivation: Option[DefinitiveDerivation] = None

  def compute: Option[P] = {
    if (cache.containsKey("value")) cache.get("value").asInstanceOf[Some[P]]
    else {
      derivation match {
        case None => None
        case Some(d) =>
          d.compute match {
            case None => None
            case Some(v) =>
              namer.map(f => chisel3.experimental.noPrefix { f(v) })
              val ret = Some(v)
              cache.put("value", ret)
              ret.asInstanceOf[Some[P]]
          }
      }
    }
  }

  def proto = {
    val ret = compute
    require(ret.nonEmpty, s"Illegal access to proto on $this: value is not known")
    ret.get
  }
}

sealed trait ContextualProxy[P] extends Proxy[P] {
  def parentOpt: Option[Proxy[Any]]
  def parentDebug = parentOpt.map(_.debug).getOrElse("~")
  def protoContextual: ContextualProxy[P] = suffixProxyOpt.map(_.protoContextual).getOrElse(this)
  val suffixProxyOpt: Option[ContextualProxy[P]]
  def parentColumn: List[String] = List(this.debug) ++ parentOpt.map(_.parentColumn).getOrElse(List.empty[String])
  def suffixMatrix: List[List[String]] = List(parentColumn)
  var derivation: Option[ContextualDerivation]
  var isResolved: Boolean = false
  var name: String = ""
  def hasDerivation = derivation.nonEmpty
  def values: List[P]
  def setValue[H](value: P): Unit
  def toWrapper = toContextual
  def toContextual = new Contextual(this, sourceInfo)
  def compute[H](h: Hierarchy[H]): Option[P]
  def markResolved(): Unit = {
    isResolved = true
    suffixProxyOpt.map(p => p.markResolved())
  }
  val sourceInfo: SourceInfo
}

case class ContextualValue[P](value: P, sourceInfo: SourceInfo) extends ContextualProxy[P] {
  def parentOpt: Option[Proxy[Any]] = None
  def proto = value
  isResolved = true
  name = value.toString
  def values: List[P] = ??? // Should never call .values on a ContextualValue
  override def compute[H](h: Hierarchy[H]): Option[P] = {
    log(s"computing $this on $h")
    Some(value)
  }
  var derivation: Option[ContextualDerivation] = None
  override def hasDerivation = true
  def setValue[H](value: P): Unit = Unit
  val suffixProxyOpt: Option[ContextualProxy[P]] = None
  def debug = sourceInfo.makeMessage(x => s"$x $parentDebug/Contextual($identity, $value)")
}

trait ContextualUserProxy[P] extends ContextualProxy[P] {
  def parentOpt: Option[Proxy[Any]]

  var derivation:     Option[ContextualDerivation] = None
  val absoluteValues: mutable.ArrayBuffer[P] = mutable.ArrayBuffer[P]()

  val sourceInfo: SourceInfo
  def compute[H](h: Hierarchy[H]): Option[P] = {
    if (cache.containsKey(h.proxy)) cache.get(h.proxy).asInstanceOf[Some[P]]
    else {
      val absoluteValue = derivation match {
        case None =>
          suffixProxyOpt match {
            case None              =>
              log(s"$debug no derivation, no suffix!")
              None
            case Some(suffixProxy) =>
              log(s"$debug no derivation, suffix!")
              suffixProxy.compute(h)
          }
        case Some(p) => p.compute(h)
      }
      absoluteValue match {
        case None => None
        case Some(v) =>
          cache.put(h.proxy, Some(v))
          Some(v.asInstanceOf[P])
      }
    }
  }
  def setValue[H](value: P): Unit = {
    absoluteValues += value
    suffixProxyOpt.map(p => p.setValue(value))
  }

  def values: List[P] = {
    require(isResolved, s"$debug is not resolved $derivation")
    if (cache.containsKey("values")) cache.get("values").asInstanceOf[List[P]]
    else {
      val allValuesList = absoluteValues.toList
      cache.put("values", allValuesList)
      allValuesList
    }
  }
  // Returns value of this contextual, with all predecessor values computed.
  def proto: P = {
    if(values.size == 1) values.head else ???
  }
}

trait ContextualMockProxy[P] extends ContextualUserProxy[P] {
  isResolved = suffixProxyOpt.map(_.isResolved).getOrElse(false)
  if (isResolved) {
    absoluteValues ++= suffixProxyOpt.get.values
  }
}

trait ContextualProtoProxy[P] extends ContextualUserProxy[P] {
  def parentOpt:      Option[Proxy[Any]] = None
  val suffixProxyOpt: Option[ContextualProxy[P]] = None
  isResolved = false
}

//TODO: In order to have nested definitive's, e.g. a definitive case class, we need to avoid using the proto as the key in the lookup.
//trait InterfaceProxy[+P] extends RootProxy[P] {
//  def predecessor: DeclarationProxy[P]
//  def toImplementationProxy: ImplementationProxy[P]
//
//  def proto: P = predecessor.proto
//  def builder: Option[ImplementationBuilder[P]] = predecessor.builder
//  def predecessorOption = Some(predecessor)
//  def toHierarchy: Interface[P] = new Interface(this)
//  def canInstantiate: Boolean = true
//}
//
//trait ImplementationProxy[+P] extends RootProxy[P] {
//  def predecessor: InterfaceProxy[P]
//  def toDefinitionProxy: DefinitionProxy[P]
//
//  def proto: P = predecessor.proto
//  def builder: Option[ImplementationBuilder[P]] = None
//  def predecessorOption = Some(predecessor)
//  def toHierarchy: Implementation[P] = new Implementation(this)
//  def canInstantiate: Boolean = true
//}
//
//trait DefinitionProxy[+P] extends RootProxy[P] {
//  def predecessor: ImplementationProxy[P]
//
//  def proto: P = predecessor.proto
//  def builder: Option[ImplementationBuilder[P]] = None
//  def predecessorOption = Some(predecessor)
//  def toHierarchy: Definition[P] = new Definition(this)
//  //override def toSetter: HierarchySetter[P] = HierarchySetter(this)
//  def canInstantiate: Boolean = true
//}

/** DefinitionProxy implementation for all proto's which extend IsWrappable
  *
  * TODO Move to IsWrappable.scala
  * @param proto underlying object we are creating a proxy of
  */
final class IsWrappableDefinition[P](val underlying: Underlying[P]) extends DefinitionProxy[P] {
  def debug = "IsWrappableDefinition"
  def builder = None
}
object IsWrappableDefinition {
  val cache: IdentityHashMap[Any, Any] = new IdentityHashMap[Any, Any]()
  def apply[P](underlying: Underlying[P]): IsWrappableDefinition[P] = {
    val key = underlying.proto
    if (cache.containsKey(key)) cache.get(key).asInstanceOf[IsWrappableDefinition[P]] else {
      val ret = new IsWrappableDefinition(underlying)
      cache.put(key, ret)
      ret
    }
  }
}

/** Transparent implementation for all proto's which extend IsWrappable
  *
  * Note: Clone is not needed for IsWrappables, as they cannot be instantiated via Instance(..)
  *
  * TODO Move to IsWrappable.scala
  * @param proto underlying object we are creating a proxy of
  */
final class IsWrappableTransparent[P](val suffixProxy: IsWrappableDefinition[P]) extends Transparent[P] {
  def debug = "IsWrappableTransparent"
}
object IsWrappableTransparent {
  val cache: IdentityHashMap[Any, Any] = new IdentityHashMap[Any, Any]()
  def apply[P](suffixProxy: IsWrappableDefinition[P]): IsWrappableTransparent[P] = {
    val key = suffixProxy
    if (cache.containsKey(key)) cache.get(key).asInstanceOf[IsWrappableTransparent[P]] else {
      val ret = new IsWrappableTransparent(suffixProxy)
      cache.put(key, ret)
      ret
    }
  }
}

/** Mock implementation for all proto's which extend IsWrappable
  *
  * Note: Clone is not needed for IsWrappables, as they cannot be instantiated via Instance(..)
  *
  * TODO Move to IsWrappable.scala
  * @param proto underlying object we are creating a proxy of
  */
final class IsWrappableMock[P](val suffixProxy: InstanceProxy[P], val parent: Proxy[Any]) extends Mock[P] {
  def debug = "IsWrappableMock"
}
object IsWrappableMock {
  val cache: IdentityHashMap[Any, Any] = new IdentityHashMap[Any, Any]()
  def apply[P](suffixProxy: InstanceProxy[P], parent: Proxy[Any]): IsWrappableMock[P] = {
    suffixProxy.mock(suffixProxy, parent)(new IsWrappableMock(suffixProxy, parent))
    //val key = (suffixProxy, parent)
    //if (cache.containsKey(key)) cache.get(key).asInstanceOf[IsWrappableMock[P]] else {
    //  val ret = new IsWrappableMock(suffixProxy, parent)
    //  cache.put(key, ret)
    //  ret
    //}
  }
}
