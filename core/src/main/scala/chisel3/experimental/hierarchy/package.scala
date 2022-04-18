// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental
import chisel3._
import chisel3.internal.{Builder, DynamicContext}
import chisel3.internal.sourceinfo.SourceInfo
import chisel3.experimental.BaseModule
import chisel3.experimental.hierarchy.core._
import chisel3.experimental.dataview.{isView, reify, reifySingleData}
import chisel3.internal.firrtl.{Arg, ILit, Index, Slot, ULit}
import chisel3.internal.{throwException, AggregateViewBinding, Builder, ChildBinding, ViewBinding, ViewParent}
import _root_.firrtl.annotations._

package object hierarchy {
  import chisel3.experimental.hierarchy.Utils._

  /** Classes or traits which will be used with the [[Definition]] + [[Instance]] api should be marked
    * with the [[@instantiable]] annotation at the class/trait definition.
@@ -45,4 +57,254 @@ package object hierarchy {
    * }}}
    */
  class public extends chisel3.internal.public

  implicit val mg = new chisel3.internal.MacroGenerated {}
  implicit val info = chisel3.internal.sourceinfo.UnlocatableSourceInfo
  implicit val opt = chisel3.ExplicitCompileOptions.Strict

  // TYPECLASS Basics

  def buildExtension[V <: BaseModule](isBaseModule: Boolean): HierarchicalExtensions[V, BaseModule] = new HierarchicalExtensions[V, BaseModule] {
    def getParent(value: V): Option[BaseModule] = value._parent
    def getProxyParent(value: Proxy[V]): Option[BaseModule] = value.asInstanceOf[BaseModule]._parent

    def buildDeclaration(proto: => V): ModuleDeclaration[V] = {
      val dynamicContext = new DynamicContext(Nil, Builder.captureContext().throwOnFirstError)
      Builder.globalNamespace.copyTo(dynamicContext.globalNamespace)
      dynamicContext.inDefinition = true
      val (ir, module) = Builder.build(Module(proto), dynamicContext, false)
      Builder.components ++= ir.components
      Builder.annotations ++= ir.annotations
      module._circuit = Builder.currentModule
      dynamicContext.globalNamespace.copyTo(Builder.globalNamespace)
      ModuleDeclaration(module, module._circuit)
    }
    def buildInstance(root: Root[V]): ModuleClone[V] = {
      val ports = experimental.CloneModuleAsRecord(root)
      val clone = ports._parent.get.asInstanceOf[ModuleClone[V]]
      clone._madeFromDefinition = true
      clone
    }
    //def define(proto: V): ModuleDefinition[V] = ???
    //def transparent(proto: V): ModuleTransparent[V] = ???
    def clone[P](value: Hierarchy[V], hierarchy: Hierarchy[P]): InstanceProxy[V] = {
      (value, hierarchy.proxy) match {
        case (d: Root[V], t: ModuleTransparent[V]) =>
          val ports = experimental.CloneModuleAsRecord(d)
          val clone = ports._parent.get.asInstanceOf[ModuleClone[V]]
          clone._madeFromDefinition = true
          clone
        case (i: Instance[V], _) =>
          ModuleMock(i.proxyAs[BaseModule], hierarchy.proxyAs[BaseModule])
      }
    }
    def mockInstance[P](value: Instance[V], parent: Hierarchy[P]): Instance[V] = {
      ModuleMock(value.proxyAs[BaseModule], parent.proxyAs[BaseModule]).toInstance
    }
    def mockValue[P](value: V, parent: Hierarchy[P]): Instance[V] = {
      val d = (new ModuleDeclaration(value, None)).toDefinition
      val t = ModuleTransparent(d.proxyAs[ModuleRoot[V]])
      ModuleMock(t, parent.proxyAs[BaseModule]).toInstance
    }
    def parentSelection: PartialFunction[Any, Hierarchy[BaseModule]] = {
      case h: Hierarchy[BaseModule] if h.isA[BaseModule] => h
    }
    def toDefinition(value: V): Definition[V] = value.toDefinition
    def parentExtensions: HierarchicalExtensions[BaseModule, BaseModule] = {
      if(isBaseModule) this.asInstanceOf[HierarchicalExtensions[BaseModule, BaseModule]] else buildExtension[BaseModule](true)
    }
  }

  implicit def moduleExtensions[V <: BaseModule]: HierarchicalExtensions[V, BaseModule] = buildExtension[V](false)
  implicit def dataExtensions[V <: Data] = new CloneableExtensions[V, BaseModule] {
    def parentExtensions: HierarchicalExtensions[BaseModule, BaseModule] = buildExtension[BaseModule](true)
    def getParent(value: V): Option[BaseModule] = value._parent
    def parentSelection: PartialFunction[Any, Hierarchy[BaseModule]] = {
      case h: Hierarchy[BaseModule] if h.isA[BaseModule] => h
    }
    def clone(value: V, hierarchy: Hierarchy[BaseModule]): V = hierarchy.proxy match {
      case m: ModuleClone[_]       => cloneData(value, m, m.ioMap)
      case m: ModuleTransparent[_] => value
      case m: ModuleMock[_]        => cloneData(value, m)
      case m: ModuleDefinition[_]  => cloneData(value, m)
      case _ => value
    }
    private def cloneData[D <: Data](data: D, newParent: BaseModule, ioMap: Map[Data, Data] = Map.empty[Data, Data]): D = {
      def impl[C <: Data](d: C): C = d match {
        case x: Data if ioMap.contains(x) => ioMap(x).asInstanceOf[C]
        case x if x._parent == Some(newParent) => x
        case x =>
          val newData = x.cloneTypeFull
          newData.setRef(x.getRef, true)
          newData.bind(internal.CrossModuleBinding)
          newData.setAllParents(Some(newParent))
          newData
      }
      data.binding match {
        case Some(_: ChildBinding) => Utils.mapRootAndExtractSubField(data, impl)
        case _ => impl(data)
      }
    }
  }
  implicit def memExtensions[V <: MemBase[_]] = new CloneableExtensions[V, BaseModule] {
    def parentExtensions: HierarchicalExtensions[BaseModule, BaseModule] = buildExtension[BaseModule](true)
    def getParent(value: V): Option[BaseModule] = value._parent
    def parentSelection: PartialFunction[Any, Hierarchy[BaseModule]] = {
      case h: Hierarchy[BaseModule] if h.isA[BaseModule] => h
    }
    def clone(value: V, hierarchy: Hierarchy[BaseModule]): V = hierarchy.proxy match {
      case m: ModuleClone[_]       => cloneMem(value, m)
      case m: ModuleTransparent[_] => value
      case m: ModuleMock[_]        => cloneMem(value, m)
      case m: ModuleDefinition[_]  => cloneMem(value, m)
      case _ => value
    }

    private def cloneMem[M <: MemBase[_]](mem: M, newParent: BaseModule): M = {
      if (mem._parent == Some(newParent)) mem
      else {
        val existingMod = Builder.currentModule
        Builder.currentModule = Some(newParent)
        val newChild: M = mem match {
          case m: Mem[_] => new Mem(m.t.asInstanceOf[Data].cloneTypeFull, m.length).asInstanceOf[M]
          case m: SyncReadMem[_] =>
            new SyncReadMem(m.t.asInstanceOf[Data].cloneTypeFull, m.length, m.readUnderWrite).asInstanceOf[M]
        }
        Builder.currentModule = existingMod
        newChild.setRef(mem.getRef, true)
        newChild
      }
    }
  }

  // There is an implicit resolution bug which causes this lookupable to
  //   be confused with lookupableIterable
  implicit val lookupableString = new Lookupable.SimpleLookupable[String] {}

  // Exposing core hierarchy types to enable importing just hierarchy, not hierarchy.core

  type Hierarchy[P] = core.Hierarchy[P]
  type Root[P] = core.Root[P]
  type Instance[P] = core.Instance[P]
  val Instance = core.Instance
  type Definition[P] = core.Definition[P]
  val Definition = core.Definition
  type Declaration[P] = core.Declaration[P]
  val Declaration = core.Declaration
  type Interface[P] = core.Interface[P]
  val Interface = core.Interface
  type Implementation[P] = core.Implementation[P]
  val Implementation = core.Implementation
  type Definitive[V] = core.Definitive[V]
  val Definitive = core.Definitive
  type IsLookupable = core.IsLookupable
  type IsInstantiable = core.IsInstantiable
  type ImplementationBuilder[P] = core.ImplementationBuilder[P]
  val Folder = core.Folder
  type Folder[P] = core.Folder[P]

  // ========= Extensions =========

  implicit class BaseModuleExtensions[P <: BaseModule](proto: P) {
    import chisel3.experimental.hierarchy.core.{Definition, Instance}
    def asInstanceProxy: core.InstanceProxy[P] = {
      require(proto._parent.nonEmpty, s"Cannot call .asInstance on $proto because it has no parent! Try .toDefinition?")
      proto match {
        case i: Proxy[_] => ??? //should be unreachable
        case b: P =>
          val definition = toDefinition
          ModuleTransparent(definition.proxy.asInstanceOf[ModuleDefinition[P]])
      }
    }
    def toInstance:   core.Instance[P] = asInstanceProxy.toInstance
    def toDefinition: core.Definition[P] = ModuleDeclaration(proto, proto.getCircuit).toDefinition
  }
  implicit class HierarchyBaseModuleExtensions[T <: BaseModule](i: core.Hierarchy[T]) {

    /** Returns the toTarget of this hierarchy
      * @return target of this hierarchy
      */
    def toTarget: IsModule = i match {
      case d: Definition[T] => new DefinitionBaseModuleExtensions(d).toTarget
      case i: Instance[T]   => new InstanceBaseModuleExtensions(i).toTarget
    }

    /** Returns the toAbsoluteTarget of this hierarchy
      * @return absoluteTarget of this Hierarchy
      */
    def toAbsoluteTarget: IsModule = i match {
      case d: Definition[T] => new DefinitionBaseModuleExtensions(d).toAbsoluteTarget
      case i: Instance[T]   => new InstanceBaseModuleExtensions(i).toAbsoluteTarget
    }
  }
  implicit class InstanceBaseModuleExtensions[T <: BaseModule](i: Instance[T]) {

    /** If this is an instance of a Module, returns the toTarget of this instance
      * @return target of this instance
      */
    def toTarget: IsModule = i.proxy match {
      case x: Proxy[BaseModule] with BaseModule => x.getTarget
    }

    /** If this is an instance of a Module, returns the toAbsoluteTarget of this instance
      * @return absoluteTarget of this instance
      */
    def toAbsoluteTarget: IsModule = i.proxy match {
      case x: Proxy[BaseModule] with BaseModule => x.toAbsoluteTarget
    }
  }
  implicit class DefinitionBaseModuleExtensions[T <: BaseModule](d: Definition[T]) {

    /** If this is an instance of a Module, returns the toTarget of this instance
      * @return target of this instance
      */
    def toTarget: ModuleTarget = d.proto.toTarget

    /** If this is an instance of a Module, returns the toAbsoluteTarget of this instance
      * @return absoluteTarget of this instance
      */
    def toAbsoluteTarget: IsModule = d.proto.toTarget
  }

  implicit class IsInstantiableExtensions[T <: IsInstantiable](i: T) {

    /** If this is an instance of a Module, returns the toTarget of this instance
      * @return target of this instance
      */
    def toInstance:   Instance[T] = ???
    def toRoot: Root[T] = toInstance.toRoot
  }

}