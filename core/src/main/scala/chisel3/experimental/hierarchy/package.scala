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
    *
    * @example {{{
    * @instantiable
    * class MyModule extends Module {
    *   ...
    * }
    *
    * val d = Definition(new MyModule)
    * val i0 = Instance(d)
    * val i1 = Instance(d)
    * }}}
    */
  class instantiable extends chisel3.internal.instantiable

  /** Classes marked with [[@instantiable]] can have their vals marked with the [[@public]] annotation to
    * enable accessing these values from a [[Definition]] or [[Instance]] of the class.
    *
    * Only vals of the the following types can be marked [[@public]]:
    *   1. IsInstantiable
    *   2. IsLookupable
    *   3. Data
    *   4. BaseModule
    *   5. Iterable/Option containing a type that meets these requirements
    *   6. Basic type like String, Int, BigInt etc.
    *
    * @example {{{
    * @instantiable
    * class MyModule extends Module {
    *   @public val in = IO(Input(UInt(3.W)))
    *   @public val out = IO(Output(UInt(3.W)))
    *   ..
    * }
    *
    * val d = Definition(new MyModule)
    * val i0 = Instance(d)
    * val i1 = Instance(d)
    *
    * i1.in := i0.out
    * }}}
    */
  class public extends chisel3.internal.public

  implicit val mg = new chisel3.internal.MacroGenerated {}
  implicit val info = chisel3.internal.sourceinfo.UnlocatableSourceInfo
  implicit val opt = chisel3.ExplicitCompileOptions.Strict

  // TYPECLASS Basics

  // TYPECLASSES for BaseModule
  // Required by Definition(..)
  implicit def buildable[T <: BaseModule](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) =
    new ProxyDefiner[T] {
      def apply(proto: => T): ModuleDefinition[T] = {
        val dynamicContext = new DynamicContext(Nil, Builder.captureContext().throwOnFirstError)
        Builder.globalNamespace.copyTo(dynamicContext.globalNamespace)
        dynamicContext.inDefinition = true
        val (ir, module) = Builder.build(Module(proto), dynamicContext, false)
        Builder.components ++= ir.components
        Builder.annotations ++= ir.annotations
        module._circuit = Builder.currentModule
        dynamicContext.globalNamespace.copyTo(Builder.globalNamespace)
        ModuleDefinition(module, module._circuit)
      }
    }

  implicit def stampable[T <: BaseModule](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) =
    new ProxyInstancer[T] {
      def apply(definition: Definition[T], contexts: Seq[RootContext[T]]): ModuleClone[T] = {
        val ports = experimental.CloneModuleAsRecord(definition, contexts)
        val clone = ports._parent.get.asInstanceOf[ModuleClone[T]]
        clone._madeFromDefinition = true
        clone
      }
    }

  // There is an implicit resolution bug which causes this lookupable to
  //   be confused with lookupableIterable
  implicit val lookupableString = new Lookupable.SimpleLookupable[String] {}

  // Non-implicit lookup of BaseModule. This has to be different than the
  // implicit def on subclasses of BaseModule because Proxy's which also
  // extend BaseModule will have a different return type. Instead of
  // returning Hierarchy[Proxy[M]], we'd need to return Hierarchy[M].
  // This usecase only occurs when looking up parents of other proto
  // values, so always returning Hierarchy[BaseModule] is acceptable.
  object lookupBaseModule extends Lookupable[BaseModule] {
    type R = Hierarchy[BaseModule]
    type S = Context[BaseModule]
    type G = Context[BaseModule]
    def setter[P](value: BaseModule, context: Context[P]): S = {
      NestedContext(apply(value, context.toHierarchy).asInstanceOf[Instance[BaseModule]].proxy, context.root)
    }
    def getter[P](value: BaseModule, context: Context[P]): G = {
      NestedContext(apply(value, context.toHierarchy).asInstanceOf[Instance[BaseModule]].proxy, context.root)
    }
    def apply[P](value: BaseModule, hierarchy: Hierarchy[P]): Hierarchy[BaseModule] = {
      require(!value.isInstanceOf[Proxy[_]], "BAD!")
      val h = hierarchy.getLineageOf { case h: Hierarchy[BaseModule] if h.isA[BaseModule] => h } match {
        case Some(h: Hierarchy[BaseModule]) => h
        case None => hierarchy
      }
      value match {
        case v: Proxy[BaseModule] if v.proto == h.proto => h.asInstanceOf[Hierarchy[BaseModule]]
        case v: BaseModule if v == h.proto              => h.asInstanceOf[Hierarchy[BaseModule]]
        case other =>
          value._parent match {
            case None    => value.toDefinition
            case Some(p) => apply(value, h._lookup(_ => p)(lookupBaseModule, mg))
          }
      }
    }
  }

  implicit def lookupModule[V <: BaseModule] = new Lookupable[V] {
    type R = Instance[V]
    type S = Context[V]
    type G = Context[V]
    def setter[P](value: V, context: Context[P]): S = {
      NestedContext(apply(value, context.toHierarchy).asInstanceOf[Instance[V]].proxy, context.root)
    }
    def getter[P](value: V, context: Context[P]): G = {
      NestedContext(apply(value, context.toHierarchy).asInstanceOf[Instance[V]].proxy, context.root)
    }
    // Note if value is a Proxy, we are assuming V is BaseModule, not a specific Proxy type
    // If this is not the case, its an internal error and we should get a dynamic error
    def apply[P](value: V, hierarchy: Hierarchy[P]) = {
      require(!value.isInstanceOf[Proxy[_]], "BAD!")
      (value._parent, value._parent == Some(hierarchy.proto), hierarchy) match {
        case (None, _, _) => println(value); ??? //ERROR
        case (Some(p), true, h: Hierarchy[BaseModule]) =>
          // Create Mock, hierarchy proxy is parent
          val d = ModuleDefinition(value)
          val t = ModuleTransparent(d)
          val contexts = h.proxy.contexts.map { l: Context[BaseModule] =>
            l.getter(value)(this).asInstanceOf[Context[V]]
          }
          ModuleMock(t, h.proxyAs[BaseModule], contexts).toInstance
        case (Some(p), false, h: Hierarchy[P]) =>
          // Create Mock, newParentHierarchy proxy is parent
          val newParentHierarchy = lookupBaseModule(p, hierarchy)
          val d = ModuleDefinition(value)
          val t = ModuleTransparent(d)
          val contexts = newParentHierarchy.proxy.contexts.map { l: Context[BaseModule] =>
            l.getter(value)(this).asInstanceOf[Context[V]]
          }
          ModuleMock(t, newParentHierarchy.proxyAs[BaseModule], contexts).toInstance
      }
    }
  }

  implicit def lookupInstance[U <: BaseModule] = new Lookupable[Instance[U]] {
    type R = Instance[U]
    type S = Context[U]
    type G = Context[U]
    def setter[P](value: Instance[U], context: Context[P]): S = {
      NestedContext(apply(value, context.toHierarchy).asInstanceOf[Instance[U]].proxy, context.root)
    }
    def getter[P](value: Instance[U], context: Context[P]): G = {
      NestedContext(apply(value, context.toHierarchy).asInstanceOf[Instance[U]].proxy, context.root)
    }
    def apply[P](value: Instance[U], hierarchy: Hierarchy[P]) = {
      value.proxyAs[BaseModule]._parent match {
        case None => println(value); ??? //ERROR, should be a definition?
        case Some(p) =>
          val newParentHierarchy = p match {
            case b: Proxy[BaseModule] if b.proto == hierarchy.proto => hierarchy
            case b: BaseModule if b == hierarchy.proto              => hierarchy
            case other => lookupBaseModule(p, hierarchy)
          }
          val contexts = newParentHierarchy.proxy.contexts.map { l: Context[_] =>
            l.getter(value)(this).asInstanceOf[Context[U]]
          }
          // Create mock, set up genesis etc with h as parent
          ModuleMock(value.proxyAs[BaseModule], newParentHierarchy.proxyAs[BaseModule], contexts).toInstance
      }
    }
  }

  def cloneData[D <: Data](data: D, newParent: BaseModule, ioMap: Map[Data, Data] = Map.empty[Data, Data]): D = {
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

  def cloneMem[M <: MemBase[_]](mem: M, newParent: BaseModule): M = {
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

  implicit def lookupableData[V <: Data](
    implicit sourceInfo: SourceInfo,
    compileOptions:      CompileOptions
  ) = new Lookupable[V] {
    type R = V
    type S = V
    type G = V
    def setter[P](value: V, context:   Context[P]): S = apply(value, context.toHierarchy)
    def getter[P](value: V, context:   Context[P]): G = apply(value, context.toHierarchy)
    def apply[P](value:  V, hierarchy: Hierarchy[P]): V = value._parent match {
      case None => value
      case Some(p: BaseModule) =>
        val newParentHierarchy = p match {
          case b: Proxy[_] if b.proto == hierarchy.proto => hierarchy
          case b: BaseModule if b == hierarchy.proto     => hierarchy
          case _ => lookupBaseModule(p, hierarchy)
        }
        newParentHierarchy.proxy match {
          case m: ModuleClone[_]       => cloneData(value, m, m.ioMap)
          case m: ModuleTransparent[_] => value
          case m: ModuleMock[_]        => cloneData(value, m)
          case m: ModuleDefinition[_]  => cloneData(value, m)
          case _ => value
        }
    }
  }

  implicit def lookupableMem[V <: MemBase[_]](
    implicit sourceInfo: SourceInfo,
    compileOptions:      CompileOptions
  ) = new Lookupable[V] {
    type R = V
    type S = V
    type G = V
    def setter[P](value: V, context:   Context[P]): S = apply(value, context.toHierarchy)
    def getter[P](value: V, context:   Context[P]): G = apply(value, context.toHierarchy)
    def apply[P](value:  V, hierarchy: Hierarchy[P]) = value._parent match {
      case None => value
      case Some(p: BaseModule) =>
        val newParentHierarchy = p match {
          case b: Proxy[BaseModule] if b.proto == hierarchy.proto => hierarchy
          case b: BaseModule if b == hierarchy.proto              => hierarchy
          case _ => lookupBaseModule(p, hierarchy)
        }
        newParentHierarchy.proxy match {
          case m: ModuleClone[_]       => cloneMem(value, m)
          case m: ModuleTransparent[_] => value
          case m: ModuleMock[_]        => cloneMem(value, m)
          case m: ModuleDefinition[_]  => cloneMem(value, m)
        }
    }
  }

  // Exposing core hierarchy types to enable importing just hierarchy, not hierarchy.core

  type Hierarchy[P] = core.Hierarchy[P]
  val Hierarchy = core.Hierarchy
  type Instance[P] = core.Instance[P]
  val Instance = core.Instance
  type Definition[P] = core.Definition[P]
  val Definition = core.Definition
  type Contextual[P] = core.Contextual[P]
  val Contextual = core.Contextual
  type IsLookupable = core.IsLookupable
  type IsInstantiable = core.IsInstantiable

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
    def toDefinition: core.Definition[P] = core.Definition(ModuleDefinition(proto, proto.getCircuit))
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
    def toDefinition: Definition[T] = toInstance.toDefinition
  }

}
