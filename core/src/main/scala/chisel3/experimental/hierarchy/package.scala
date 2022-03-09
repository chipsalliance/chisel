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
    *   1. IsContext
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

  implicit val mg = new chisel3.internal.MacroGenerated{}
  implicit val info = chisel3.internal.sourceinfo.UnlocatableSourceInfo
  implicit val opt = chisel3.ExplicitCompileOptions.Strict

  // TYPECLASS Basics

  // TYPECLASSES for BaseModule
  // Required by Definition(..)
  implicit def buildable[T <: BaseModule](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) = new ProxyDefiner[T] {
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
  
  implicit def stampable[T <: BaseModule](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) = new ProxyInstancer[T] {
    def apply(definition: Definition[T]): ModuleClone[T] = {
      val ports = experimental.CloneModuleAsRecord(definition)
      val clone = ports._parent.get.asInstanceOf[ModuleClone[T]]
      clone._madeFromDefinition = true
      clone
    }
  }

  implicit def mockerBaseModule[U <: BaseModule](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) = new Mocker[U, BaseModule] {
    def apply(newValue: HierarchicalProxy[U, BaseModule], parent: BaseModule): Mock[U, BaseModule] =
      ModuleMock(newValue.asInstanceOf[HierarchicalProxy[U, BaseModule] with BaseModule], parent)
  }
  //implicit def mockerIsInstantiable[U <: IsInstantiable] = new Mocker[U, BaseModule] {
  //  def apply(newValue: InstanceProxy[U, BaseModule], parent: BaseModule): BaseModule = 
  //    InstantiableMock(newValue, parent)
  //}

  implicit def proxifierBaseModule(implicit mocker: Mocker[BaseModule, BaseModule]) = new Proxifier[BaseModule] {
    def apply[P](protoValue: BaseModule, proxy: Proxy[P]): BaseModule with Proxy[BaseModule] = {
      protoValue._parent match {
        case Some(p) =>
          val ip = asInstance(protoValue).proxy.asInstanceOf[InstanceProxy[BaseModule, BaseModule]]
          proxy.lookup(ip)(mocker, mocker, this).asInstanceOf[BaseModule with Proxy[BaseModule]]
        case None => ModuleDefinition(protoValue, protoValue.getCircuit)
      }
    }
  }

  implicit def lookupModule[V <: BaseModule](implicit mocker: Mocker[V, BaseModule], proxifier: Proxifier[BaseModule]) = new Lookuper[V] {
    type R = Instance[V]
    // Note if value is a Proxy, we are assuming V is BaseModule, not a specific Proxy type
    // If this is not the case, its an internal error and we should get a dynamic error
    def apply[P](value: V, proxy: Proxy[P]) = {
      (value, value._parent) match {
        case (p: Proxy[_], _) => println(p);??? // error
        case (v, None)        => println(v);??? // error! should be definition
        case (v: BaseModule, Some(p)) =>
          proxifier(v, proxy).asInstanceOf[InstanceProxy[V, BaseModule]].toInstance
      }
    }
  }
  def asInstance(proto: BaseModule):   core.Instance[BaseModule] = {
    require(proto._parent.nonEmpty, s"Cannot call .asInstance on $proto because it has no parent! Try .toDefinition?")
    proto match {
      case i: InstanceProxy[BaseModule, BaseModule] with BaseModule => i.toInstance
      case d: DefinitionProxy[BaseModule] with BaseModule => ??? //should be unreachable
      case b: BaseModule =>
        val definition = proto.toDefinition
        ModuleTransparent(definition.proxy.asInstanceOf[ModuleDefinition[BaseModule]]).toInstance
    }
  }

  implicit def lookupInstance[U <: BaseModule](implicit mocker: Mocker[U, BaseModule], cmocker: Mocker[BaseModule, BaseModule], proxifier: Proxifier[BaseModule]) = new Lookuper[Instance[U]] {
    type C = BaseModule
    type R = Instance[U]
    def apply[P](value: Instance[U], proxy: Proxy[P]) = {
      proxy.ilookup(value.proxy.asInstanceOf[InstanceProxy[U, C] with C])(mocker, cmocker, proxifier).toInstance
    }
  }

  implicit def lookupIsInstantiable[U <: IsInstantiable](implicit proxifier: Proxifier[BaseModule]) = new Lookuper[U] {
    type C = BaseModule
    type R = Instance[U]
    def apply[P](value: U, proxy: Proxy[P]): Instance[U] = {
      def getParent[X](i: X): BaseModule = i match {
        case p: BaseModule => p
        case InstantiableProxy(_, p) => getParent(p)
      }
      val p = getParent(proxy)
      println(p)
      InstantiableProxy(value, p).toInstance
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
    if(mem._parent == Some(newParent)) mem else {
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

  implicit def lookuperData[V <: Data](
    implicit sourceInfo: SourceInfo,
    compileOptions: CompileOptions,
    mocker: Mocker[BaseModule, BaseModule],
    proxifier: Proxifier[BaseModule],
  ) = new Lookuper[V] {
    type C = BaseModule
    type R = V
    def apply[P](value: V, proxy: Proxy[P]) = if(value._parent.isEmpty) value else ((proxy, proxy.proto == value._parent.get) match {
      case (t: ModuleClone[_], true)        => cloneData(value, t, t.ioMap)
      case (t: ModuleTransparent[_], true)  => value
      case (t: ModuleTransparent[_], false) => value
      case (t: ModuleDefinition[_], true)   => cloneData(value, t)
      case (t: ModuleDefinition[_], false)  =>
        val newParent = t.blookup(value._parent.get)(mocker, mocker, proxifier, proxifier)
        cloneData(value, newParent.asInstanceOf[BaseModule])
      case (t: ModuleMock[_], true)         => cloneData(value, t)
      case (t: ModuleMock[_], false)        =>
        val newParent = t.blookup(value._parent.get)(mocker, mocker, proxifier, proxifier)
        cloneData(value, newParent.asInstanceOf[BaseModule])
      case (InstantiableProxy(_, parent: Proxy[_]), _) => apply(value, parent)
    })
  }

  implicit def lookuperMem[V <: MemBase[_]](
    implicit sourceInfo: SourceInfo,
    compileOptions: CompileOptions,
    mocker: Mocker[BaseModule, BaseModule],
    proxifier: Proxifier[BaseModule],
  ) = new Lookuper[V] {
    type C = BaseModule
    type R = V
    def apply[P](value: V, proxy: Proxy[P]) = if(value._parent.isEmpty) value else ((proxy, proxy.proto == value._parent.get) match {
      case (t: ModuleClone[_], true)        => cloneMem(value, t)
      case (t: ModuleTransparent[_], true)  => value
      case (t: ModuleTransparent[_], false) => value
      case (t: ModuleDefinition[_], true)   => cloneMem(value, t)
      case (t: ModuleDefinition[_], false)  =>
        val newParent = t.blookup(value._parent.get)(mocker, mocker, proxifier, proxifier)
        cloneMem(value, newParent.asInstanceOf[BaseModule])
      case (t: ModuleMock[_], true)         => cloneMem(value, t)
      case (t: ModuleMock[_], false)        =>
        val newParent = t.blookup(value._parent.get)(mocker, mocker, proxifier, proxifier)
        cloneMem(value, newParent.asInstanceOf[BaseModule])
      case (InstantiableProxy(_, parent: Proxy[_]), _) => apply(value, parent)
    })
  }

  //implicit def lookuperIsInstantiable[V <: IsInstantiable](
  //  implicit sourceInfo: SourceInfo,
  //  compileOptions:      CompileOptions,
  //  proxifier: Proxifier[V]
  //) = new Lookuper[V] {
  //  type R = Instance[V]
  //  def apply[P](value: V, hierarchy: core.Hierarchy[P]): R = {
  //    new Instance(hierarchy._lookup(_ => proxifier(value)).asInstanceOf[StandIn[V]])
  //  }
  //}

  //type Definition[P] = core.Definition[P]
  //type Instance[P] = core.Instance[P]
  //type Hierarchy[P] = core.Hierarchy[P]
  //type IsContext = core.IsContext


  //def baseModuleContexterBuilder[V](): Contexter[V, BaseModule] = new Contexter[V, BaseModule] {
  //  def apply[P](value: V, hierarchy: Hierarchy[P]): R = {
  //    hierarchy.proxy.lookupContext match {
  //      case Context(Some(value: BaseModule)) => Context(Some(value))
  //      case _ => Context(None)
  //    }
  //  }
  //}
  //implicit def contexterData[V <: Data] = baseModuleContexterBuilder[V]()
  //implicit def contexterMem[V <: MemBase[_]] = baseModuleContexterBuilder[V]()
  //implicit def contexterModule[V <: BaseModule] = baseModuleContexterBuilder[V]()
  //implicit def contexterInstanceModule[V <: BaseModule] = baseModuleContexterBuilder[Instance[V]]()
  //implicit def contexterIsInstantiable[V <: IsInstantiable] = baseModuleContexterBuilder[V]()



  // ========= Extensions =========

  implicit class BaseModuleExtensions[P <: BaseModule](proto: P) {
    import chisel3.experimental.hierarchy.core.{Definition, Instance}
    // Require proto is not a ContextStandIn, as ContextStandIn should always implement toInstance/toDefinition
    //require(!proto.isInstanceOf[ContextStandIn[_]], s"Cannot have $proto be a ContextStandIn, must be a proto!!")
    def asInstanceProxy: core.InstanceProxy[P, BaseModule] = {
      require(proto._parent.nonEmpty, s"Cannot call .asInstance on $proto because it has no parent! Try .toDefinition?")
      proto match {
        case i: Proxy[_] => ??? //should be unreachable
        case b: P =>
          val definition = toDefinition
          ModuleTransparent(definition.proxy.asInstanceOf[ModuleDefinition[P]])
      }
    }
    def asInstance:   core.Instance[P] = asInstanceProxy.toInstance
    def toDefinition: core.Definition[P] = Definition(ModuleDefinition(proto, proto.getCircuit))
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
    def toInstance: Instance[T] = ???
    def toDefinition: Definition[T] = toInstance.toDefinition
  }

}
