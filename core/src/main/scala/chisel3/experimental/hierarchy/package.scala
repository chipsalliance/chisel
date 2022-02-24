package chisel3.experimental
import chisel3._
import chisel3.internal.{Builder, DynamicContext}
import chisel3.internal.sourceinfo.SourceInfo
import chisel3.experimental.BaseModule
import chisel3.experimental.hierarchy.core._
import chisel3.experimental.dataview.isView
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


  // TYPECLASS Basics

  // TYPECLASSES for BaseModule
  // Required by Definition(..)
  implicit def buildable[T <: BaseModule](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) = new ProxyDefiner[T] {
    def apply(proto: => T): Proxy[T] = {
      val dynamicContext = new DynamicContext(Nil, Builder.captureContext().throwOnFirstError)
      Builder.globalNamespace.copyTo(dynamicContext.globalNamespace)
      dynamicContext.inDefinition = true
      val (ir, module) = Builder.build(Module(proto), dynamicContext, false)
      Builder.components ++= ir.components
      Builder.annotations ++= ir.annotations
      module._circuit = Builder.currentModule
      dynamicContext.globalNamespace.copyTo(Builder.globalNamespace)
      StandIn(StandInDefinition(module, module._circuit))
    }
  }
  
  implicit def stampable[T <: BaseModule](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) = new ProxyInstancer[T] {
    def apply(definition: Definition[T]): Proxy[T] = {
      val ports = experimental.CloneModuleAsRecord(definition)
      val clone = ports._parent.get.asInstanceOf[StandInModule[T]]
      clone._madeFromDefinition = true
      StandIn(clone)
    }
  }

  implicit def proxifierInstance[M <: BaseModule](
    implicit sourceInfo: SourceInfo,
    compileOptions: CompileOptions,
    contexter: Contexter[Instance[M], BaseModule]
  ) = new Proxifier[Instance[M]] {
    type U = M
    def apply[P](value: Instance[M], hierarchy: core.Hierarchy[P]) = {
      cloneModuleToContext(value.proxy, contexter(value, hierarchy).context.get.asInstanceOf[BaseModule])
    }
  }

  implicit def proxifierModule[V <: BaseModule](
    implicit sourceInfo: SourceInfo,
    compileOptions: CompileOptions,
    contexter: Contexter[V, BaseModule]
  ) = new Proxifier[V] {
      type U = V
      def apply[P](value: V, hierarchy: core.Hierarchy[P]) = contexter(value, hierarchy) match {
        case Context(None) => value.asProxy.asInstanceOf[this.R]
        case Context(Some(p: BaseModule)) => cloneModuleToContext(value.asProxy, p).asInstanceOf[this.R]
      }
    }

  implicit def contextualizerData[V <: Data](
    implicit sourceInfo: SourceInfo,
    compileOptions: CompileOptions,
    contexter: Contexter[V, BaseModule]
  ) =
    new Contextualizer[V] {
      type R = V
      def apply[P](value: V, hierarchy: core.Hierarchy[P]): R = {
        val ioMap: Option[Map[Data, Data]] = hierarchy.proxy match {
          case StandIn(x: StandInModule[_]) => Some(x.ioMap)
          case Proto(x: BaseModule, _) => Some(x.getChiselPorts.map { case (_, data) => data -> data }.toMap)
          case m => None
        }
        if (isView(value)) {
          cloneViewToContext(value, ioMap, contexter(value, hierarchy).context)
        } else {
          doLookupData(value, ioMap, contexter(value, hierarchy).context.get)
        }
      }
    }

  implicit def contextualizerMem[V <: MemBase[_]](
    implicit sourceInfo: SourceInfo,
    compileOptions: CompileOptions,
    contexter: Contexter[V, BaseModule]
  ) =
    new Contextualizer[V] {
      type R = V
      def apply[P](value: V, hierarchy: core.Hierarchy[P]): R = {
        cloneMemToContext(value, contexter(value, hierarchy).context.get)
      }
    }

  implicit def lookuperInstance[M <: BaseModule](
    implicit sourceInfo: SourceInfo,
    compileOptions: CompileOptions,
    proxifier: Proxifier[Instance[M]]
  ) = new Lookuper[Instance[M]] {
    type R = Instance[proxifier.U]
    def apply[P](value: Instance[M], hierarchy: core.Hierarchy[P]) = {
      Instance(proxifier(value, hierarchy))
    }
  }

  implicit def lookuperModule[V <: BaseModule](
    implicit sourceInfo: SourceInfo,
    compileOptions: CompileOptions,
    proxifier: Proxifier[V]
  ) = new Lookuper[V] {
    type R = Instance[proxifier.U]
    def apply[P](value: V, hierarchy: core.Hierarchy[P]) = {
      Instance(proxifier(value, hierarchy))
    }
  }

  implicit def lookuperIsInstantiable[V <: IsInstantiable](
    implicit sourceInfo: SourceInfo,
    compileOptions:      CompileOptions,
    proxifier: Proxifier[V]
  ) = new Lookuper[V] {
    type R = Instance[proxifier.U]
    def apply[P](value: V, hierarchy: core.Hierarchy[P]): R = {
      Instance(proxifier(value, hierarchy))
    }
  }

  // TODO: make Data extend IsContextual, then this can go in Contextual object
  implicit def lookuperData[V <: Data](
    implicit sourceInfo: SourceInfo,
    compileOptions: CompileOptions,
    contextualizer: Contextualizer[V]
  ) = new Lookuper[V] {
    type R = contextualizer.R
    def apply[P](value: V, hierarchy: core.Hierarchy[P]) = {
      contextualizer(value, hierarchy)
    }
  }
  implicit def lenserData[V <: Data](
    implicit sourceInfo: SourceInfo,
    compileOptions: CompileOptions,
    contextualizer: Contextualizer[V]
  ) = new Lenser[V] {
    type R = contextualizer.R
    def apply[P](value: V, lense: core.Lense[P]) = {
      contextualizer(value, new Instance(lense.proxy))
    }
  }

  // TODO: make MemBase extend IsContextual, then this can go in Contextual object
  implicit def lookuperMem[M <: MemBase[_]](
    implicit sourceInfo: SourceInfo,
    compileOptions: CompileOptions,
    contextualizer: Contextualizer[M]
  ) = new Lookuper[M] {
    type R = contextualizer.R
    def apply[P](value: M, hierarchy: core.Hierarchy[P]) = {
      contextualizer(value, hierarchy)
    }
  }
  implicit def lenserMem[V <: MemBase[_]](
    implicit sourceInfo: SourceInfo,
    compileOptions: CompileOptions,
    contextualizer: Contextualizer[V]
  ) = new Lenser[V] {
    type R = contextualizer.R
    def apply[P](value: V, lense: core.Lense[P]) = {
      contextualizer(value, new Instance(lense.proxy))
    }
  }

  //type Definition[P] = core.Definition[P]
  //type Instance[P] = core.Instance[P]
  //type Hierarchy[P] = core.Hierarchy[P]
  //type IsContext = core.IsContext


  def baseModuleContexterBuilder[V](): Contexter[V, BaseModule] = new Contexter[V, BaseModule] {
    def apply[P](value: V, hierarchy: Hierarchy[P]): R = {
      hierarchy.proxy.lookupContext match {
        case Context(Some(value: BaseModule)) => Context(Some(value))
        case _ => Context(None)
      }
    }
  }
  implicit def contexterData[V <: Data] = baseModuleContexterBuilder[V]()
  implicit def contexterMem[V <: MemBase[_]] = baseModuleContexterBuilder[V]()
  implicit def contexterModule[V <: BaseModule] = baseModuleContexterBuilder[V]()
  implicit def contexterInstanceModule[V <: BaseModule] = baseModuleContexterBuilder[Instance[V]]()
  implicit def contexterIsInstantiable[V <: IsInstantiable] = baseModuleContexterBuilder[V]()



  // ========= Extensions =========

  implicit class BaseModuleExtensions[P <: BaseModule](proto: P) {
    import chisel3.experimental.hierarchy.core.{Definition, Instance}
    // Require proto is not a ContextStandIn, as ContextStandIn should always implement toInstance/toDefinition
    require(!proto.isInstanceOf[ContextStandIn[_]], s"Cannot have $proto be a ContextStandIn, must be a proto!!")
    def toInstance:   core.Instance[P] = new core.Instance(proto.asProxy)
    def toDefinition: core.Definition[P] = {
      new core.Definition(StandIn(StandInDefinition(proto, proto.getCircuit)))
    }
    def asProxy: Proxy[P] = {
      Proto(proto, proto._parent.map{
        case p: ContextStandIn[BaseModule] => p.asProxy
        case p: BaseModule => p.asProxy
      })
    }
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
      case Proto(x: BaseModule, _) => x.getTarget
      case StandIn(x: ContextStandIn[_] with BaseModule) => x.getTarget
    }

    /** If this is an instance of a Module, returns the toAbsoluteTarget of this instance
      * @return absoluteTarget of this instance
      */
    def toAbsoluteTarget: IsModule = i.proxy match {
      case Proto(x, _) => x.toAbsoluteTarget
      case StandIn(x: ContextStandIn[_] with BaseModule) => x.toAbsoluteTarget
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
    def toInstance: Instance[T] = Instance(core.Proto(i, None))
    def toDefinition: Definition[T] = Definition(core.Proto(i, None))
  }

}
