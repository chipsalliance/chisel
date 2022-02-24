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
    *   1. IsHierarchical
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
  implicit def buildable[T <: BaseModule](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) = new Buildable[T] {
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
  
  implicit def stampable[T <: BaseModule](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) = new Stampable[T] {
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
    contexter: Contexter[BaseModule]
  ) = new Proxifier[Instance[M]] {
    type U = M
    def apply[H](value: Instance[M], hierarchy: core.Hierarchy[H]) = {
      cloneModuleToContext(value.proxy, contexter.lookupContext(hierarchy).get)
    }
  }


  implicit def proxifierModule[V <: BaseModule](
    implicit sourceInfo: SourceInfo,
    compileOptions: CompileOptions,
    contexter: Contexter[BaseModule]
  ) = new Proxifier[V] {
      type U = V
      def apply[H](value: V, hierarchy: core.Hierarchy[H]) = contexter.lookupContext(hierarchy) match {
        case None => toUnderlyingAsInstance(value).asInstanceOf[this.R]
        case Some(p) => cloneModuleToContext(toUnderlyingAsInstance(value), p).asInstanceOf[this.R]
      }
    }

  implicit def contextualizerData[V <: Data](
    implicit sourceInfo: SourceInfo,
    compileOptions: CompileOptions,
    contexter: Contexter[BaseModule]
  ) =
    new Contextualizer[V] {
      type R = V
      def apply[H](v: V, hierarchy: core.Hierarchy[H]): R = {
        val ioMap: Option[Map[Data, Data]] = hierarchy.proxy match {
          case StandIn(x: StandInModule[_]) => Some(x.ioMap)
          case Proto(x: BaseModule, _) => Some(x.getChiselPorts.map { case (_, data) => data -> data }.toMap)
          case m =>
            //println(s"NOWHERE! $m")
            None
        }
        if (isView(v)) {
          cloneViewToContext(v, ioMap, contexter.lookupContext(hierarchy))
        } else {
          doLookupData(v, ioMap, contexter.lookupContext(hierarchy).get)
        }
      }
    }

  implicit def contextualizerMem[M <: MemBase[_]](
    implicit sourceInfo: SourceInfo,
    compileOptions: CompileOptions,
    contexter: Contexter[BaseModule]
  ) =
    new Contextualizer[M] {
      type R = M
      def apply[H](v: M, hierarchy: core.Hierarchy[H]): M = {
        cloneMemToContext(v, contexter.lookupContext(hierarchy).get)
      }
    }

  implicit def instancifyInstance[M <: BaseModule](
    implicit sourceInfo: SourceInfo,
    compileOptions: CompileOptions,
    proxifier: Proxifier[Instance[M]]
  ) = new Instancify[Instance[M]] {
    type R = Instance[proxifier.U]
    def apply[H](value: Instance[M], hierarchy: core.Hierarchy[H]) = {
      Instance(proxifier(value, hierarchy))
    }
  }

  implicit def instancifyModule[V <: BaseModule](
    implicit sourceInfo: SourceInfo,
    compileOptions: CompileOptions,
    proxifier: Proxifier[V]
  ) = new Instancify[V] {
    type R = Instance[proxifier.U]
    def apply[H](value: V, hierarchy: core.Hierarchy[H]) = {
      Instance(proxifier(value, hierarchy))
    }
  }

  implicit def instancifyIsInstantiable[V <: IsInstantiable](
    implicit sourceInfo: SourceInfo,
    compileOptions:      CompileOptions,
    proxifier: Proxifier[V]
  ) = new Instancify[V] {
    type R = Instance[proxifier.U]
    def apply[H](value: V, hierarchy: core.Hierarchy[H]): R = {
      Instance(proxifier(value, hierarchy))
    }
  }

  implicit def instancifyData[V <: Data](
    implicit sourceInfo: SourceInfo,
    compileOptions: CompileOptions,
    contextualizer: Contextualizer[V]
  ) = new Instancify[V] {
    type R = contextualizer.R
    def apply[H](value: V, hierarchy: core.Hierarchy[H]) = {
      contextualizer(value, hierarchy)
    }
  }

  implicit def instancifyMem[M <: MemBase[_]](
    implicit sourceInfo: SourceInfo,
    compileOptions: CompileOptions,
    contextualizer: Contextualizer[M]
  ) = new Instancify[M] {
    type R = contextualizer.R
    def apply[H](value: M, hierarchy: core.Hierarchy[H]) = {
      contextualizer(value, hierarchy)
    }
  }

  //type Definition[H] = core.Definition[H]
  //type Instance[H] = core.Instance[H]
  //type Hierarchy[H] = core.Hierarchy[H]
  //type IsHierarchical = core.IsHierarchical


  implicit val contexter = new Contexter[BaseModule] {
    def lookupContext[H](hierarchy: core.Hierarchy[H]): Option[BaseModule] =
      hierarchy.proxy.lookupContext match {
        case Some(value: BaseModule) => Some(value)
        case _ => None
      }
      //Some(hierarchy.proxy)
  }



  // ========= Extensions =========

  implicit class BaseModuleExtensions[T <: BaseModule](value: T) {
    import chisel3.experimental.hierarchy.core.{Definition, Instance}
    value match {
      case _: IsStandIn[_] => chisel3.internal.throwException("BAD!")
      case other =>
    }
    def toInstance:   core.Instance[T] = new core.Instance(Utils.toUnderlyingAsInstance(value).asInstanceOf[Proxy[T]])
    def toDefinition: core.Definition[T] = new core.Definition(Utils.toUnderlyingAsDefinition(value).asInstanceOf[Proxy[T]])
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
      case StandIn(x: IsStandIn[_] with BaseModule) => x.getTarget
    }

    /** If this is an instance of a Module, returns the toAbsoluteTarget of this instance
      * @return absoluteTarget of this instance
      */
    def toAbsoluteTarget: IsModule = i.proxy match {
      case Proto(x, _) => x.toAbsoluteTarget
      case StandIn(x: IsStandIn[_] with BaseModule) => x.toAbsoluteTarget
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
