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

  implicit def viewableInstance[B <: BaseModule](
    implicit sourceInfo: SourceInfo,
    compileOptions: CompileOptions
  ) = new Proxifier[Instance[B]] {
    type U = B
    def apply[A](b: Instance[B], context: core.Hierarchy[A])(implicit h: Hierarchicalizer[A]) = {
      cloneModuleToContext(b.proxy, getInnerDataContext(context).get)
    }
  }


  implicit def viewableModule[B <: BaseModule](implicit sourceInfo: SourceInfo,compileOptions: CompileOptions) =
    new Proxifier[B] {
      type U = B
      def apply[A](b: B, context: core.Hierarchy[A])(implicit h: Hierarchicalizer[A]) = getInnerDataContext(context) match {
        case None =>
          //println("NONE")
          toUnderlyingAsInstance(b).asInstanceOf[this.C]
        case Some(p) =>
          //println(s"SOME: $p")
          val ret = cloneModuleToContext(toUnderlyingAsInstance(b), p)
          //println(s"RET: $ret")
          ret.asInstanceOf[this.C]
      }
    }

  implicit def contextualizerData[D <: Data](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) =
    new Contextualizer[D] {
      type C = D
      def apply[A](v: D, context: core.Hierarchy[A])(implicit h: Hierarchicalizer[A]): D = {
        val ioMap: Option[Map[Data, Data]] = context.proxy match {
          case StandIn(x: StandInModule[_]) => Some(x.ioMap)
          case Proto(x: BaseModule, _) => Some(x.getChiselPorts.map { case (_, data) => data -> data }.toMap)
          case m =>
            //println(s"NOWHERE! $m")
            None
        }
        if (isView(v)) {
          cloneViewToContext(v, ioMap, getInnerDataContext(context))
        } else {
          doLookupData(v, ioMap, getInnerDataContext(context).get)
        }
      }
    }

  implicit def contextualizerMem[M <: MemBase[_]](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) =
    new Contextualizer[M] {
      type C = M
      def apply[A](v: M, context: core.Hierarchy[A])(implicit h: Hierarchicalizer[A]): M = {
        cloneMemToContext(v, getInnerDataContext(context).get)
      }
    }

  implicit def instancifyInstance[B <: BaseModule](
    implicit sourceInfo: SourceInfo,
    compileOptions: CompileOptions,
    proxifier: Proxifier[Instance[B]]
  ) = new Instancify[Instance[B]] {
    type C = Instance[proxifier.U]
    def apply[A](b: Instance[B], context: core.Hierarchy[A])(implicit h: Hierarchicalizer[A]) = {
      Instance(proxifier(b, context))
    }
  }

  implicit def instancifyModule[B <: BaseModule](
    implicit sourceInfo: SourceInfo,
    compileOptions: CompileOptions,
    proxifier: Proxifier[B]
  ) = new Instancify[B] {
    type C = Instance[proxifier.U]
    def apply[A](b: B, context: core.Hierarchy[A])(implicit h: Hierarchicalizer[A]) = {
      Instance(proxifier(b, context))
    }
  }

  implicit def instancifyIsInstantiable[B <: IsInstantiable](
    implicit sourceInfo: SourceInfo,
    compileOptions:      CompileOptions,
    proxifier: Proxifier[B]
  ) = new Instancify[B] {
    type C = Instance[proxifier.U]
    def apply[A](b: B, context: core.Hierarchy[A])(implicit h: Hierarchicalizer[A]): C = {
      Instance(proxifier(b, context))
    }
  }

  implicit def instancifyData[D <: Data](
    implicit sourceInfo: SourceInfo,
    compileOptions: CompileOptions,
    contextualizer: Contextualizer[D]
  ) = new Instancify[D] {
    type C = contextualizer.C
    def apply[A](b: D, context: core.Hierarchy[A])(implicit h: Hierarchicalizer[A]) = {
      contextualizer(b, context)
    }
  }

  implicit def instancifyMem[M <: MemBase[_]](
    implicit sourceInfo: SourceInfo,
    compileOptions: CompileOptions,
    contextualizer: Contextualizer[M]
  ) = new Instancify[M] {
    type C = contextualizer.C
    def apply[A](b: M, context: core.Hierarchy[A])(implicit h: Hierarchicalizer[A]) = {
      contextualizer(b, context)
    }
  }

  //type Definition[A] = core.Definition[A]
  //type Instance[A] = core.Instance[A]
  //type Hierarchy[A] = core.Hierarchy[A]
  //type IsHierarchical = core.IsHierarchical


  implicit def Hierarchicalizer[B <: BaseModule] = new Hierarchicalizer[B] {
    def hierarchy(context: core.Hierarchy[B]): Option[Proxy[core.IsHierarchical]] =
      Some(context.proxy)
    def asUnderlying[X <: BaseModule](value: X): Proxy[X] = value match {
      case i: IsStandIn[_] => chisel3.internal.throwException("Bad!")
      case _ => toUnderlyingAsInstance(value).asInstanceOf[Proxy[X]]
    }
  }

  implicit def instantiable(implicit h: Hierarchicalizer[BaseModule]) = new Hierarchicalizer[IsInstantiable] {
    def hierarchy(context: core.Hierarchy[IsInstantiable]): Option[Proxy[core.IsHierarchical]] = context.proxy match {
      case Proto(p: IsInstantiable, parent) => parent
      case StandIn(standIn: IsStandIn[IsInstantiable]) => standIn.parent match {
        case Some(value: IsStandIn[IsHierarchical]) => Some(StandIn(value))
        case Some(other: BaseModule with IsHierarchical) => Some(Proto(other, other._parent.map(h.asUnderlying)))
        case None => None
      }
    }
    def asUnderlying[X <: IsInstantiable](value: X): Proxy[X] = value match {
      case i: IsStandIn[X] => StandIn(i)
      case other: X => Proto(other, None)
    }
  }


  // ========= Extensions =========

  implicit class BaseModuleExtensions[T <: BaseModule](b: T) {
    import chisel3.experimental.hierarchy.core.{Definition, Instance}
    b match {
      case _: IsStandIn[_] => chisel3.internal.throwException("BAD!")
      case other =>
    }
    def toInstance:   core.Instance[T] = new core.Instance(Utils.toUnderlyingAsInstance(b).asInstanceOf[Proxy[T]])
    def toDefinition: core.Definition[T] = new core.Definition(Utils.toUnderlyingAsDefinition(b).asInstanceOf[Proxy[T]])
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
