// SPDX-License-Identifier: Apache-2.0

package chisel3.properties

import firrtl.{ir => fir}
import chisel3.{Data, RawModule, SpecifiedDirection}
import chisel3.experimental.{BaseModule, SourceInfo}
import chisel3.experimental.hierarchy.{Definition, Instance, ModuleClone}
import chisel3.internal.{throwException, Builder}
import chisel3.internal.binding.{ClassBinding, OpBinding}
import chisel3.internal.firrtl.ir.{Arg, Block, Command, Component, DefClass, DefObject, ModuleIO, Port, PropAssign}
import chisel3.internal.firrtl.Converter
import chisel3.layer.Layer

import scala.collection.mutable.ArrayBuffer

/** Represents a user-defined Class, which is a module-like container of properties.
  *
  * A Class has ports like a hardware module, but its ports must be of Property type.
  *
  * Within a Class body, ports may be connected and other Classes may be instantiated. This means classes cannot
  * construct hardware, only graphs of non-hardware Property information.
  */
class Class extends BaseModule {
  private[chisel3] override def generateComponent(): Option[Component] = {
    // Close the Class.
    require(!_closed, "Can't generate Class more than once")

    evaluateAtModuleBodyEnd()
    _closed = true

    // Ports are named in the same way as regular Modules
    namePorts()

    // Now that elaboration is complete for this Module, we can finalize names
    for (id <- getIds) {
      id match {
        case id: ModuleClone[_] => id.setRefAndPortsRef(_namespace) // special handling
        case id: DynamicObject => {
          // Force name of the Object, and set its Property[ClassType] type's ref to the Object.
          // The type's ref can't be set within instantiate, because the Object hasn't been named yet.
          // This also updates the source Class ref to the DynamicObject ref now that it's named.
          id.forceName(default = "_object", _namespace)
          id.getReference.setRef(id.getRef)
          id.setSourceClassRef()
        }
        case id: StaticObject => {
          // Set the StaticObject's ref and Property[ClassType] type's ref to the BaseModule for the Class.
          // These refs can't be set upon instantiation, because the ModuleClone hasn't been named yet.
          id.setRef(id.getInstanceModule.getRef)
          id.getReference.setRef(id.getInstanceModule.getRef)
        }
        case id: Data =>
          if (id.isSynthesizable) {
            id.topBinding match {
              case ClassBinding(_) =>
                id.forceName(default = "_T", _namespace)
              case _ => ()
            }
          }
        case _ => ()
      }
    }

    // Create IR Ports and set the firrtlPorts variable.
    val ports = getModulePortsAndLocators.map { case (port, sourceInfo, _) =>
      Port(port, port.specifiedDirection, Seq.empty, sourceInfo)
    }

    // No more commands.
    _body.close()

    // Save the Component.
    _component = Some(DefClass(this, name, ports, _body))

    // Return the Component.
    _component
  }

  private[chisel3] override def initializeInParent(): Unit = ()

  /** Add a PropAssign command to the Class
    *
    * Most commands are unsupported in Class, so the internal addCommand API explicitly supports certain commands.
    */
  private[chisel3] def addCommand(c: PropAssign): Unit = addCommandImpl(c)

  /** Add a DefObject command to the Class
    *
    * Most commands are unsupported in Class, so the internal addCommand API explicitly supports certain commands.
    */
  private[chisel3] def addCommand(c: DefObject): Unit = addCommandImpl(c)

  /** Internal state and logic to maintain a buffer of commands.
    */
  protected override def hasBody: Boolean = true

  private def addCommandImpl(c: Command): Unit = {
    require(!_closed, "Can't write to Class after close")
    require(Builder.currentBlock == Some(_body), "can't add commands to blocks that aren't the body")
    _body.addCommand(c)
  }
}

/** Represent a Class type for referencing a Class in a Property[ClassType]
  */
case class ClassType private[chisel3] (name: String) { self =>

  /** A tag type representing an instance of this ClassType
    *
    * This can be used to create a Property IOs
    * {{{
    *   val cls = ClassType("foobar")
    *   val io = IO(Property[cls.Type]())
    *
    *   io :#= cls.unsafeGetReferenceType
    * }}}
    */
  sealed trait Type

  object Type {
    implicit val classTypeProvider: ClassTypeProvider[Type] = ClassTypeProvider(name)
    implicit val propertyType: ClassTypePropertyType.Aux[Property[ClassType] with self.Type, Arg] =
      new ClassTypePropertyType[Property[ClassType] with self.Type](classTypeProvider.classType) {
        override def convert(value: Underlying, ctx: Component, info: SourceInfo): fir.Expression =
          Converter.convert(value, ctx, info)
        type Underlying = Arg
        override def convertUnderlying(value: Property[ClassType] with self.Type, info: SourceInfo) = value.ref(info)
      }
  }
  def copy(name: String = this.name) = new ClassType(name)
}

object ClassType {
  private def apply(name:            String): ClassType = new ClassType(name)
  def unsafeGetClassTypeByName(name: String): ClassType = ClassType(name)
}

sealed trait AnyClassType

object AnyClassType {
  implicit val classTypeProvider: ClassTypeProvider[AnyClassType] = ClassTypeProvider(fir.AnyRefPropertyType)
  implicit val propertyType: RecursivePropertyType.Aux[Property[ClassType] with AnyClassType, ClassType, Arg] =
    new RecursivePropertyType[Property[ClassType] with AnyClassType] {
      type Type = ClassType
      override def getPropertyType(): fir.PropertyType = fir.AnyRefPropertyType

      override def convert(value: Underlying, ctx: Component, info: SourceInfo): fir.Expression =
        Converter.convert(value, ctx, info)
      type Underlying = Arg
      override def convertUnderlying(value: Property[ClassType] with AnyClassType, info: SourceInfo) = value.ref(info)
    }
}

object Class {

  /** Helper to create a Property[ClassType] type for a Class of a given name.
    *
    * This is useful when a Property[ClassType] type is needed but the class does not yet exist or is not available.
    *
    * *WARNING*: It is the caller's resonsibility to ensure the Class exists, this is not checked automatically.
    */
  def unsafeGetReferenceType(className: String): Property[ClassType] = {
    val cls = ClassType.unsafeGetClassTypeByName(className)
    Property[cls.Type]()
  }

  /** Helper to create a DynamicObject for a Class of a given name.
    *
    * *WARNING*: It is the caller's resonsibility to ensure the Class exists, this is not checked automatically.
    */
  def unsafeGetDynamicObject(className: String)(implicit sourceInfo: SourceInfo): DynamicObject = {
    // Instantiate the Object.
    val obj = new DynamicObject(ClassType.unsafeGetClassTypeByName(className))

    // Get its Property[ClassType] type.
    val classProp = obj.getReference

    // Get the BaseModule this connect is occuring within, which may be a RawModule or Class.
    val contextMod = Builder.referenceUserContainer

    // Add the DefObject command directly onto the correct BaseModule subclass.
    // Bind the Property[ClassType] type for this Object.
    contextMod match {
      case rm: RawModule => {
        rm.addCommand(DefObject(sourceInfo, obj, obj.className.name))
        classProp.bind(OpBinding(rm, Builder.currentBlock), SpecifiedDirection.Unspecified)
      }
      case cls: Class => {
        cls.addCommand(DefObject(sourceInfo, obj, obj.className.name))
        classProp.bind(ClassBinding(cls), SpecifiedDirection.Unspecified)
      }
      case _ => throwException("Internal Error! Property connection can only occur within RawModule or Class.")
    }

    obj
  }

  implicit class ClassDefinitionOps[T <: Class](definition: Definition[T]) {

    /** Get a Property[ClassType] type from a Definition[Class].
      *
      * This is useful when a Property[ClassType] type is needed for references to instances of the Class.
      *
      * This method is safe, and should be used over unsafeGetReferenceType when possible.
      */
    def getPropertyType: Property[ClassType] = {
      // Get the BaseModule for the Class this is a definition of.
      val baseModule = definition.getInnerDataContext.getOrElse(
        throwException(
          s"Internal Error! Class instance did not have an associated BaseModule for definition ${definition}."
        )
      )

      // Get a Property[ClassType] type from the Class name.
      val classType = ClassType.unsafeGetClassTypeByName(baseModule.name)
      Property[classType.Type]()
    }

    /** Get a ClassType type from a Definition[Class].
      *
      * This is useful when a ClassType type is needed for other Property types.
      *
      * This method is safe, and should be used over unsafeGetClassTypeByName when possible.
      */
    def getClassType: ClassType = {
      // Get the BaseModule for the Class this is a definition of.
      val baseModule = definition.getInnerDataContext.getOrElse(
        throwException(
          s"Internal Error! Class instance did not have an associated BaseModule for definition ${definition}."
        )
      )

      // Get a ClassType from the Class name.
      ClassType.unsafeGetClassTypeByName(baseModule.name)
    }
  }

  implicit class ClassInstanceOps[T <: Class](instance: Instance[T]) {

    /** Get a reference to an Instance[Class] as a Property[ClassType] property type.
      *
      * This method allows Instances of Classes to be safely connected to Property[ClassType] ports, so the references
      * can be passed through the hierarchy.
      */
    def getPropertyReference: Property[ClassType] = {
      // Get the BaseModule from the Instance.
      val baseModule = instance.getInnerDataContext.getOrElse(
        throwException(
          s"Internal Error! Class instance did not have an associated BaseModule for instance ${instance}."
        )
      )

      // Get a StaticObject for bookkeeping.
      val staticObject = new StaticObject(baseModule)
      val ref = staticObject.getReference

      // Bind the source type.
      val contextMod = Builder.referenceUserContainer
      contextMod match {
        case rm: RawModule => {
          ref.bind(OpBinding(rm, Builder.currentBlock), SpecifiedDirection.Unspecified)
        }
        case cls: Class => {
          ref.bind(ClassBinding(cls), SpecifiedDirection.Unspecified)
        }
        case _ => throwException("Internal Error! Property connection can only occur within RawModule or Class.")
      }

      ref
    }
  }
}
