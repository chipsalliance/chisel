// SPDX-License-Identifier: Apache-2.0
package chisel3.debug

import logger.LazyLogging

import chisel3._
import chisel3.internal._
import chisel3.internal.binding._
import chisel3.internal.firrtl.ir._
import chisel3.debug.CtorParamExtractor.dataToTypeName
import chisel3.experimental.{BaseModule, SourceInfo}

import scala.collection.mutable
import upickle.{default => json}

private[chisel3] object DebugIntrinsics {

  def generate(circuit: Circuit): Unit = {
    val emitter = new ComponentDebugEmitter
    circuit.components.foreach(emitter.generate)
  }

  private class ComponentDebugEmitter extends LazyLogging {
    private val emittedEnums = mutable.HashSet.empty[String]
    private val emittedIds = mutable.HashSet.empty[HasId]

    private val ctorExtractor = new CtorParamExtractor

    def generate(component: Component): Unit = component match {
      case ctx @ DefModule(id, _, _, _, ports, block) =>
        processModule(id, ports ++ ctx.secretPorts, block)
      case ctx @ DefClass(id, _, ports, block) =>
        processModule(id, ports ++ ctx.secretPorts, block)
      case _: DefBlackBox        => ()
      case _: DefIntrinsicModule => ()
      case ctx =>
        throw new InternalErrorException(s"generate: unknown Component type: $ctx")
    }

    private def processModule(id: BaseModule, allPorts: Seq[Port], block: Block): Unit = {
      emittedIds.clear()
      createIntrinsic(id, id._getSourceLocator).foreach(block.addSecretCommand)
      allPorts.foreach { p => createIntrinsic(p.id, None, p.sourceInfo).foreach(block.addSecretCommand) }
      processBlock(block)
    }

    // `When` is special-cased so that we read `elseRegion` only via `hasElse`;
    // the getter would otherwise materialise an empty Block on first call.
    private def processBlock(block: Block): Unit = block.getCommands().foreach { c =>
      generate(c).foreach(block.addSecretCommand)
      c match {
        case w: When =>
          processBlock(w.ifRegion)
          if (w.hasElse) processBlock(w.elseRegion)
        case lb: LayerBlock  => processBlock(lb.region)
        case dc: DefContract => processBlock(dc.region)
        case _ => ()
      }
    }

    private def generate(cmd: Command): Seq[Command] = cmd match {
      case e: DefPrim[_] => createIntrinsic(e.id, None, e.sourceInfo)
      case DefWire(si, id)                              => createIntrinsic(id, None, si)
      case DefReg(si, id, _)                            => createIntrinsic(id, None, si)
      case DefRegInit(si, id, _, _, _)                  => createIntrinsic(id, None, si)
      case DefMemory(si, id, t, size)                   => createIntrinsicMem(id, t, size, si)
      case DefSeqMemory(si, id, t, size, _)             => createIntrinsicMem(id, t, size, si)
      case FirrtlMemory(si, id, t, size, _, _, _, _, _) => createIntrinsicMem(id, t, size, si)
      case _                                            => Seq.empty
    }

    private def suppressDebugParams(target: Any): Boolean = target match {
      case _: chisel3.Bits | _: chisel3.Clock | _: chisel3.Reset => true
      case _: chisel3.experimental.Analog => true
      case _: chisel3.Vec[_] | _: chisel3.EnumType => true
      case _: chisel3.debug.SuppressDebugParams => true
      case _ => false
    }

    private def extractParams(target: Any): Seq[ClassParam] =
      if (suppressDebugParams(target)) Nil else ctorExtractor.getCtorParams(target)

    private def paramsAttr(params: Seq[ClassParam]): Seq[(String, Param)] =
      if (params.isEmpty) Nil else Seq("params" -> StringParam(json.write(params)))

    private def createIntrinsicMem(target: HasId, innerType: Data, size: BigInt, si: SourceInfo): Seq[Command] = {
      val typeName = s"${target.getClass.getSimpleName}[${dataToTypeName(innerType)}[$size]]"
      val name = target.getOptionRef
        .map(_.localName)
        .getOrElse(target match {
          case m: MemBase[_] => m.instanceName
          case _ => ""
        })
      if (name.isEmpty) {
        logger.warn(s"createIntrinsicMem: skipping memory with empty name: $target")
        return Seq.empty
      }
      Seq(
        DefIntrinsic(
          si,
          "circt_debug_var",
          Nil,
          Seq("typeName" -> StringParam(typeName), "name" -> StringParam(name)) ++ paramsAttr(extractParams(target))
        )
      )
    }

    private def createIntrinsic(target: Data, parent: Option[String], si: SourceInfo): Seq[Command] = {
      if (!emittedIds.add(target)) return Seq.empty
      val typeName = dataToTypeName(target)
      // `parent` on every subfield carries the *root* variable's FQN (the name of
      // the enclosing `circt_debug_var`), not the immediate enclosing bundle.
      // CIRCT's CirctDebugVarConverter matches leaves to the root by exact equality.
      val childParent: Option[String] = parent.orElse(Some(signalRef(target)))
      val subCmds: Seq[Command] = target match {
        case e:      EnumType => createEnumDefIntrinsic(e, si).toSeq
        case record: Record =>
          record.elements.values.flatMap(createIntrinsic(_, childParent, si)).toSeq
        case vecLike: VecLike[_] =>
          vecLike.toSeq.flatMap(e => createIntrinsic(e.asInstanceOf[Data], childParent, si))
        case _ => Nil
      }
      subCmds ++ createDebugIntrinsic(target, typeName, parent, extractParams(target), si).toSeq
    }

    private case class EnumVariant(name: String, value: String)
    private implicit val enumVariantRW: json.ReadWriter[EnumVariant] = json.macroRW

    private def enumNames(e: EnumType): (String, String) = {
      val fqn = e.factory.enumTypeName.stripSuffix("$")
      (fqn, fqn.split("\\.").last)
    }

    private def createEnumDefIntrinsic(e: EnumType, si: SourceInfo): Option[Command] = {
      val (fqn, simple) = enumNames(e)
      if (!emittedEnums.add(fqn)) return None
      val variants = e.factory.allWithNames.map { case (v, name) => EnumVariant(name, v.litValue.toString) }
      // Pass the enum's actual bit-width so LowerIntrinsics can build
      // variantsMap with an IntegerAttr of matching width (otherwise
      // it falls back to i64 and downstream consumers like EmitUHDI
      // surface `underlyingTypeRef: "uint64"` for what is in fact a
      // tiny enum). getWidth is always known for a Chisel EnumType.
      val widthParam: Seq[(String, Param)] =
        if (e.width.known) Seq("width" -> IntParam(BigInt(e.width.get)))
        else Nil
      Some(
        DefIntrinsic(
          si,
          "circt_debug_enumdef",
          Nil,
          Seq(
            "typeName" -> StringParam(simple),
            "fqn" -> StringParam(fqn),
            "variants" -> StringParam(json.write(variants))
          ) ++ widthParam
        )
      )
    }

    private def createIntrinsic(target: BaseModule, si: SourceInfo): Seq[Command] = Seq(
      DefIntrinsic(
        si,
        "circt_debug_moduleinfo",
        Nil,
        Seq("typeName" -> StringParam(target.desiredName)) ++ paramsAttr(ctorExtractor.getCtorParams(target))
      )
    )

    private def createDebugIntrinsic(
      target:   Data,
      typeName: String,
      parent:   Option[String],
      params:   Seq[ClassParam],
      si:       SourceInfo
    ): Option[Command] = {
      val name = signalRef(target)
      if (name.isEmpty) return None
      // Synthetic `_`-prefixed names don't survive to final FIRRTL.
      if (parent.isEmpty && name.startsWith("_")) return None
      // firrtl.int.generic operands must be passive.
      val ssaOperands: Seq[Arg] =
        if (target.direction.isInstanceOf[ActualDirection.Bidirectional]) Nil else Seq(Node(target))
      val intrinsicName = if (parent.isDefined) "circt_debug_subfield" else "circt_debug_var"
      val enumParam: Seq[(String, Param)] = target match {
        case e: EnumType =>
          val (fqn, simple) = enumNames(e)
          Seq("enumTypeName" -> StringParam(simple), "enumFqn" -> StringParam(fqn))
        case _ => Nil
      }
      val parentParam = parent.map("parent" -> StringParam(_)).toSeq
      Some(
        DefIntrinsic(
          si,
          intrinsicName,
          ssaOperands,
          Seq("typeName" -> StringParam(typeName), "name" -> StringParam(name))
            ++ parentParam ++ enumParam ++ paramsAttr(params)
        )
      )
    }
  }

  private def signalRef(d: Data): String =
    d.getOptionRef.map(_.localName).getOrElse("")
}
