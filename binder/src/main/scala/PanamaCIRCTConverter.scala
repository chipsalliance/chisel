// SPDX-License-Identifier: Apache-2.0

package chisel3.internal.panama.circt

import java.io.OutputStream
import geny.Writable

import scala.collection.mutable
import scala.math._
import firrtl.{ir => fir}
import firrtl.annotations.NoTargetAnnotation
import chisel3.{Data => ChiselData, _}
import chisel3.properties.PropertyType
import chisel3.experimental._
import chisel3.internal._
import chisel3.internal.firrtl.ir._
import chisel3.internal.firrtl.Converter
import chisel3.assert.{Assert => VerifAssert}
import chisel3.assume.{Assume => VerifAssume}
import chisel3.cover.{Cover => VerifCover}
import chisel3.printf.{Printf => VerifPrintf}
import chisel3.stop.{Stop => VerifStop}
import chisel3.internal.CIRCTConverter

case class Region(region: MlirRegion, blocks: Seq[MlirBlock]) {
  def get(): MlirRegion = region
  def block(i: Int): MlirBlock = blocks(i)
}

case class Op(state: MlirOperationState, op: MlirOperation, regions: Seq[Region], results: Seq[MlirValue]) {
  def region(i: Int): Region = {
    regions(i)
  }
}

case class Ports(
  types:           Seq[MlirType],
  dirs:            Seq[FIRRTLDirection],
  locs:            Seq[MlirLocation],
  names:           Seq[String],
  nameAttrs:       Seq[MlirAttribute],
  typeAttrs:       Seq[MlirAttribute],
  annotationAttrs: Seq[MlirAttribute],
  symAttrs:        Seq[MlirAttribute],
  locAttrs:        Seq[MlirAttribute])

final case class FirTypeLazy(private var tpeOrData: Either[fir.Type, ChiselData]) {
  def get: fir.Type = {
    tpeOrData match {
      case Left(tpe) => tpe
      case Right(data) =>
        val tpe = Converter.extractType(data, null)
        tpeOrData = Left(tpe)
        tpe
    }
  }
}
object FirTypeLazy {
  implicit def apply(tpe:  fir.Type):   FirTypeLazy = FirTypeLazy(Left(tpe))
  implicit def apply(data: ChiselData): FirTypeLazy = FirTypeLazy(Right(data))
}

sealed abstract class Reference
object Reference {
  final case class Null() extends Reference
  final case class Value(value: MlirValue, private val typeLazy: FirTypeLazy) extends Reference {
    def tpe: fir.Type = typeLazy.get
  }
  // We made it for BlackBox port reference, as there will be only `io` port inside BlackBox and it will be stripped
  final case class BlackBoxIO(enclosure: BaseModule) extends Reference
  final case class SubField(index: Int, tpe: fir.Type) extends Reference
  final case class SubIndex(index: Int, tpe: fir.Type) extends Reference
  final case class SubIndexDynamic(index: MlirValue, tpe: fir.Type) extends Reference
}

case class WhenContext(op: Op, parent: MlirBlock, var inAlt: Boolean) {
  def block: MlirBlock = op.region(if (!inAlt) 0 else 1).block(0)
}

class FirContext {
  var opCircuit: Op = null
  var opModules: Seq[(String, Op)] = Seq.empty
  val whenStack = mutable.Stack.empty[WhenContext]

  // TODO: It's a bit dirty, let's refactor it later
  val items = mutable.Map.empty[Long, Seq[MlirValue]]
  val ops = mutable.Map.empty[Long, MlirOperation]
  def newItem(id:    HasId, value: MlirValue) = items += ((id._id, Seq(value)))
  def newItemVec(id: HasId, value: Seq[MlirValue]) = items += ((id._id, value))
  def getItem(id: HasId): Option[MlirValue] = {
    items
      .get(id._id)
      .map(i => {
        assert(i.length == 1, "item is a vector")
        i(0)
      })
  }
  def getItemVec(id: HasId): Option[Seq[MlirValue]] = items.get(id._id)

  def enterNewCircuit(newCircuit: Op): Unit = {
    items.clear()
    opCircuit = newCircuit
  }

  def enterNewModule(name: String, newModule: Op): Unit = {
    items.clear()
    opModules = opModules :+ (name, newModule)
  }

  def enterWhen(whenOp: Op): Unit = whenStack.push(WhenContext(whenOp, currentBlock, false))
  def enterAlt(): Unit = whenStack.top.inAlt = true
  def leaveOtherwise(depth: Int): Unit = (1 to depth).foreach(_ => whenStack.pop)
  def leaveWhen(depth:      Int, hasAlt: Boolean): Unit = if (!hasAlt) (0 to depth).foreach(_ => whenStack.pop)

  def circuitBlock: MlirBlock = opCircuit.region(0).block(0)
  def findModuleBlock(name: String): MlirBlock = opModules.find(_._1 == name).get._2.region(0).block(0)
  def currentModuleName:  String = opModules.last._1
  def currentModuleBlock: MlirBlock = opModules.last._2.region(0).block(0)
  def currentBlock:       MlirBlock = if (whenStack.nonEmpty) whenStack.top.block else currentModuleBlock
  def currentWhen:        Option[WhenContext] = Option.when(whenStack.nonEmpty)(whenStack.top)
  def rootWhen:           Option[WhenContext] = Option.when(whenStack.nonEmpty)(whenStack.last)
}

case class PanamaCIRCTConverterAnnotation(converter: PanamaCIRCTConverter) extends NoTargetAnnotation
class PanamaCIRCTConverter extends CIRCTConverter {
  val circt = new PanamaCIRCT
  val firCtx = new FirContext
  val mlirRootModule = circt.mlirModuleCreateEmpty(circt.unkLoc)

  object util {
    def getWidthOrSentinel(width: fir.Width): BigInt = width match {
      case fir.UnknownWidth => -1
      case fir.IntWidth(v)  => v
    }

    /// If this is an IntType, AnalogType, or sugar type for a single bit (Clock,
    /// Reset, etc) then return the bitwidth.  Return -1 if the is one of these
    /// types but without a specified bitwidth.  Return -2 if this isn't a simple
    /// type.
    def getWidthOrSentinel(tpe: fir.Type): BigInt = {
      tpe match {
        case fir.ClockType | fir.ResetType | fir.AsyncResetType => 1
        case fir.UIntType(width)                                => getWidthOrSentinel(width)
        case fir.SIntType(width)                                => getWidthOrSentinel(width)
        case fir.AnalogType(width)                              => getWidthOrSentinel(width)
        case fir.ProbeType(underlying)                          => getWidthOrSentinel(underlying)
        case fir.RWProbeType(underlying)                        => getWidthOrSentinel(underlying)
        case _: fir.BundleType | _: fir.VectorType => -2
        case unhandled => throw new Exception(s"unhandled: $unhandled")
      }
    }

    def convert(firType: fir.Type): MlirType = {
      firType match {
        case t: fir.UIntType => circt.firrtlTypeGetUInt(getWidthOrSentinel(t.width).toInt)
        case t: fir.SIntType => circt.firrtlTypeGetSInt(getWidthOrSentinel(t.width).toInt)
        case fir.ClockType      => circt.firrtlTypeGetClock()
        case fir.ResetType      => circt.firrtlTypeGetReset()
        case fir.AsyncResetType => circt.firrtlTypeGetAsyncReset()
        case t: fir.AnalogType => circt.firrtlTypeGetAnalog(getWidthOrSentinel(t.width).toInt)
        case t: fir.VectorType => circt.firrtlTypeGetVector(convert(t.tpe), t.size)
        case t: fir.BundleType =>
          circt.firrtlTypeGetBundle(
            t.fields.map(field =>
              new FIRRTLBundleField(
                field.name,
                field.flip match {
                  case fir.Default => false
                  case fir.Flip    => true
                },
                convert(field.tpe)
              )
            )
          )
        case fir.ProbeType(underlying)   => circt.firrtlTypeGetRef(convert(underlying), false)
        case fir.RWProbeType(underlying) => circt.firrtlTypeGetRef(convert(underlying), true)
        case fir.AnyRefPropertyType      => circt.firrtlTypeGetAnyRef()
        case fir.IntegerPropertyType     => circt.firrtlTypeGetInteger()
        case fir.DoublePropertyType      => circt.firrtlTypeGetDouble()
        case fir.StringPropertyType      => circt.firrtlTypeGetString()
        case fir.BooleanPropertyType     => circt.firrtlTypeGetBoolean()
        case fir.PathPropertyType        => circt.firrtlTypeGetPath()
        case t: fir.SequencePropertyType => circt.firrtlTypeGetList(convert(t.tpe))
        case t: fir.ClassPropertyType =>
          circt.firrtlTypeGetClass(
            circt.mlirFlatSymbolRefAttrGet(t.name),
            Seq.empty /* TODO: where is the elements? */
          )
      }
    }

    def convert(sourceInfo: SourceInfo): MlirLocation = {
      sourceInfo match {
        case _: NoSourceInfo => circt.unkLoc
        case SourceLine(filename, line, col) => circt.mlirLocationFileLineColGet(filename, line, col)
      }
    }

    def convert(name: String, parameter: Param): MlirAttribute = {
      val (tpe, value) = parameter match {
        case IntParam(value) =>
          val tpe = circt.mlirIntegerTypeGet(max(bitLength(value), 32))
          (tpe, circt.mlirIntegerAttrGet(tpe, value.toLong))
        case DoubleParam(value) =>
          val tpe = circt.mlirF64TypeGet()
          (tpe, circt.mlirFloatAttrDoubleGet(tpe, value))
        case StringParam(value) =>
          val tpe = circt.mlirNoneTypeGet()
          (tpe, circt.mlirStringAttrGet(value))
      }
      circt.firrtlAttrGetParamDecl(name, tpe, value)
    }

    def convert(ports: Seq[Port], topDir: SpecifiedDirection = SpecifiedDirection.Unspecified): Ports = {
      val irs = ports.map(Converter.convert(_, topDir)) // firrtl.Port -> IR.Port
      val types = irs.foldLeft(Seq.empty[MlirType]) { case (types, port) => types :+ util.convert(port.tpe) }
      val locs = ports.map(port => util.convert(port.sourceInfo))

      Ports(
        types = types,
        dirs = irs.map(_.direction match {
          case fir.Input  => FIRRTLDirection.In
          case fir.Output => FIRRTLDirection.Out
        }),
        locs = locs,
        names = irs.map(_.name),
        nameAttrs = irs.map(port => circt.mlirStringAttrGet(port.name)),
        typeAttrs = types.map(circt.mlirTypeAttrGet(_)),
        annotationAttrs = ports.map(_ => circt.emptyArrayAttr),
        symAttrs = Seq.empty,
        locAttrs = locs.map(circt.mlirLocationGetAttribute(_))
      )
    }

    def moduleBuilderInsertPorts(builder: OpBuilder, ports: Ports): OpBuilder = {
      builder
        .withNamedAttr("portDirections", circt.firrtlAttrGetPortDirs(ports.dirs))
        .withNamedAttr("portNames", circt.mlirArrayAttrGet(ports.nameAttrs))
        .withNamedAttr("portTypes", circt.mlirArrayAttrGet(ports.typeAttrs))
        .withNamedAttr("portAnnotations", circt.mlirArrayAttrGet(ports.annotationAttrs))
        .withNamedAttr("portSyms", circt.mlirArrayAttrGet(ports.symAttrs))
        .withNamedAttr("portLocations", circt.mlirArrayAttrGet(ports.locAttrs))
    }

    def bitLength(n: BigInt): Int = max(n.bitLength, 1)

    def widthShl(lhs: fir.Width, rhs: fir.Width): fir.Width = (lhs, rhs) match {
      case (l: fir.IntWidth, r: fir.IntWidth) => fir.IntWidth(l.width << r.width.toInt)
      case _ => fir.UnknownWidth
    }

    case class OpBuilder(opName: String, parent: MlirBlock, loc: MlirLocation) {
      var regionsBlocks:   Seq[Option[Seq[(Seq[MlirType], Seq[MlirLocation])]]] = Seq.empty
      var attrs:           Seq[MlirNamedAttribute] = Seq.empty
      var operands:        Seq[MlirValue] = Seq.empty
      var results:         Seq[MlirType] = Seq.empty
      var resultInference: Option[Int] = None

      def withRegion(block: Seq[(Seq[MlirType], Seq[MlirLocation])]): OpBuilder = {
        regionsBlocks = regionsBlocks :+ Some(block)
        this
      }
      def withRegionNoBlock(): OpBuilder = {
        regionsBlocks = regionsBlocks :+ None
        this
      }
      def withRegions(blocks: Seq[Seq[(Seq[MlirType], Seq[MlirLocation])]]): OpBuilder = {
        regionsBlocks = regionsBlocks ++ blocks.map(Some(_))
        this
      }

      def withNamedAttr(name: String, attr: MlirAttribute): OpBuilder = {
        attrs = attrs :+ circt.mlirNamedAttributeGet(name, attr)
        this
      }
      def withNamedAttrs(as: Seq[(String, MlirAttribute)]): OpBuilder = {
        as.foreach(a => withNamedAttr(a._1, a._2))
        this
      }

      def withOperand(o: MlirValue): OpBuilder = { operands = operands :+ o; this }
      def withOperands(os: Seq[MlirValue]): OpBuilder = { operands = operands ++ os; this }

      def withResult(r: MlirType): OpBuilder = { results = results :+ r; this }
      def withResults(rs: Seq[MlirType]): OpBuilder = { results = results ++ rs; this }
      def withResultInference(expectedCount: Int): OpBuilder = { resultInference = Some(expectedCount); this }

      private[OpBuilder] def buildImpl(inserter: MlirOperation => Unit): Op = {
        val state = circt.mlirOperationStateGet(opName, loc)

        circt.mlirOperationStateAddAttributes(state, attrs)
        circt.mlirOperationStateAddOperands(state, operands)
        if (resultInference.isEmpty) {
          circt.mlirOperationStateAddResults(state, results)
        } else {
          circt.mlirOperationStateEnableResultTypeInference(state)
        }

        val builtRegions = regionsBlocks.foldLeft(Seq.empty[Region]) {
          case (builtRegions, blocks) => {
            val region = circt.mlirRegionCreate()
            if (blocks.nonEmpty) {
              val builtBlocks = blocks.get.map {
                case (blockArgTypes, blockArgLocs) => {
                  val block = circt.mlirBlockCreate(blockArgTypes, blockArgLocs)
                  circt.mlirRegionAppendOwnedBlock(region, block)
                  block
                }
              }
              builtRegions :+ Region(region, builtBlocks)
            } else {
              builtRegions :+ Region(region, Seq.empty)
            }
          }
        }
        circt.mlirOperationStateAddOwnedRegions(state, builtRegions.map(_.region))

        val op = circt.mlirOperationCreate(state)
        inserter(op)

        val resultVals = (0 until resultInference.getOrElse(results.length)).map(
          circt.mlirOperationGetResult(op, _)
        )

        Op(state, op, builtRegions, resultVals)
      }

      def build(): Op = buildImpl(circt.mlirBlockAppendOwnedOperation(parent, _))
      def buildAfter(ref:  Op): Op = buildImpl(circt.mlirBlockInsertOwnedOperationAfter(parent, ref.op, _))
      def buildBefore(ref: Op): Op = buildImpl(circt.mlirBlockInsertOwnedOperationBefore(parent, ref.op, _))
    }

    def newConstantValue(resultType: fir.Type, valueType: MlirType, value: BigInt, loc: MlirLocation): MlirValue = {
      util
        .OpBuilder("firrtl.constant", firCtx.currentBlock, loc)
        .withNamedAttr("value", circt.mlirIntegerAttrGet(valueType, value.toLong))
        .withResult(util.convert(resultType))
        .build()
        .results(0)
    }

    // Get reference chain for a node
    def valueReferenceChain(id: HasId, srcInfo: SourceInfo): Seq[Reference] = {
      def rec(id: HasId, chain: Seq[Reference]): Seq[Reference] = {
        def referToPort(data: ChiselData, enclosure: BaseModule): Reference = {
          enclosure match {
            case enclosure: BlackBox => Reference.BlackBoxIO(enclosure)
            case enclosure =>
              val index = enclosure.getChiselPorts.indexWhere(_._2 == data)
              assert(index >= 0, s"can't find port '$data' from '$enclosure'")

              val value = if (enclosure.name != firCtx.currentModuleName) {
                // Reference to a port from instance
                firCtx.getItemVec(enclosure).get(index)
              } else {
                // Reference to a port from current module
                circt.mlirBlockGetArgument(firCtx.currentModuleBlock, index)
              }
              Reference.Value(value, data)
          }
        }

        def referToElement(data: ChiselData): Reference = {
          val tpe = Converter.extractType(data, null)

          data.binding.getOrElse(throw new Exception("non-child data")) match {
            case binding: ChildBinding =>
              binding.parent match {
                case vec: Vec[_] =>
                  data.getRef match {
                    case LitIndex(_, index)    => Reference.SubIndex(index, tpe)
                    case Index(_, ILit(index)) => Reference.SubIndex(index.toInt, tpe)
                    case Index(_, dynamicIndex) =>
                      val index = referTo(dynamicIndex, srcInfo)
                      Reference.SubIndexDynamic(index.value, tpe)
                  }
                case record: Record =>
                  val index = record.elements.size - record.elements.values.iterator.indexOf(data) - 1
                  assert(index >= 0, s"can't find field '$data'")
                  Reference.SubField(index, tpe)
              }
            case _ => throw new Exception("non-child data")
          }
        }

        def referToValue(data: ChiselData) = Reference.Value(
          firCtx.getItem(data) match {
            case Some(value) => value
            case None        => throw new Exception(s"data $data not found")
          },
          data
        )

        id match {
          case module: BaseModule => chain
          case data:   ChiselData =>
            data.binding.getOrElse(throw new Exception("unbound data")) match {
              case PortBinding(enclosure)                   => rec(enclosure, chain :+ referToPort(data, enclosure))
              case ChildBinding(parent)                     => rec(parent, chain :+ referToElement(data))
              case SampleElementBinding(parent)             => rec(parent, chain :+ referToElement(data))
              case MemoryPortBinding(enclosure, visibility) => rec(enclosure, chain :+ referToValue(data))
              case WireBinding(enclosure, visibility)       => rec(enclosure, chain :+ referToValue(data))
              case OpBinding(enclosure, visibility)         => rec(enclosure, chain :+ referToValue(data))
              case RegBinding(enclosure, visibility)        => rec(enclosure, chain :+ referToValue(data))
              case SecretPortBinding(enclosure)             => rec(enclosure, chain :+ referToPort(data, enclosure))
              case unhandled                                => throw new Exception(s"unhandled binding $unhandled")
            }
          case mem:  Mem[ChiselData]         => chain :+ referToValue(mem.t)
          case smem: SyncReadMem[ChiselData] => chain :+ referToValue(smem.t)
          case unhandled => throw new Exception(s"unhandled node $unhandled")
        }
      }
      rec(id, Seq()).reverse // Reverse to make it root first
    }

    def referTo(id: HasId, srcInfo: SourceInfo): Reference.Value = {
      val loc = util.convert(srcInfo)
      val indexType = circt.mlirIntegerTypeGet(32)

      val refChain = valueReferenceChain(id, srcInfo)

      // Root value will be the first element in the chain
      // So the initialization value of the `foldLeft` is unnecessary
      refChain.foldLeft(Reference.Null().asInstanceOf[Reference]) {
        case (parent: Reference, ref: Reference) => {
          ref match {
            case ref @ Reference.Value(_, _)   => ref
            case ref @ Reference.BlackBoxIO(_) => ref
            case Reference.SubField(index, tpe) =>
              val value = parent match {
                case Reference.Value(parentValue, parentType) =>
                  val op = if (circt.firrtlTypeIsAOpenBundle(circt.mlirValueGetType(parentValue))) {
                    "firrtl.opensubfield"
                  } else {
                    "firrtl.subfield"
                  }
                  util
                    .OpBuilder(op, firCtx.currentBlock, loc)
                    .withNamedAttr("fieldIndex", circt.mlirIntegerAttrGet(indexType, index))
                    .withOperand(parentValue)
                    .withResult(util.convert(tpe))
                    .build()
                    .results(0)
                case Reference.BlackBoxIO(enclosure) =>
                  // Look up the field under the instance
                  firCtx.getItemVec(enclosure).map(_(index)).get
              }
              Reference.Value(value, tpe)
            case Reference.SubIndex(index, tpe) =>
              val (parentValue, parentType) = parent match {
                case Reference.Value(parentValue, parentType) => (parentValue, parentType)
              }
              Reference.Value(
                util
                  .OpBuilder("firrtl.subindex", firCtx.currentBlock, loc)
                  .withNamedAttr("index", circt.mlirIntegerAttrGet(indexType, index))
                  .withOperand(parentValue)
                  .withResult(util.convert(tpe))
                  .build()
                  .results(0),
                tpe
              )
            case Reference.SubIndexDynamic(index, tpe) =>
              val (parentValue, parentType) = parent match {
                case Reference.Value(parentValue, parentType) => (parentValue, parentType)
              }
              Reference.Value(
                util
                  .OpBuilder("firrtl.subaccess", firCtx.currentBlock, loc)
                  .withOperand( /* input */ parentValue)
                  .withOperand( /* index */ index)
                  .withResult(util.convert(tpe))
                  .build()
                  .results(0),
                tpe
              )
          }
        }
      } match {
        case ref @ Reference.Value(_, _) => ref
      }
    }

    def referTo(
      arg:     Arg,
      srcInfo: SourceInfo,
      parent:  Option[Component] = None
    ): Reference.Value = {
      def referToNewConstant(n: BigInt, w: Width, isSigned: Boolean): Reference.Value = {
        val (firWidth, valWidth) = w match {
          case _: UnknownWidth =>
            val bitLen = util.bitLength(n)
            (fir.IntWidth(bitLen), bitLen)
          case w: KnownWidth => (fir.IntWidth(w.get), w.get)
        }
        val resultType = if (isSigned) fir.SIntType(firWidth) else fir.UIntType(firWidth)
        val valueType =
          if (isSigned) circt.mlirIntegerTypeSignedGet(valWidth) else circt.mlirIntegerTypeUnsignedGet(valWidth)
        Reference.Value(util.newConstantValue(resultType, valueType, n, util.convert(srcInfo)), resultType)
      }

      def referToNewProperty[T, U](propLit: PropertyLit[T, U]): Reference.Value = {
        def rec(tpe: fir.PropertyType, exp: fir.Expression): MlirValue = {
          val (opName, attrs, operands) = exp match {
            case fir.IntegerPropertyLiteral(value) =>
              val attrs = Seq(
                (
                  "value",
                  circt.mlirIntegerAttrGet(circt.mlirIntegerTypeSignedGet(util.bitLength(value) + 1), value.toLong)
                )
              )
              ("integer", attrs, Seq.empty)
            case fir.DoublePropertyLiteral(value) =>
              val attrs = Seq(("value", circt.mlirFloatAttrDoubleGet(circt.mlirF64TypeGet(), value)))
              ("double", attrs, Seq.empty)
            case fir.StringPropertyLiteral(value) =>
              val attrs = Seq(("value", circt.mlirStringAttrGet(value)))
              ("string", attrs, Seq.empty)
            case fir.BooleanPropertyLiteral(value) =>
              val attrs = Seq(("value", circt.mlirBoolAttrGet(value)))
              ("bool", attrs, Seq.empty)
            case fir.PathPropertyLiteral(value) =>
              val attrs = Seq(("target", circt.mlirStringAttrGet(value)))
              ("unresolved_path", attrs, Seq.empty)
            case fir.SequencePropertyValue(elementTpe, values) =>
              ("list.create", Seq.empty, values.map(rec(elementTpe.asInstanceOf[fir.PropertyType], _)))
          }
          util
            .OpBuilder(s"firrtl.$opName", firCtx.currentBlock, util.convert(srcInfo))
            .withNamedAttrs(attrs)
            .withOperands(operands)
            .withResult(util.convert(tpe))
            .build()
            .results(0)
        }
        val tpe = propLit.propertyType.getPropertyType()
        val exp = propLit.propertyType.convert(propLit.lit, parent.get, srcInfo);
        Reference.Value(rec(tpe, exp), tpe)
      }

      def referToNewProbe(expr: Arg, resultType: fir.Type): Option[Reference.Value] = {
        val builder = expr match {
          case ProbeExpr(probe) =>
            util
              .OpBuilder(s"firrtl.ref.send", firCtx.currentBlock, util.convert(srcInfo))
              .withOperand(referTo(probe, srcInfo).value)
          case RWProbeExpr(probe) =>
            circt.mlirOperationSetInherentAttributeByName(
              firCtx.ops.get(probe.asInstanceOf[Node].id._id).get,
              "inner_sym",
              circt.hwInnerSymAttrGet(probe.localName)
            )
            util
              .OpBuilder("firrtl.ref.rwprobe", firCtx.currentBlock, util.convert(srcInfo))
              .withNamedAttr("target", circt.hwInnerRefAttrGet(parent.get.id.name, probe.localName))
          case ProbeRead(probe) =>
            util
              .OpBuilder(s"firrtl.ref.resolve", firCtx.currentBlock, util.convert(srcInfo))
              .withOperand(referTo(probe, srcInfo).value)
          case _ => return None
        }
        val op = builder.withResult(util.convert(resultType)).build()
        Some(Reference.Value(op.results(0), resultType))
      }

      arg match {
        case Node(id) =>
          // Workaround, as the current implementation relies on Binding. We will probably remove the
          // current Binding implementation eventually, and use Expression instead
          id match {
            case data: ChiselData if data.probeInfo.nonEmpty || data.getOptionRef.isDefined =>
              referToNewProbe(Converter.getRef(id, srcInfo), Converter.extractType(data, srcInfo)).getOrElse {
                referTo(id, srcInfo)
              }
            case _ => referTo(id, srcInfo)
          }
        case arg @ ProbeExpr(data) =>
          val retTpe =
            fir.ProbeType(Converter.extractType(data.asInstanceOf[Node].id.asInstanceOf[ChiselData], srcInfo))
          referToNewProbe(arg, retTpe).get
        case arg @ ProbeRead(data) =>
          val retTpe = Converter
            .extractType(data.asInstanceOf[Node].id.asInstanceOf[ChiselData], srcInfo)
            .asInstanceOf[fir.ProbeType]
            .underlying
          referToNewProbe(arg, retTpe).get
        case ULit(value, width) => referToNewConstant(value, width, false)
        case SLit(value, width) => referToNewConstant(value, width, true)
        case propLit: PropertyLit[_, _] => referToNewProperty(propLit)
        case unhandled => throw new Exception(s"unhandled arg type to be reference: $unhandled")
      }
    }

    def newNode(id: HasId, name: String, resultType: fir.Type, input: MlirValue, loc: MlirLocation): Unit = {
      val op = util
        .OpBuilder("firrtl.node", firCtx.currentBlock, loc)
        .withNamedAttr("name", circt.mlirStringAttrGet(name))
        .withNamedAttr("nameKind", circt.firrtlAttrGetNameKind(FIRRTLNameKind.InterestingName))
        .withNamedAttr("annotations", circt.emptyArrayAttr)
        .withOperand(input)
        .withResult(util.convert(resultType))
        // .withResult( /* ref */ )
        .build()
      firCtx.ops += ((id._id, op.op))
      firCtx.newItem(id, op.results(0))
    }

    def emitConnect(dest: Reference.Value, srcVal: Reference.Value, loc: MlirLocation): Unit = {
      val indexType = circt.mlirIntegerTypeGet(32)
      var src = srcVal

      // TODO: Strict connect

      (dest.tpe, src.tpe) match {
        case (fir.BundleType(fields), fir.BundleType(srcFields)) => {
          assert(srcFields.size == fields.size)

          def subField(index: Int, value: Reference.Value): Reference.Value = {
            val opName = if (circt.firrtlTypeIsAOpenBundle(circt.mlirValueGetType(value.value))) {
              "firrtl.opensubfield"
            } else {
              "firrtl.subfield"
            }
            val fieldTpe = value.tpe.asInstanceOf[fir.BundleType].fields(index).tpe
            val op = util
              .OpBuilder(opName, firCtx.currentBlock, loc)
              .withNamedAttr("fieldIndex", circt.mlirIntegerAttrGet(indexType, index))
              .withOperand(value.value)
              .withResult(util.convert(fieldTpe))
              .build()
            Reference.Value(op.results(0), fieldTpe)
          }

          for (index <- 0 until fields.size) {
            var destField = subField(index, dest)
            var srcField = subField(index, src)
            if (fields(index).flip == fir.Flip) {
              emitConnect(srcField, destField, loc)
            } else {
              emitConnect(destField, srcField, loc)
            }
          }
          return
        }
        case (fir.VectorType(tpe, size), fir.VectorType(_, srcSize)) => {
          assert(srcSize == size)

          def subIndex(index: Int, value: Reference.Value): Reference.Value = {
            val fieldTpe = value.tpe.asInstanceOf[fir.VectorType].tpe
            val op = util
              .OpBuilder("firrtl.subindex", firCtx.currentBlock, loc)
              .withNamedAttr("index", circt.mlirIntegerAttrGet(indexType, index))
              .withOperand(value.value)
              .withResult(util.convert(fieldTpe))
              .build()
            Reference.Value(op.results(0), fieldTpe)
          }

          for (index <- 0 until size) {
            val destElement = subIndex(index, dest)
            val srcElement = subIndex(index, src)
            emitConnect(destElement, srcElement, loc)
          }
          return
        }
        case (_, _) => {}
      }

      val destWidth = util.getWidthOrSentinel(dest.tpe)
      val srcWidth = util.getWidthOrSentinel(src.tpe)

      if (!(destWidth < 0 || srcWidth < 0)) {
        if (destWidth < srcWidth) {
          val isSignedDest = dest.tpe.isInstanceOf[fir.SIntType]
          val tmpType = dest.tpe match {
            case t: fir.UIntType => t
            case fir.SIntType(width) => fir.UIntType(width)
          }
          src = Reference.Value(
            util
              .OpBuilder("firrtl.tail", firCtx.currentBlock, loc)
              .withNamedAttrs(
                Seq(("amount", circt.mlirIntegerAttrGet(circt.mlirIntegerTypeGet(32), (srcWidth - destWidth).toLong)))
              )
              .withOperands(Seq(src.value))
              .withResult(util.convert(tmpType))
              .build()
              .results(0),
            tmpType
          )

          if (isSignedDest) {
            src = Reference.Value(
              util
                .OpBuilder("firrtl.asSInt", firCtx.currentBlock, loc)
                .withOperands(Seq(src.value))
                .withResult(util.convert(dest.tpe))
                .build()
                .results(0),
              dest.tpe
            )
          }
        } else if (srcWidth < destWidth) {
          src = Reference.Value(
            util
              .OpBuilder("firrtl.pad", firCtx.currentBlock, loc)
              .withNamedAttrs(Seq(("amount", circt.mlirIntegerAttrGet(circt.mlirIntegerTypeGet(32), destWidth.toLong))))
              .withOperands(Seq(src.value))
              .withResult(util.convert(dest.tpe))
              .build()
              .results(0),
            dest.tpe
          )
        }
      } else {
        // TODO: const cast
      }

      util
        .OpBuilder("firrtl.connect", firCtx.currentBlock, loc)
        .withOperand( /* dest */ dest.value)
        .withOperand( /* src */ src.value)
        .build()
    }
  }

  val mlirStream = new Writable {
    def writeBytesTo(out: OutputStream): Unit = {
      circt.mlirOperationPrint(circt.mlirModuleGetOperation(mlirRootModule), message => out.write(message.getBytes))
      out.flush()
    }
  }

  val firrtlStream = new Writable {
    def writeBytesTo(out: OutputStream): Unit = {
      circt.mlirExportFIRRTL(mlirRootModule, message => out.write(message.getBytes))
      out.flush()
    }
  }

  val verilogStream = new Writable {
    def writeBytesTo(out: OutputStream): Unit = {
      def assertResult(result: MlirLogicalResult): Unit = {
        assert(circt.mlirLogicalResultIsSuccess(result))
      }

      val pm = circt.mlirPassManagerCreate()
      val options = circt.circtFirtoolOptionsCreateDefault()
      assertResult(circt.circtFirtoolPopulatePreprocessTransforms(pm, options))
      assertResult(circt.circtFirtoolPopulateCHIRRTLToLowFIRRTL(pm, options, mlirRootModule, "-"))
      assertResult(circt.circtFirtoolPopulateLowFIRRTLToHW(pm, options))
      assertResult(circt.circtFirtoolPopulateHWToSV(pm, options))
      assertResult(circt.circtFirtoolPopulateExportVerilog(pm, options, message => out.write(message.getBytes)))
      assertResult(circt.mlirPassManagerRunOnOp(pm, circt.mlirModuleGetOperation(mlirRootModule)))
      out.flush()
    }
  }

  def exportSplitVerilog(directory: os.Path): Unit = {
    def assertResult(result: MlirLogicalResult): Unit = {
      assert(circt.mlirLogicalResultIsSuccess(result))
    }

    val pm = circt.mlirPassManagerCreate()
    val options = circt.circtFirtoolOptionsCreateDefault()
    assertResult(circt.circtFirtoolPopulatePreprocessTransforms(pm, options))
    assertResult(circt.circtFirtoolPopulateCHIRRTLToLowFIRRTL(pm, options, mlirRootModule, "-"))
    assertResult(circt.circtFirtoolPopulateLowFIRRTLToHW(pm, options))
    assertResult(circt.circtFirtoolPopulateHWToSV(pm, options))
    assertResult(circt.circtFirtoolPopulateExportSplitVerilog(pm, options, directory.toString))
    assertResult(circt.mlirPassManagerRunOnOp(pm, circt.mlirModuleGetOperation(mlirRootModule)))
  }

  def passManager(): CIRCTPassManager = new PanamaCIRCTPassManager(circt, mlirRootModule)
  def om():          CIRCTOM = new PanamaCIRCTOM(circt, mlirRootModule)

  def visitCircuit(name: String): Unit = {
    val firCircuit = util
      .OpBuilder("firrtl.circuit", circt.mlirModuleGetBody(mlirRootModule), circt.unkLoc)
      .withRegion(Seq((Seq.empty, Seq.empty)))
      .withNamedAttr("name", circt.mlirStringAttrGet(name))
      .withNamedAttr("annotations", circt.emptyArrayAttr)
      .build()

    firCtx.enterNewCircuit(firCircuit)
  }

  def visitDefBlackBox(defBlackBox: DefBlackBox): Unit = {
    val ports = util.convert(defBlackBox.ports ++ defBlackBox.id.secretPorts, defBlackBox.topDir)
    val nameAttr = circt.mlirStringAttrGet(defBlackBox.name)

    val builder = util
      .OpBuilder("firrtl.extmodule", firCtx.circuitBlock, circt.unkLoc)
      .withRegionNoBlock()
      .withNamedAttr("sym_name", nameAttr)
      .withNamedAttr("sym_visibility", circt.mlirStringAttrGet("private"))
      .withNamedAttr("defname", nameAttr)
      .withNamedAttr("parameters", circt.mlirArrayAttrGet(defBlackBox.params.map(p => util.convert(p._1, p._2)).toSeq))
      .withNamedAttr("convention", circt.firrtlAttrGetConvention(FIRRTLConvention.Internal)) // TODO: handle it corretly
      .withNamedAttr("annotations", circt.emptyArrayAttr)
    val firModule = util.moduleBuilderInsertPorts(builder, ports).build()

    firCtx.enterNewModule(defBlackBox.name, firModule)
  }

  def visitDefIntrinsicModule(defIntrinsicModule: DefIntrinsicModule): Unit = {
    val ports = util.convert(defIntrinsicModule.ports ++ defIntrinsicModule.id.secretPorts, defIntrinsicModule.topDir)

    val builder = util
      .OpBuilder("firrtl.intmodule", firCtx.circuitBlock, circt.unkLoc)
      .withRegionNoBlock()
      .withNamedAttr("sym_name", circt.mlirStringAttrGet(defIntrinsicModule.name))
      .withNamedAttr("sym_visibility", circt.mlirStringAttrGet("private"))
      .withNamedAttr("intrinsic", circt.mlirStringAttrGet(defIntrinsicModule.id.intrinsic))
      .withNamedAttr(
        "parameters",
        circt.mlirArrayAttrGet(defIntrinsicModule.params.map(p => util.convert(p._1, p._2)).toSeq)
      )
      .withNamedAttr("annotations", circt.emptyArrayAttr)
    val firModule = util.moduleBuilderInsertPorts(builder, ports).build()

    firCtx.enterNewModule(defIntrinsicModule.name, firModule)
  }

  def visitDefModule(defModule: DefModule): Unit = {
    val ports = util.convert(defModule.ports ++ defModule.id.secretPorts)

    val isMainModule = defModule.id.circuitName == defModule.name

    val builder = util
      .OpBuilder("firrtl.module", firCtx.circuitBlock, circt.unkLoc)
      .withRegion(Seq((ports.types, ports.locs)))
      .withNamedAttr("sym_name", circt.mlirStringAttrGet(defModule.name))
      .withNamedAttr("sym_visibility", circt.mlirStringAttrGet(if (isMainModule) "public" else "private"))
      .withNamedAttr("convention", circt.firrtlAttrGetConvention(FIRRTLConvention.Internal)) // TODO: handle it corretly
      .withNamedAttr("annotations", circt.emptyArrayAttr)
    val firModule = util.moduleBuilderInsertPorts(builder, ports).build()

    firCtx.enterNewModule(defModule.name, firModule)
  }

  def visitAltBegin(altBegin: AltBegin): Unit = {
    firCtx.enterAlt()
  }

  def visitAttach(attach: Attach): Unit = {
    util
      .OpBuilder("firrtl.attach", firCtx.currentBlock, util.convert(attach.sourceInfo))
      .withOperands(attach.locs.map(node => util.referTo(node.id, attach.sourceInfo).value))
      .build()
  }

  def visitConnect(connect: Connect): Unit = {
    val dest = util.referTo(connect.loc.id, connect.sourceInfo)
    var src = util.referTo(connect.exp, connect.sourceInfo)
    util.emitConnect(dest, src, util.convert(connect.sourceInfo))
  }

  def visitDefWire(defWire: DefWire): Unit = {
    val wireName = Converter.getRef(defWire.id, defWire.sourceInfo).name
    val op = util
      .OpBuilder("firrtl.wire", firCtx.currentBlock, util.convert(defWire.sourceInfo))
      .withNamedAttr("name", circt.mlirStringAttrGet(wireName))
      .withNamedAttr("nameKind", circt.firrtlAttrGetNameKind(FIRRTLNameKind.InterestingName))
      .withNamedAttr("annotations", circt.emptyArrayAttr)
      .withResult(util.convert(Converter.extractType(defWire.id, defWire.sourceInfo)))
      // .withResult( /* ref */ )
      .build()
    firCtx.ops += ((defWire.id._id, op.op))
    firCtx.newItem(defWire.id, op.results(0))
  }

  def visitDefInvalid(defInvalid: DefInvalid): Unit = {
    val loc = util.convert(defInvalid.sourceInfo)
    val dest = util.referTo(defInvalid.arg, defInvalid.sourceInfo)

    val invalidValue = util
      .OpBuilder("firrtl.invalidvalue", firCtx.currentBlock, loc)
      .withResult(util.convert(dest.tpe))
      .build()
      .results(0)

    util
      .OpBuilder("firrtl.connect", firCtx.currentBlock, loc)
      .withOperand( /* dest */ dest.value)
      .withOperand( /* src */ invalidValue)
      .build()
  }

  def visitOtherwiseEnd(otherwiseEnd: OtherwiseEnd): Unit = {
    firCtx.leaveOtherwise(otherwiseEnd.firrtlDepth)
  }

  def visitWhenBegin(whenBegin: WhenBegin): Unit = {
    val loc = util.convert(whenBegin.sourceInfo)
    val cond = util.referTo(whenBegin.pred, whenBegin.sourceInfo)

    val op = util
      .OpBuilder("firrtl.when", firCtx.currentBlock, loc)
      .withRegion( /* then */ Seq((Seq.empty, Seq.empty)))
      .withRegion( /* else */ Seq((Seq.empty, Seq.empty)))
      .withOperand( /* condition */ cond.value)
      .build()

    firCtx.enterWhen(op)
  }

  def visitWhenEnd(whenEnd: WhenEnd): Unit = {
    firCtx.leaveWhen(whenEnd.firrtlDepth, whenEnd.hasAlt)
  }

  def visitDefInstance(defInstance: DefInstance): Unit = {
    val loc = util.convert(defInstance.sourceInfo)
    val ports = util.convert(defInstance.ports ++ defInstance.id.secretPorts)
    val moduleName = defInstance.id.name

    val op = util
      .OpBuilder("firrtl.instance", firCtx.currentBlock, loc)
      .withNamedAttr("moduleName", circt.mlirFlatSymbolRefAttrGet(moduleName))
      .withNamedAttr("name", circt.mlirStringAttrGet(defInstance.name))
      .withNamedAttr("nameKind", circt.firrtlAttrGetNameKind(FIRRTLNameKind.InterestingName))
      .withNamedAttr("portDirections", circt.firrtlAttrGetPortDirs(ports.dirs))
      .withNamedAttr("portNames", circt.mlirArrayAttrGet(ports.nameAttrs))
      .withNamedAttr("portAnnotations", circt.mlirArrayAttrGet(ports.annotationAttrs))
      .withNamedAttr("annotations", circt.emptyArrayAttr)
      .withResults(ports.types)
      .build()
    val results = op.results
    firCtx.ops += ((defInstance.id._id, op.op))
    firCtx.newItemVec(defInstance.id, results)
  }

  def visitDefSeqMemory(defSeqMemory: DefSeqMemory): Unit = {
    val name = Converter.getRef(defSeqMemory.id, defSeqMemory.sourceInfo).name

    val op = util
      .OpBuilder("chirrtl.seqmem", firCtx.currentBlock, util.convert(defSeqMemory.sourceInfo))
      .withNamedAttr(
        "ruw",
        circt.firrtlAttrGetRUW(defSeqMemory.readUnderWrite match {
          case fir.ReadUnderWrite.Undefined => firrtlAttrGetRUW.Undefined
          case fir.ReadUnderWrite.Old       => firrtlAttrGetRUW.Old
          case fir.ReadUnderWrite.New       => firrtlAttrGetRUW.New
        })
      )
      .withNamedAttr("name", circt.mlirStringAttrGet(name))
      .withNamedAttr("nameKind", circt.firrtlAttrGetNameKind(FIRRTLNameKind.InterestingName))
      .withNamedAttr("annotations", circt.emptyArrayAttr)
      .withResult(
        circt.chirrtlTypeGetCMemory(
          util.convert(Converter.extractType(defSeqMemory.t, defSeqMemory.sourceInfo)),
          defSeqMemory.size.intValue
        )
      )
      .build()
    firCtx.newItem(defSeqMemory.t, op.results(0))
  }

  def visitDefMemPort[T <: ChiselData](defMemPort: DefMemPort[T]): Unit = {
    val loc = util.convert(defMemPort.sourceInfo)

    val (parent, build) = firCtx.rootWhen match {
      case Some(when) => (when.parent, (opBuilder: util.OpBuilder) => opBuilder.buildBefore(when.op))
      case None       => (firCtx.currentBlock, (opBuilder: util.OpBuilder) => opBuilder.build())
    }

    val name = Converter.getRef(defMemPort.id, defMemPort.sourceInfo).name
    val op = build(
      util
        .OpBuilder("chirrtl.memoryport", parent, loc)
        .withNamedAttr(
          "direction",
          circt.firrtlAttrGetMemDir(defMemPort.dir match {
            case MemPortDirection.READ  => FIRRTLMemDir.Read
            case MemPortDirection.WRITE => FIRRTLMemDir.Write
            case MemPortDirection.RDWR  => FIRRTLMemDir.ReadWrite
            case MemPortDirection.INFER => FIRRTLMemDir.Infer
          })
        )
        .withNamedAttr("name", circt.mlirStringAttrGet(name))
        .withNamedAttr("annotations", circt.emptyArrayAttr)
        .withOperand( /* memory */ util.referTo(defMemPort.source.id, defMemPort.sourceInfo).value)
        .withResult( /* data */ util.convert(Converter.extractType(defMemPort.id, defMemPort.sourceInfo)))
        .withResult( /* port */ circt.chirrtlTypeGetCMemoryPort())
    )

    util
      .OpBuilder("chirrtl.memoryport.access", firCtx.currentBlock, loc)
      .withOperand( /* port */ op.results(1))
      .withOperand( /* index */ util.referTo(defMemPort.index, defMemPort.sourceInfo).value)
      .withOperand( /* clock */ util.referTo(defMemPort.clock, defMemPort.sourceInfo).value)
      .build()

    firCtx.newItem(defMemPort.id, op.results(0))
  }

  def visitDefMemory(defMemory: DefMemory): Unit = {
    val op = util
      .OpBuilder("chirrtl.combmem", firCtx.currentBlock, util.convert(defMemory.sourceInfo))
      .withNamedAttr("name", circt.mlirStringAttrGet(Converter.getRef(defMemory.id, defMemory.sourceInfo).name))
      .withNamedAttr("nameKind", circt.firrtlAttrGetNameKind(FIRRTLNameKind.InterestingName))
      .withNamedAttr("annotations", circt.emptyArrayAttr)
      .withResult(
        circt.chirrtlTypeGetCMemory(
          util.convert(Converter.extractType(defMemory.t, defMemory.sourceInfo)),
          defMemory.size.intValue
        )
      )
      .build()
    firCtx.newItem(defMemory.t, op.results(0))
  }

  def visitDefPrim[T <: ChiselData](defPrim: DefPrim[T]): Unit = {
    def arg(index: Int): Reference.Value = {
      util.referTo(defPrim.args(index), defPrim.sourceInfo)
    }

    def litArg(index: Int): BigInt = {
      defPrim.args(index) match {
        case ILit(value)    => value
        case ULit(value, _) => value
        case unhandled      => throw new Exception(s"unhandled lit arg type to be extracted: $unhandled")
      }
    }

    val name = Converter.getRef(defPrim.id, defPrim.sourceInfo).name

    val (attrs, operands, resultType) = defPrim.op match {
      // Operands
      //   lhs: sint or uint
      //   rhs: sint or uint
      // Results
      //   result: sint or uint : <max(lhs, rhs) + 1>
      case PrimOp.AddOp | PrimOp.SubOp =>
        val (lhs, rhs) = (arg(0), arg(1))
        val retType = (lhs.tpe, rhs.tpe) match {
          case (fir.SIntType(lhsWidth), fir.SIntType(rhsWidth)) =>
            fir.SIntType(lhsWidth.max(rhsWidth) + fir.IntWidth(1))
          case (fir.UIntType(lhsWidth), fir.UIntType(rhsWidth)) =>
            fir.UIntType(lhsWidth.max(rhsWidth) + fir.IntWidth(1))
        }
        (Seq.empty, Seq(lhs, rhs), retType)

      // Attributes
      //   amount: 32-bit signless integer
      // Operands
      //   input: sint or uint
      // Results
      //   result: uint : <input - amount>
      case PrimOp.TailOp =>
        val (input, amount) = (arg(0), litArg(1))
        val width = input.tpe match {
          case fir.SIntType(inputWidth) => inputWidth - fir.IntWidth(amount)
          case fir.UIntType(inputWidth) => inputWidth - fir.IntWidth(amount)
        }
        val attrs = Seq(("amount", circt.mlirIntegerAttrGet(circt.mlirIntegerTypeGet(32), amount.toLong)))
        (attrs, Seq(input), fir.UIntType(width))

      // Attributes
      //   amount: 32-bit signless integer
      // Operands
      //   input: sint or uint
      // Results
      //   result: uint : <amount>
      case PrimOp.HeadOp =>
        val (input, amount) = (arg(0), litArg(1))
        val width = input.tpe match {
          case fir.SIntType(_) => amount
          case fir.UIntType(_) => amount
        }
        val attrs = Seq(("amount", circt.mlirIntegerAttrGet(circt.mlirIntegerTypeGet(32), amount.toLong)))
        (attrs, Seq(input), fir.UIntType(fir.IntWidth(width)))

      // Operands
      //   lhs: sint or uint
      //   rhs: sint or uint
      // Results
      //   result: sint or uint : <lhs + rhs>
      case PrimOp.TimesOp =>
        val (lhs, rhs) = (arg(0), arg(1))
        val retType = (lhs.tpe, rhs.tpe) match {
          case (fir.SIntType(lhsWidth), fir.SIntType(rhsWidth)) => fir.SIntType(lhsWidth + rhsWidth)
          case (fir.UIntType(lhsWidth), fir.UIntType(rhsWidth)) => fir.UIntType(lhsWidth + rhsWidth)
        }
        (Seq.empty, Seq(lhs, rhs), retType)

      // Operands
      //   lhs: sint or uint
      //   rhs: sint or uint
      // Results
      //   result: sint or uint : <if {uint} then {lhs} else {lhs + 1}>
      case PrimOp.DivideOp =>
        val (lhs, rhs) = (arg(0), arg(1))
        val retType = (lhs.tpe, rhs.tpe) match {
          case (fir.SIntType(lhsWidth), fir.SIntType(rhsWidth)) => fir.SIntType(lhsWidth + fir.IntWidth(1))
          case (fir.UIntType(lhsWidth), fir.UIntType(rhsWidth)) => fir.UIntType(lhsWidth)
        }
        (Seq.empty, Seq(lhs, rhs), retType)

      // Operands
      //   lhs: sint or uint
      //   rhs: sint or uint
      // Results
      //   result: sint or uint : <min(lhs, rhs)>
      case PrimOp.RemOp =>
        val (lhs, rhs) = (arg(0), arg(1))
        val retType = (lhs.tpe, rhs.tpe) match {
          case (fir.SIntType(lhsWidth), fir.SIntType(rhsWidth)) => fir.SIntType(lhsWidth.min(rhsWidth))
          case (fir.UIntType(lhsWidth), fir.UIntType(rhsWidth)) => fir.UIntType(lhsWidth.min(rhsWidth))
        }
        (Seq.empty, Seq(lhs, rhs), retType)

      // Attributes
      //   amount: 32-bit signless integer
      // Operands
      //   input: sint or uint
      // Results
      //   result: sint or uint : <input + amount>
      case PrimOp.ShiftLeftOp =>
        val (input, amount) = (arg(0), litArg(1))
        val (width, retTypeFn) = input.tpe match {
          case fir.SIntType(inputWidth) => (inputWidth + fir.IntWidth(amount), fir.SIntType)
          case fir.UIntType(inputWidth) => (inputWidth + fir.IntWidth(amount), fir.UIntType)
        }
        val attrs = Seq(("amount", circt.mlirIntegerAttrGet(circt.mlirIntegerTypeGet(32), amount.toLong)))
        (attrs, Seq(input), retTypeFn(width))

      // Attributes
      //   amount: 32-bit signless integer
      // Operands
      //   input: sint or uint
      // Results
      //   result: sint or uint : <max(input - amount, 1)>
      case PrimOp.ShiftRightOp =>
        val (input, amount) = (arg(0), litArg(1))
        val (width, retTypeFn) = input.tpe match {
          case fir.SIntType(inputWidth) => ((inputWidth - fir.IntWidth(amount)).max(fir.IntWidth(1)), fir.SIntType)
          case fir.UIntType(inputWidth) => ((inputWidth - fir.IntWidth(amount)).max(fir.IntWidth(1)), fir.UIntType)
        }
        val attrs = Seq(("amount", circt.mlirIntegerAttrGet(circt.mlirIntegerTypeGet(32), amount.toLong)))
        (attrs, Seq(input), retTypeFn(width))

      // Operands
      //   lhs: sint or uint
      //   rhs: uint
      // Results
      //   result: sint or uint : <lhs + 2^rhs - 1>
      case PrimOp.DynamicShiftLeftOp =>
        val (lhs, rhs) = (arg(0), arg(1))
        val retType = (lhs.tpe, rhs.tpe) match {
          case (fir.SIntType(lhsWidth), fir.UIntType(rhsWidth)) =>
            fir.SIntType(lhsWidth + util.widthShl(fir.IntWidth(1), rhsWidth) - fir.IntWidth(1))
          case (fir.UIntType(lhsWidth), fir.UIntType(rhsWidth)) =>
            fir.UIntType(lhsWidth + util.widthShl(fir.IntWidth(1), rhsWidth) - fir.IntWidth(1))
        }
        (Seq.empty, Seq(lhs, rhs), retType)

      // Operands
      //   lhs: sint or uint
      //   rhs: uint
      // Results
      //   result: sint or uint : <lhs>
      case PrimOp.DynamicShiftRightOp =>
        val (lhs, rhs) = (arg(0), arg(1))
        val retType = (lhs.tpe, rhs.tpe) match {
          case (fir.SIntType(lhsWidth), fir.UIntType(rhsWidth)) => fir.SIntType(lhsWidth)
          case (fir.UIntType(lhsWidth), fir.UIntType(rhsWidth)) => fir.UIntType(lhsWidth)
        }
        (Seq.empty, Seq(lhs, rhs), retType)

      // Operands
      //   lhs: sint or uint
      //   rhs: sint or uint
      // Results
      //   result: uint : <max(lhs, rhs)>
      case PrimOp.BitAndOp | PrimOp.BitOrOp | PrimOp.BitXorOp =>
        val (lhs, rhs) = (arg(0), arg(1))
        val width = (lhs.tpe, rhs.tpe) match {
          case (fir.SIntType(lhsWidth), fir.SIntType(rhsWidth)) => lhsWidth.max(rhsWidth)
          case (fir.UIntType(lhsWidth), fir.UIntType(rhsWidth)) => lhsWidth.max(rhsWidth)
        }
        (Seq.empty, Seq(lhs, rhs), fir.UIntType(width))

      // Operands
      //   input: sint or uint
      // Results
      //   result: uint : <input>
      case PrimOp.BitNotOp =>
        val input = arg(0)
        val width = input.tpe match {
          case fir.SIntType(width) => width
          case fir.UIntType(width) => width
        }
        (Seq.empty, Seq(input), fir.UIntType(width))

      // Operands
      //   lhs: sint or uint
      //   rhs: sint or uint
      // Results
      //   result: uint : <lhs + rhs>
      case PrimOp.ConcatOp =>
        val (lhs, rhs) = (arg(0), arg(1))
        val width = (lhs.tpe, rhs.tpe) match {
          case (fir.SIntType(lhsWidth), fir.SIntType(rhsWidth)) => lhsWidth + rhsWidth
          case (fir.UIntType(lhsWidth), fir.UIntType(rhsWidth)) => lhsWidth + rhsWidth
        }
        (Seq.empty, Seq(lhs, rhs), fir.UIntType(width))

      // Attributes
      //   hi: 32-bit signless integer
      //   lo: 32-bit signless integer
      // Operands
      //   input: sint or uint
      // Results
      //   result: uint : <hi - lo + 1>
      case PrimOp.BitsExtractOp =>
        val (input, hi, lo) =
          (arg(0), litArg(1), litArg(2))
        val width = hi - lo + 1
        val intType = circt.mlirIntegerTypeGet(32)
        val attrs = Seq(
          ("hi", circt.mlirIntegerAttrGet(intType, hi.toLong)),
          ("lo", circt.mlirIntegerAttrGet(intType, lo.toLong))
        )
        (attrs, Seq(input), fir.UIntType(fir.IntWidth(width)))

      // Operands
      //   lhs: sint or uint
      //   rhs: sint or uint
      // Results
      //   result: 1-bit uint
      case PrimOp.LessOp | PrimOp.LessEqOp | PrimOp.GreaterOp | PrimOp.GreaterEqOp | PrimOp.EqualOp |
          PrimOp.NotEqualOp =>
        val (lhs, rhs) = (arg(0), arg(1))
        (Seq.empty, Seq(lhs, rhs), fir.UIntType(fir.IntWidth(1)))

      // Attributes
      //   amount: 32-bit signless integer
      // Operands
      //   input: sint or uint
      // Results
      //   result: sint or uint : <max(input, amount)>
      case PrimOp.PadOp =>
        val (input, amount) = (arg(0), litArg(1))
        val (width, retTypeFn) = input.tpe match {
          case fir.SIntType(fir.IntWidth(inputWidth)) => (inputWidth.max(amount), fir.SIntType)
          case fir.UIntType(fir.IntWidth(inputWidth)) => (inputWidth.max(amount), fir.UIntType)
        }
        val attrs = Seq(("amount", circt.mlirIntegerAttrGet(circt.mlirIntegerTypeGet(32), amount.toLong)))
        (attrs, Seq(input), retTypeFn(fir.IntWidth(width)))

      // Operands
      //   input: sint or uint
      // Results
      //   result: sint : <input + 1>
      case PrimOp.NegOp =>
        val input = arg(0)
        val width = input.tpe match {
          case fir.SIntType(inputWidth) => inputWidth + fir.IntWidth(1)
          case fir.UIntType(inputWidth) => inputWidth + fir.IntWidth(1)
        }
        (Seq.empty, Seq(input), fir.SIntType(width))

      // Operands
      //   sel: 1-bit uint or uint with uninferred width
      //   high: a passive base type (contain no flips)
      //   low: a passive base type (contain no flips)
      // Results
      //   result: a passive base type (contain no flips)
      case PrimOp.MultiplexOp =>
        val (sel, high, low) = (arg(0), arg(1), arg(2))
        (Seq.empty, Seq(sel, high, low), Converter.extractType(defPrim.id, defPrim.sourceInfo))

      // Operands
      //   input: sint or uint
      // Results
      //   result: 1-bit uint
      case PrimOp.AndReduceOp | PrimOp.OrReduceOp | PrimOp.XorReduceOp =>
        val input = arg(0)
        (Seq.empty, Seq(input), fir.UIntType(fir.IntWidth(1)))

      // Operands
      //   input: sint or uint
      // Results
      //   result: sint <if {uint} then {input + 1} else {input}>
      case PrimOp.ConvertOp =>
        val input = arg(0)
        val width = input.tpe match {
          case fir.SIntType(inputWidth) => inputWidth
          case fir.UIntType(inputWidth) => inputWidth + fir.IntWidth(1)
        }
        (Seq.empty, Seq(input), fir.SIntType(width))

      // Operands
      //   input: base type
      // Results
      //   result: uint(AsUInt) sint(AsSInt) : <if {sint or uint} then {input} else {1}>
      case PrimOp.AsUIntOp | PrimOp.AsSIntOp =>
        val input = arg(0)
        val width = input.tpe match {
          case fir.SIntType(inputWidth)                           => inputWidth
          case fir.UIntType(inputWidth)                           => inputWidth
          case fir.ClockType | fir.ResetType | fir.AsyncResetType => fir.IntWidth(1)
        }
        val retTypeFn = defPrim.op match {
          case PrimOp.AsUIntOp => fir.UIntType
          case PrimOp.AsSIntOp => fir.SIntType
        }
        (Seq.empty, Seq(input), retTypeFn(width))

      case PrimOp.AsFixedPointOp | PrimOp.AsIntervalOp | PrimOp.WrapOp | PrimOp.SqueezeOp | PrimOp.ClipOp |
          PrimOp.SetBinaryPoint | PrimOp.IncreasePrecision | PrimOp.DecreasePrecision =>
        throw new Exception(s"deprecated primitive op: $defPrim")

      // Operands
      //   input: 1-bit uint/sint/analog, reset, asyncreset, or clock
      // Results
      //   result: clock
      case PrimOp.AsClockOp =>
        val input = arg(0)
        (Seq.empty, Seq(input), fir.ClockType)

      // Operands
      //   input: 1-bit uint/sint/analog, reset, asyncreset, or clock
      // Results
      //   result: async reset
      case PrimOp.AsAsyncResetOp =>
        val input = arg(0)
        (Seq.empty, Seq(input), fir.AsyncResetType)

      case _ => throw new Exception(s"defPrim: $defPrim")
    }

    val loc = util.convert(defPrim.sourceInfo)
    val op = util
      .OpBuilder(s"firrtl.${defPrim.op.toString}", firCtx.currentBlock, loc)
      .withNamedAttrs(attrs)
      .withOperands(operands.map(_.value))
      .withResult(util.convert(resultType))
      .build()
    util.newNode(defPrim.id, name, resultType, op.results(0), loc)
  }

  def visitDefReg(defReg: DefReg): Unit = {
    val name = Converter.getRef(defReg.id, defReg.sourceInfo).name
    val op = util
      .OpBuilder("firrtl.reg", firCtx.currentBlock, util.convert(defReg.sourceInfo))
      .withNamedAttr("name", circt.mlirStringAttrGet(name))
      .withNamedAttr("nameKind", circt.firrtlAttrGetNameKind(FIRRTLNameKind.InterestingName))
      .withNamedAttr("annotations", circt.emptyArrayAttr)
      .withOperand( /* clockVal */ util.referTo(defReg.clock, defReg.sourceInfo).value)
      .withResult( /* result */ util.convert(Converter.extractType(defReg.id, defReg.sourceInfo)))
      .build()
    firCtx.ops += ((defReg.id._id, op.op))
    firCtx.newItem(defReg.id, op.results(0))
  }

  def visitDefRegInit(defRegInit: DefRegInit): Unit = {
    val name = Converter.getRef(defRegInit.id, defRegInit.sourceInfo).name
    val op = util
      .OpBuilder("firrtl.regreset", firCtx.currentBlock, util.convert(defRegInit.sourceInfo))
      .withNamedAttr("name", circt.mlirStringAttrGet(name))
      .withNamedAttr("nameKind", circt.firrtlAttrGetNameKind(FIRRTLNameKind.InterestingName))
      .withNamedAttr("annotations", circt.emptyArrayAttr)
      .withOperand( /* clockVal */ util.referTo(defRegInit.clock, defRegInit.sourceInfo).value)
      .withOperand( /* reset */ util.referTo(defRegInit.reset, defRegInit.sourceInfo).value)
      .withOperand( /* init */ util.referTo(defRegInit.init, defRegInit.sourceInfo).value)
      .withResult( /* result */ util.convert(Converter.extractType(defRegInit.id, defRegInit.sourceInfo)))
      .build()
    firCtx.ops += ((defRegInit.id._id, op.op))
    firCtx.newItem(defRegInit.id, op.results(0))
  }

  def visitPrintf(parent: Component, printf: Printf): Unit = {
    val loc = util.convert(printf.sourceInfo)
    val (fmt, args) = Converter.unpack(printf.pable, parent)
    util
      .OpBuilder("firrtl.printf", firCtx.currentBlock, loc)
      .withNamedAttr("formatString", circt.mlirStringAttrGet(fmt))
      .withNamedAttr("name", circt.mlirStringAttrGet(Converter.getRef(printf.id, printf.sourceInfo).name))
      .withOperand( /* clock */ util.referTo(printf.clock, printf.sourceInfo).value)
      .withOperand(
        /* cond */ util.newConstantValue(fir.UIntType(fir.IntWidth(1)), circt.mlirIntegerTypeUnsignedGet(1), 1, loc)
      )
      .withOperands( /* substitutions */ args.map(util.referTo(_, printf.sourceInfo).value))
      .build()
  }

  def visitStop(stop: Stop): Unit = {
    val loc = util.convert(stop.sourceInfo)
    util
      .OpBuilder("firrtl.stop", firCtx.currentBlock, loc)
      .withNamedAttr("exitCode", circt.mlirIntegerAttrGet(circt.mlirIntegerTypeGet(32), stop.ret))
      .withNamedAttr("name", circt.mlirStringAttrGet(Converter.getRef(stop.id, stop.sourceInfo).name))
      .withOperand( /* clock */ util.referTo(stop.clock, stop.sourceInfo).value)
      .withOperand(
        /* cond */ util.newConstantValue(fir.UIntType(fir.IntWidth(1)), circt.mlirIntegerTypeUnsignedGet(1), 1, loc)
      )
      .build()
  }

  def visitVerification[T <: VerificationStatement](
    verifi: Verification[T],
    opName: String,
    args:   Seq[Arg]
  ): Unit = {
    val loc = util.convert(verifi.sourceInfo)
    util
      .OpBuilder(opName, firCtx.currentBlock, loc)
      .withNamedAttr("message", circt.mlirStringAttrGet(verifi.message))
      .withNamedAttr("name", circt.mlirStringAttrGet(Converter.getRef(verifi.id, verifi.sourceInfo).name))
      .withOperand( /* clock */ util.referTo(verifi.clock, verifi.sourceInfo).value)
      .withOperand( /* predicate */ util.referTo(verifi.predicate, verifi.sourceInfo).value)
      .withOperand(
        /* enable */ util.newConstantValue(fir.UIntType(fir.IntWidth(1)), circt.mlirIntegerTypeUnsignedGet(1), 1, loc)
      )
      .withOperands( /* substitutions */ args.map(util.referTo(_, verifi.sourceInfo).value))
      .build()
  }

  def visitAssert(assert: Verification[VerifAssert]): Unit = {
    visitVerification(assert, "firrtl.assert", Seq.empty)
  }

  def visitAssume(assume: Verification[VerifAssume]): Unit = {
    // TODO: CIRCT emits `assert` for this, is it expected?
    visitVerification(assume, "firrtl.assume", Seq.empty)
  }

  def visitCover(cover: Verification[VerifCover]): Unit = {
    // TODO: CIRCT emits `assert` for this, is it expected?
    visitVerification(cover, "firrtl.cover", Seq.empty)
  }

  def visitProbeDefine(parent: Component, probeDefine: ProbeDefine): Unit = {
    util
      .OpBuilder("firrtl.ref.define", firCtx.currentBlock, util.convert(probeDefine.sourceInfo))
      .withOperand( /* dest */ util.referTo(probeDefine.sink, probeDefine.sourceInfo, Some(parent)).value)
      .withOperand( /* src */ util.referTo(probeDefine.probe, probeDefine.sourceInfo, Some(parent)).value)
      .build()
  }

  def visitProbeForceInitial(parent: Component, probeForceInitial: ProbeForceInitial): Unit = {
    val loc = util.convert(probeForceInitial.sourceInfo)
    util
      .OpBuilder("firrtl.ref.force_initial", firCtx.currentBlock, loc)
      .withOperand(
        /* predicate */ util
          .newConstantValue(fir.UIntType(fir.IntWidth(1)), circt.mlirIntegerTypeUnsignedGet(1), 1, loc)
      )
      .withOperand( /* dest */ util.referTo(probeForceInitial.probe, probeForceInitial.sourceInfo, Some(parent)).value)
      .withOperand( /* src */ util.referTo(probeForceInitial.value, probeForceInitial.sourceInfo, Some(parent)).value)
      .build()
  }

  def visitProbeReleaseInitial(parent: Component, probeReleaseInitial: ProbeReleaseInitial): Unit = {
    val loc = util.convert(probeReleaseInitial.sourceInfo)
    util
      .OpBuilder("firrtl.ref.release_initial", firCtx.currentBlock, loc)
      .withOperand(
        /* predicate */ util
          .newConstantValue(fir.UIntType(fir.IntWidth(1)), circt.mlirIntegerTypeUnsignedGet(1), 1, loc)
      )
      .withOperand(
        /* dest */ util.referTo(probeReleaseInitial.probe, probeReleaseInitial.sourceInfo, Some(parent)).value
      )
      .build()
  }

  def visitProbeForce(parent: Component, probeForce: ProbeForce): Unit = {
    util
      .OpBuilder("firrtl.ref.force", firCtx.currentBlock, util.convert(probeForce.sourceInfo))
      .withOperand( /* clock */ util.referTo(probeForce.clock, probeForce.sourceInfo, Some(parent)).value)
      .withOperand( /* predicate */ util.referTo(probeForce.cond, probeForce.sourceInfo, Some(parent)).value)
      .withOperand( /* dest */ util.referTo(probeForce.probe, probeForce.sourceInfo, Some(parent)).value)
      .withOperand( /* src */ util.referTo(probeForce.value, probeForce.sourceInfo, Some(parent)).value)
      .build()
  }

  def visitProbeRelease(parent: Component, probeRelease: ProbeRelease): Unit = {
    util
      .OpBuilder("firrtl.ref.release", firCtx.currentBlock, util.convert(probeRelease.sourceInfo))
      .withOperand(
        /* clock */ util.referTo(probeRelease.clock, probeRelease.sourceInfo, Some(parent)).value
      )
      .withOperand(
        /* predicate */ util.referTo(probeRelease.cond, probeRelease.sourceInfo, Some(parent)).value
      )
      .withOperand( /* dest */ util.referTo(probeRelease.probe, probeRelease.sourceInfo, Some(parent)).value)
      .build()
  }

  def visitPropAssign(parent: Component, propAssign: PropAssign): Unit = {
    val dest = util.referTo(propAssign.loc.id, propAssign.sourceInfo)
    var src = util.referTo(
      Converter.getRef(
        propAssign.exp match {
          case Node(id) => id
        },
        propAssign.sourceInfo
      ),
      propAssign.sourceInfo,
      Some(parent)
    )

    util
      .OpBuilder("firrtl.propassign", firCtx.currentBlock, util.convert(propAssign.sourceInfo))
      .withOperand( /* dest */ dest.value)
      .withOperand( /* src */ src.value)
      .build()
  }
}

private[chisel3] object PanamaCIRCTConverter {
  def convert(circuit: Circuit): PanamaCIRCTConverter = {
    implicit val cvt = new PanamaCIRCTConverter
    visitCircuit(circuit)
    cvt
  }

  def visitCircuit(circuit: Circuit)(implicit cvt: CIRCTConverter): Unit = {
    cvt.visitCircuit(circuit.name)
    circuit.components.foreach {
      case defBlackBox:        DefBlackBox        => visitDefBlackBox(defBlackBox)
      case defModule:          DefModule          => visitDefModule(defModule)
      case defIntrinsicModule: DefIntrinsicModule => visitDefIntrinsicModule(defIntrinsicModule)
    }
  }
  def visitDefBlackBox(defBlackBox: DefBlackBox)(implicit cvt: CIRCTConverter): Unit = {
    cvt.visitDefBlackBox(defBlackBox)
  }
  def visitDefModule(defModule: DefModule)(implicit cvt: CIRCTConverter): Unit = {
    cvt.visitDefModule(defModule)
    val commands = defModule.commands ++ defModule.secretCommands
    // Workaround for https://github.com/chipsalliance/chisel/issues/3435, peeking the next command
    commands.zip(commands.map(Some(_)).drop(1) :+ None).foreach {
      case (cmd, nextCmd) =>
        cmd match {
          // Command
          case altBegin: AltBegin => visitAltBegin(altBegin)
          case attach:   Attach   => visitAttach(attach)
          case connect:  Connect  => visitConnect(connect)
          // case partialConnect: PartialConnect => {} // TODO
          case connectInit:  ConnectInit  => visitConnectInit(connectInit)
          case defInvalid:   DefInvalid   => visitDefInvalid(defInvalid)
          case otherwiseEnd: OtherwiseEnd => visitOtherwiseEnd(otherwiseEnd)
          case whenBegin:    WhenBegin    => visitWhenBegin(whenBegin)
          case whenEnd:      WhenEnd      => visitWhenEnd(whenEnd, nextCmd)

          // Definition
          case defInstance:         DefInstance               => visitDefInstance(defInstance)
          case defMemPort:          DefMemPort[ChiselData]    => visitDefMemPort(defMemPort)
          case defMemory:           DefMemory                 => visitDefMemory(defMemory)
          case defPrim:             DefPrim[ChiselData]       => visitDefPrim(defPrim)
          case defReg:              DefReg                    => visitDefReg(defReg)
          case defRegInit:          DefRegInit                => visitDefRegInit(defRegInit)
          case defSeqMemory:        DefSeqMemory              => visitDefSeqMemory(defSeqMemory)
          case defWire:             DefWire                   => visitDefWire(defWire)
          case printf:              Printf                    => visitPrintf(defModule, printf)
          case stop:                Stop                      => visitStop(stop)
          case assert:              Verification[VerifAssert] => visitVerfiAssert(assert)
          case assume:              Verification[VerifAssume] => visitVerfiAssume(assume)
          case cover:               Verification[VerifCover]  => visitVerfiCover(cover)
          case printf:              Verification[VerifPrintf] => visitVerfiPrintf(printf)
          case stop:                Verification[VerifStop]   => visitVerfiStop(stop)
          case probeDefine:         ProbeDefine               => visitProbeDefine(defModule, probeDefine)
          case probeForceInitial:   ProbeForceInitial         => visitProbeForceInitial(defModule, probeForceInitial)
          case probeReleaseInitial: ProbeReleaseInitial       => visitProbeReleaseInitial(defModule, probeReleaseInitial)
          case probeForce:          ProbeForce                => visitProbeForce(defModule, probeForce)
          case probeRelease:        ProbeRelease              => visitProbeRelease(defModule, probeRelease)
          case propAssign:          PropAssign                => visitPropAssign(defModule, propAssign)
          case unhandled => throw new Exception(s"unhandled op: $unhandled")
        }
    }
  }
  def visitDefIntrinsicModule(defIntrinsicModule: DefIntrinsicModule)(implicit cvt: CIRCTConverter): Unit = {
    cvt.visitDefIntrinsicModule(defIntrinsicModule)
  }
  def visitAltBegin(altBegin: AltBegin)(implicit cvt: CIRCTConverter): Unit = {
    cvt.visitAltBegin(altBegin)
  }
  def visitAttach(attach: Attach)(implicit cvt: CIRCTConverter): Unit = {
    cvt.visitAttach(attach)
  }
  def visitConnect(connect: Connect)(implicit cvt: CIRCTConverter): Unit = {
    cvt.visitConnect(connect)
  }
  def visitConnectInit(connectInit: ConnectInit)(implicit cvt: CIRCTConverter): Unit = {
    // Not used anywhere
    throw new Exception("unimplemented")
  }
  def visitDefInvalid(defInvalid: DefInvalid)(implicit cvt: CIRCTConverter): Unit = {
    cvt.visitDefInvalid(defInvalid)
  }
  def visitOtherwiseEnd(otherwiseEnd: OtherwiseEnd)(implicit cvt: CIRCTConverter): Unit = {
    cvt.visitOtherwiseEnd(otherwiseEnd)
  }
  def visitWhenBegin(whenBegin: WhenBegin)(implicit cvt: CIRCTConverter): Unit = {
    cvt.visitWhenBegin(whenBegin)
  }
  def visitWhenEnd(whenEnd: WhenEnd, next: Option[Command])(implicit cvt: CIRCTConverter): Unit = {
    // FIXME: workaround https://github.com/chipsalliance/chisel/issues/3435
    val hasAlt = next match {
      case Some(_: AltBegin) => true
      case _ => false
    }
    val whenEndPatched = WhenEnd(whenEnd.sourceInfo, whenEnd.firrtlDepth, hasAlt)
    cvt.visitWhenEnd(whenEndPatched)
  }
  def visitDefInstance(defInstance: DefInstance)(implicit cvt: CIRCTConverter): Unit = {
    cvt.visitDefInstance(defInstance)
  }
  def visitDefMemPort[T <: ChiselData](defMemPort: DefMemPort[T])(implicit cvt: CIRCTConverter): Unit = {
    cvt.visitDefMemPort(defMemPort)
  }
  def visitDefMemory(defMemory: DefMemory)(implicit cvt: CIRCTConverter): Unit = {
    cvt.visitDefMemory(defMemory)
  }
  def visitDefPrim[T <: ChiselData](defPrim: DefPrim[T])(implicit cvt: CIRCTConverter): Unit = {
    cvt.visitDefPrim(defPrim)
  }
  def visitDefReg(defReg: DefReg)(implicit cvt: CIRCTConverter): Unit = {
    cvt.visitDefReg(defReg)
  }
  def visitDefRegInit(defRegInit: DefRegInit)(implicit cvt: CIRCTConverter): Unit = {
    cvt.visitDefRegInit(defRegInit)
  }
  def visitDefSeqMemory(defSeqMemory: DefSeqMemory)(implicit cvt: CIRCTConverter): Unit = {
    cvt.visitDefSeqMemory(defSeqMemory)
  }
  def visitDefWire(defWire: DefWire)(implicit cvt: CIRCTConverter): Unit = {
    cvt.visitDefWire(defWire)
  }
  def visitPrintf(parent: Component, printf: Printf)(implicit cvt: CIRCTConverter): Unit = {
    cvt.visitPrintf(parent, printf)
  }
  def visitStop(stop: Stop)(implicit cvt: CIRCTConverter): Unit = {
    cvt.visitStop(stop)
  }
  def visitVerfiAssert(assert: Verification[VerifAssert])(implicit cvt: CIRCTConverter): Unit = {
    cvt.visitAssert(assert)
  }
  def visitVerfiAssume(assume: Verification[VerifAssume])(implicit cvt: CIRCTConverter): Unit = {
    cvt.visitAssume(assume)
  }
  def visitVerfiCover(cover: Verification[VerifCover])(implicit cvt: CIRCTConverter): Unit = {
    cvt.visitCover(cover)
  }
  def visitVerfiPrintf(printf: Verification[VerifPrintf])(implicit cvt: CIRCTConverter): Unit = {
    // TODO: Not used anywhere?
    throw new Exception("unimplemented")
  }
  def visitVerfiStop(stop: Verification[VerifStop])(implicit cvt: CIRCTConverter): Unit = {
    // TODO: Not used anywhere?
    throw new Exception("unimplemented")
  }
  def visitProbeDefine(parent: Component, probeDefine: ProbeDefine)(implicit cvt: CIRCTConverter): Unit = {
    cvt.visitProbeDefine(parent, probeDefine)
  }
  def visitProbeForceInitial(
    parent:            Component,
    probeForceInitial: ProbeForceInitial
  )(
    implicit cvt: CIRCTConverter
  ): Unit = {
    cvt.visitProbeForceInitial(parent, probeForceInitial)
  }
  def visitProbeReleaseInitial(
    parent:              Component,
    probeReleaseInitial: ProbeReleaseInitial
  )(
    implicit cvt: CIRCTConverter
  ): Unit = {
    cvt.visitProbeReleaseInitial(parent, probeReleaseInitial)
  }
  def visitProbeForce(
    parent:     Component,
    probeForce: ProbeForce
  )(
    implicit cvt: CIRCTConverter
  ): Unit = {
    cvt.visitProbeForce(parent, probeForce)
  }
  def visitProbeRelease(
    parent:       Component,
    probeRelease: ProbeRelease
  )(
    implicit cvt: CIRCTConverter
  ): Unit = {
    cvt.visitProbeRelease(parent, probeRelease)
  }
  def visitPropAssign(parent: Component, propAssign: PropAssign)(implicit cvt: CIRCTConverter): Unit = {
    cvt.visitPropAssign(parent, propAssign)
  }
}
