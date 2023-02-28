// SPDX-License-Identifier: Apache-2.0

package firrtl
package proto

import java.io.{File, FileInputStream, InputStream}

import collection.JavaConverters._
import FirrtlProtos._
import com.google.protobuf.CodedInputStream
import Firrtl.Statement.{Formal, ReadUnderWrite}
import firrtl.ir.DefModule
import Utils.combine
import java.io.FileNotFoundException
import firrtl.options.OptionsException
import java.nio.file.NotDirectoryException

object FromProto {

  /** Deserialize ProtoBuf representation of [[ir.Circuit]]
    *
    * @param filename Name of file containing ProtoBuf representation
    * @return Deserialized FIRRTL Circuit
    */
  def fromFile(filename: String): ir.Circuit = {
    fromInputStream(new FileInputStream(new File(filename)))
  }

  /** Deserialize ProtoBuf representation of [[ir.Circuit]]
    *
    * @param is InputStream containing ProtoBuf representation
    * @return Deserialized FIRRTL Circuit
    */
  def fromInputStream(is: InputStream): ir.Circuit = {
    val cistream = CodedInputStream.newInstance(is)
    cistream.setRecursionLimit(Integer.MAX_VALUE) // Disable recursion depth check
    val pb = firrtl.FirrtlProtos.Firrtl.parseFrom(cistream)
    proto.FromProto.convert(pb)
  }

  /** Deserialize all the ProtoBuf representations of [[ir.Circuit]] in @dir
    *
    * @param dir directory containing ProtoBuf representation(s)
    * @return Deserialized FIRRTL Circuit
    * @throws java.io.FileNotFoundException if dir does not exist
    * @throws java.nio.file.NotDirectoryException if dir exists but is not a directory
    */
  def fromDirectory(dir: String): ir.Circuit = {
    val d = new File(dir)
    if (!d.exists) {
      throw new FileNotFoundException(s"Specified directory '$d' does not exist!")
    }
    if (!d.isDirectory) {
      throw new NotDirectoryException(s"'$d' is not a directory!")
    }

    val fileList = d.listFiles.filter(_.isFile).toList
    combine(fileList.map(f => fromInputStream(new FileInputStream(f))))
  }

  // Convert from ProtoBuf message repeated Statements to FIRRRTL Block
  private def compressStmts(stmts: scala.collection.Seq[ir.Statement]): ir.Statement = stmts match {
    case scala.collection.Seq()     => ir.EmptyStmt
    case scala.collection.Seq(stmt) => stmt
    case multiple                   => ir.Block(multiple.toSeq)
  }

  def convert(info: Firrtl.SourceInfo): ir.Info =
    info.getSourceInfoCase.getNumber match {
      case Firrtl.SourceInfo.POSITION_FIELD_NUMBER =>
        val pos = info.getPosition
        val str = s"${pos.getFilename} ${pos.getLine}:${pos.getColumn}"
        ir.FileInfo.fromUnescaped(str)
      case Firrtl.SourceInfo.TEXT_FIELD_NUMBER =>
        ir.FileInfo.fromUnescaped(info.getText)
      // NONE_FIELD_NUMBER or anything else
      case _ => ir.NoInfo
    }

  val convert: Map[Firrtl.Expression.PrimOp.Op, ir.PrimOp] =
    ToProto.convert.map { case (k, v) => v -> k }

  def convert(literal: Firrtl.Expression.IntegerLiteral): BigInt =
    BigInt(literal.getValue)

  def convert(bigint: Firrtl.BigInt): BigInt = BigInt(bigint.getValue.toByteArray)

  def convert(uint: Firrtl.Expression.UIntLiteral): ir.UIntLiteral = {
    val width = if (uint.hasWidth) convert(uint.getWidth) else ir.UnknownWidth
    ir.UIntLiteral(convert(uint.getValue), width)
  }

  def convert(sint: Firrtl.Expression.SIntLiteral): ir.SIntLiteral = {
    val width = if (sint.hasWidth) convert(sint.getWidth) else ir.UnknownWidth
    ir.SIntLiteral(convert(sint.getValue), width)
  }

  def convert(subfield: Firrtl.Expression.SubField): ir.SubField =
    ir.SubField(convert(subfield.getExpression), subfield.getField, ir.UnknownType)

  def convert(index: Firrtl.Expression.SubIndex): ir.SubIndex =
    ir.SubIndex(convert(index.getExpression), convert(index.getIndex).toInt, ir.UnknownType)

  def convert(access: Firrtl.Expression.SubAccess): ir.SubAccess =
    ir.SubAccess(convert(access.getExpression), convert(access.getIndex), ir.UnknownType)

  def convert(primop: Firrtl.Expression.PrimOp): ir.DoPrim = {
    val args = primop.getArgList.asScala.map(convert(_)).toSeq
    val consts = primop.getConstList.asScala.map(convert(_)).toSeq
    ir.DoPrim(convert(primop.getOp), args, consts, ir.UnknownType)
  }

  def convert(mux: Firrtl.Expression.Mux): ir.Mux =
    ir.Mux(convert(mux.getCondition), convert(mux.getTValue), convert(mux.getFValue), ir.UnknownType)

  def convert(validif: Firrtl.Expression.ValidIf): ir.ValidIf =
    ir.ValidIf(convert(validif.getCondition), convert(validif.getValue), ir.UnknownType)

  def convert(expr: Firrtl.Expression): ir.Expression = {
    import Firrtl.Expression._
    expr.getExpressionCase.getNumber match {
      case REFERENCE_FIELD_NUMBER    => ir.Reference(expr.getReference.getId, ir.UnknownType)
      case SUB_FIELD_FIELD_NUMBER    => convert(expr.getSubField)
      case SUB_INDEX_FIELD_NUMBER    => convert(expr.getSubIndex)
      case SUB_ACCESS_FIELD_NUMBER   => convert(expr.getSubAccess)
      case UINT_LITERAL_FIELD_NUMBER => convert(expr.getUintLiteral)
      case SINT_LITERAL_FIELD_NUMBER => convert(expr.getSintLiteral)
      case PRIM_OP_FIELD_NUMBER      => convert(expr.getPrimOp)
      case MUX_FIELD_NUMBER          => convert(expr.getMux)
      case VALID_IF_FIELD_NUMBER     => convert(expr.getValidIf)
    }
  }

  def convert(con: Firrtl.Statement.Connect, info: Firrtl.SourceInfo): ir.Connect =
    ir.Connect(convert(info), convert(con.getLocation), convert(con.getExpression))

  def convert(con: Firrtl.Statement.PartialConnect, info: Firrtl.SourceInfo): ir.PartialConnect =
    ir.PartialConnect(convert(info), convert(con.getLocation), convert(con.getExpression))

  def convert(wire: Firrtl.Statement.Wire, info: Firrtl.SourceInfo): ir.DefWire =
    ir.DefWire(convert(info), wire.getId, convert(wire.getType))

  def convert(reg: Firrtl.Statement.Register, info: Firrtl.SourceInfo): ir.DefRegister =
    ir.DefRegister(
      convert(info),
      reg.getId,
      convert(reg.getType),
      convert(reg.getClock),
      convert(reg.getReset),
      convert(reg.getInit)
    )

  def convert(node: Firrtl.Statement.Node, info: Firrtl.SourceInfo): ir.DefNode =
    ir.DefNode(convert(info), node.getId, convert(node.getExpression))

  def convert(inst: Firrtl.Statement.Instance, info: Firrtl.SourceInfo): ir.DefInstance =
    ir.DefInstance(convert(info), inst.getId, inst.getModuleId)

  def convert(when: Firrtl.Statement.When, info: Firrtl.SourceInfo): ir.Conditionally = {
    val conseq = compressStmts(when.getConsequentList.asScala.map(convert(_)))
    val alt = compressStmts(when.getOtherwiseList.asScala.map(convert(_)))
    ir.Conditionally(convert(info), convert(when.getPredicate), conseq, alt)
  }

  def convert(ruw: ReadUnderWrite): ir.ReadUnderWrite.Value = ruw match {
    case ReadUnderWrite.UNDEFINED => ir.ReadUnderWrite.Undefined
    case ReadUnderWrite.OLD       => ir.ReadUnderWrite.Old
    case ReadUnderWrite.NEW       => ir.ReadUnderWrite.New
    case ReadUnderWrite.UNRECOGNIZED =>
      val msg = s"Unrecognized ReadUnderWrite value '$ruw', perhaps this version of FIRRTL is too old?"
      throw new FirrtlUserException(msg)
  }

  def convert(dt: Firrtl.Statement.CMemory.TypeAndDepth): (ir.Type, BigInt) =
    (convert(dt.getDataType), convert(dt.getDepth))

  def convert(cmem: Firrtl.Statement.CMemory, info: Firrtl.SourceInfo): ir.Statement = {
    import Firrtl.Statement.CMemory._
    val (tpe, depth) = cmem.getTypeCase.getNumber match {
      case VECTOR_TYPE_FIELD_NUMBER =>
        val vtpe = convert(cmem.getVectorType)
        (vtpe.tpe, BigInt(vtpe.size))
      case TYPE_AND_DEPTH_FIELD_NUMBER =>
        convert(cmem.getTypeAndDepth)
    }
    CDefMemory(convert(info), cmem.getId, tpe, depth, cmem.getSyncRead, convert(cmem.getReadUnderWrite))
  }

  import Firrtl.Statement.MemoryPort.Direction._
  def convert(mportdir: Firrtl.Statement.MemoryPort.Direction): MPortDir = mportdir match {
    case MEMORY_PORT_DIRECTION_INFER      => MInfer
    case MEMORY_PORT_DIRECTION_READ       => MRead
    case MEMORY_PORT_DIRECTION_WRITE      => MWrite
    case MEMORY_PORT_DIRECTION_READ_WRITE => MReadWrite
    case MEMORY_PORT_DIRECTION_UNKNOWN    => MInfer
    case UNRECOGNIZED =>
      val msg = s"Unrecognized MemoryPort Direction value '$mportdir', perhaps this version of FIRRTL is too old?"
      throw new FirrtlUserException(msg)
  }

  def convert(port: Firrtl.Statement.MemoryPort, info: Firrtl.SourceInfo): CDefMPort = {
    val exprs = Seq(convert(port.getMemoryIndex), convert(port.getExpression))
    CDefMPort(convert(info), port.getId, ir.UnknownType, port.getMemoryId, exprs, convert(port.getDirection))
  }

  def convert(printf: Firrtl.Statement.Printf, info: Firrtl.SourceInfo): ir.Print = {
    val args = printf.getArgList.asScala.map(convert(_)).toSeq
    val str = ir.StringLit(printf.getValue)
    ir.Print(convert(info), str, args, convert(printf.getClk), convert(printf.getEn))
  }

  def convert(stop: Firrtl.Statement.Stop, info: Firrtl.SourceInfo): ir.Stop =
    ir.Stop(convert(info), stop.getReturnValue, convert(stop.getClk), convert(stop.getEn))

  def convert(formal: Formal): ir.Formal.Value = formal match {
    case Formal.ASSERT => ir.Formal.Assert
    case Formal.ASSUME => ir.Formal.Assume
    case Formal.COVER  => ir.Formal.Cover
    case Formal.UNRECOGNIZED =>
      val msg = s"Unrecognized Formal value '$formal', perhaps this version of FIRRTL is too old?"
      throw new FirrtlUserException(msg)
  }

  def convert(ver: Firrtl.Statement.Verification, info: Firrtl.SourceInfo): ir.Verification =
    ir.Verification(
      convert(ver.getOp),
      convert(info),
      convert(ver.getClk),
      convert(ver.getCond),
      convert(ver.getEn),
      ir.StringLit(ver.getMsg)
    )

  def convert(mem: Firrtl.Statement.Memory, info: Firrtl.SourceInfo): ir.DefMemory = {
    val dtype = convert(mem.getType)
    val rs = mem.getReaderIdList.asScala.toSeq
    val ws = mem.getWriterIdList.asScala.toSeq
    val rws = mem.getReadwriterIdList.asScala.toSeq
    import Firrtl.Statement.Memory._
    val depth = mem.getDepthCase.getNumber match {
      case UINT_DEPTH_FIELD_NUMBER   => BigInt(mem.getUintDepth)
      case BIGINT_DEPTH_FIELD_NUMBER => convert(mem.getBigintDepth)
    }
    ir.DefMemory(
      convert(info),
      mem.getId,
      dtype,
      depth,
      mem.getWriteLatency,
      mem.getReadLatency,
      rs,
      ws,
      rws,
      convert(mem.getReadUnderWrite)
    )
  }

  def convert(attach: Firrtl.Statement.Attach, info: Firrtl.SourceInfo): ir.Attach = {
    val exprs = attach.getExpressionList.asScala.map(convert(_)).toSeq
    ir.Attach(convert(info), exprs)
  }

  def convert(stmt: Firrtl.Statement): ir.Statement = {
    import Firrtl.Statement._
    val info = stmt.getSourceInfo
    stmt.getStatementCase.getNumber match {
      case NODE_FIELD_NUMBER            => convert(stmt.getNode, info)
      case CONNECT_FIELD_NUMBER         => convert(stmt.getConnect, info)
      case PARTIAL_CONNECT_FIELD_NUMBER => convert(stmt.getPartialConnect, info)
      case WIRE_FIELD_NUMBER            => convert(stmt.getWire, info)
      case REGISTER_FIELD_NUMBER        => convert(stmt.getRegister, info)
      case WHEN_FIELD_NUMBER            => convert(stmt.getWhen, info)
      case INSTANCE_FIELD_NUMBER        => convert(stmt.getInstance, info)
      case PRINTF_FIELD_NUMBER          => convert(stmt.getPrintf, info)
      case STOP_FIELD_NUMBER            => convert(stmt.getStop, info)
      case MEMORY_FIELD_NUMBER          => convert(stmt.getMemory, info)
      case IS_INVALID_FIELD_NUMBER =>
        ir.IsInvalid(convert(info), convert(stmt.getIsInvalid.getExpression))
      case CMEMORY_FIELD_NUMBER      => convert(stmt.getCmemory, info)
      case MEMORY_PORT_FIELD_NUMBER  => convert(stmt.getMemoryPort, info)
      case ATTACH_FIELD_NUMBER       => convert(stmt.getAttach, info)
      case VERIFICATION_FIELD_NUMBER => convert(stmt.getVerification, info)
    }
  }

  /** Converts ProtoBuf width to FIRRTL width
    *
    * @note Checks for UnknownWidth must be done on the parent object
    * @param width ProtoBuf width representation
    * @return IntWidth equivalent of the ProtoBuf width, default is 0
    */
  def convert(width: Firrtl.Width): ir.IntWidth = ir.IntWidth(width.getValue)

  def convert(ut: Firrtl.Type.UIntType): ir.UIntType = {
    val w = if (ut.hasWidth) convert(ut.getWidth) else ir.UnknownWidth
    ir.UIntType(w)
  }

  def convert(st: Firrtl.Type.SIntType): ir.SIntType = {
    val w = if (st.hasWidth) convert(st.getWidth) else ir.UnknownWidth
    ir.SIntType(w)
  }

  def convert(analog: Firrtl.Type.AnalogType): ir.AnalogType = {
    val w = if (analog.hasWidth) convert(analog.getWidth) else ir.UnknownWidth
    ir.AnalogType(w)
  }

  def convert(field: Firrtl.Type.BundleType.Field): ir.Field = {
    val flip = if (field.getIsFlipped) ir.Flip else ir.Default
    ir.Field(field.getId, flip, convert(field.getType))
  }

  def convert(vtpe: Firrtl.Type.VectorType): ir.VectorType =
    ir.VectorType(convert(vtpe.getType), vtpe.getSize)

  def convert(tpe: Firrtl.Type): ir.Type = {
    import Firrtl.Type._
    tpe.getTypeCase.getNumber match {
      case UINT_TYPE_FIELD_NUMBER        => convert(tpe.getUintType)
      case SINT_TYPE_FIELD_NUMBER        => convert(tpe.getSintType)
      case CLOCK_TYPE_FIELD_NUMBER       => ir.ClockType
      case ASYNC_RESET_TYPE_FIELD_NUMBER => ir.AsyncResetType
      case RESET_TYPE_FIELD_NUMBER       => ir.ResetType
      case ANALOG_TYPE_FIELD_NUMBER      => convert(tpe.getAnalogType)
      case BUNDLE_TYPE_FIELD_NUMBER =>
        ir.BundleType(tpe.getBundleType.getFieldList.asScala.map(convert(_)).toSeq)
      case VECTOR_TYPE_FIELD_NUMBER => convert(tpe.getVectorType)
    }
  }

  def convert(dir: Firrtl.Port.Direction): ir.Direction = {
    import Firrtl.Port.Direction._
    dir match {
      case PORT_DIRECTION_IN  => ir.Input
      case PORT_DIRECTION_OUT => ir.Output
      case (PORT_DIRECTION_UNKNOWN | UNRECOGNIZED) =>
        val msg = s"Unrecognized Port Direction value '$dir', perhaps this version of FIRRTL is too old?"
        throw new FirrtlUserException(msg)
    }
  }

  def convert(port: Firrtl.Port): ir.Port = {
    val dir = convert(port.getDirection)
    val tpe = convert(port.getType)
    ir.Port(ir.NoInfo, port.getId, dir, tpe)
  }

  def convert(param: Firrtl.Module.ExternalModule.Parameter): ir.Param = {
    import Firrtl.Module.ExternalModule.Parameter._
    val name = param.getId
    param.getValueCase.getNumber match {
      case INTEGER_FIELD_NUMBER    => ir.IntParam(name, convert(param.getInteger))
      case DOUBLE_FIELD_NUMBER     => ir.DoubleParam(name, param.getDouble)
      case STRING_FIELD_NUMBER     => ir.StringParam(name, ir.StringLit(param.getString))
      case RAW_STRING_FIELD_NUMBER => ir.RawStringParam(name, param.getRawString)
    }
  }

  def convert(module: Firrtl.Module.UserModule): ir.Module = {
    val name = module.getId
    val ports = module.getPortList.asScala.map(convert(_)).toSeq
    val stmts = module.getStatementList.asScala.map(convert(_)).toSeq
    ir.Module(ir.NoInfo, name, ports, ir.Block(stmts))
  }

  def convert(module: Firrtl.Module.ExternalModule): ir.ExtModule = {
    val name = module.getId
    val ports = module.getPortList.asScala.map(convert(_)).toSeq
    val defname = module.getDefinedName
    val params = module.getParameterList.asScala.map(convert(_)).toSeq
    ir.ExtModule(ir.NoInfo, name, ports, defname, params)
  }

  def convert(module: Firrtl.Module): ir.DefModule =
    if (module.hasUserModule) convert(module.getUserModule)
    else {
      require(module.hasExternalModule, "Module must have Module or ExtModule")
      convert(module.getExternalModule)
    }

  def convert(proto: Firrtl): ir.Circuit = {
    require(proto.getCircuitCount == 1, "Only 1 circuit is currently supported")
    val c = proto.getCircuit(0)
    require(c.getTopCount == 1, "Only 1 top is currently supported")
    val modules = c.getModuleList.asScala.map(convert(_)).toSeq
    val top = c.getTop(0).getName
    ir.Circuit(ir.NoInfo, modules, top)
  }
}
