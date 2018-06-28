// See LICENSE for license details.

package firrtl
package proto

import java.io.{BufferedOutputStream, OutputStream}

import FirrtlProtos._
import Firrtl.Expression.PrimOp.Op
import com.google.protobuf.{CodedOutputStream, WireFormat}
import firrtl.PrimOps._

import scala.collection.JavaConverters._

object ToProto {


  /** Serialize a FIRRTL Circuit to an Output Stream as a ProtoBuf message
    *
    * @param ostream Output stream that will be written
    * @param circuit The Circuit to serialize
    */
  def writeToStream(ostream: OutputStream, circuit: ir.Circuit): Unit = {
    writeToStreamFast(ostream, circuit.info, circuit.modules.map(() => _), circuit.main)
  }

  /** Serialized a deconstructed Circuit with lazy Modules
    *
    * This serializer allows intermediate objects to be garbage collected during serialization
    * to save time and memory
    *
    * @param ostream Output stream that will be written
    * @param info Info of Circuit
    * @param modules Functions to generate Modules lazily
    * @param main Top-level module of the Circuit
    */
  // Note this function is sensitive to changes to the Firrtl and Circuit protobuf message definitions
  def writeToStreamFast(
    ostream: OutputStream,
    info: ir.Info,
    modules: Seq[() => ir.DefModule],
    main: String
  ): Unit = {
    val costream = CodedOutputStream.newInstance(ostream)

    // Write each module for the circuit
    val ostreamInner = new java.io.ByteArrayOutputStream()
    val costreamInner = CodedOutputStream.newInstance(ostreamInner)
    for (mod <- modules) {
      costreamInner.writeMessage(Firrtl.Circuit.MODULE_FIELD_NUMBER, convert(mod()).build)
    }
    val top = Firrtl.Top.newBuilder().setName(main).build
    costreamInner.writeMessage(Firrtl.Circuit.TOP_FIELD_NUMBER, top)

    // Write Circuit header first
    costream.writeTag(Firrtl.CIRCUIT_FIELD_NUMBER, WireFormat.WIRETYPE_LENGTH_DELIMITED)
    costream.writeUInt32NoTag(costreamInner.getTotalBytesWritten)
    costream.flush()

    // Write Modules
    costreamInner.flush()
    ostreamInner.writeTo(ostream)
    ostreamInner.flush()
  }

  val convert: Map[ir.PrimOp, Op] = Map(
    Add -> Op.OP_ADD,
    Sub -> Op.OP_SUB,
    Mul -> Op.OP_TIMES,
    Div -> Op.OP_DIVIDE,
    Rem -> Op.OP_REM,
    Lt -> Op.OP_LESS,
    Leq -> Op.OP_LESS_EQ,
    Gt -> Op.OP_GREATER,
    Geq -> Op.OP_GREATER_EQ,
    Eq -> Op.OP_EQUAL,
    Neq -> Op.OP_NOT_EQUAL,
    Pad -> Op.OP_PAD,
    AsUInt -> Op.OP_AS_UINT,
    AsSInt -> Op.OP_AS_SINT,
    AsClock -> Op.OP_AS_CLOCK,
    AsFixedPoint -> Op.OP_AS_FIXED_POINT,
    Shl -> Op.OP_SHIFT_LEFT,
    Shr -> Op.OP_SHIFT_RIGHT,
    Dshl -> Op.OP_DYNAMIC_SHIFT_LEFT,
    Dshr -> Op.OP_DYNAMIC_SHIFT_RIGHT,
    Cvt -> Op.OP_CONVERT,
    Neg -> Op.OP_NEG,
    Not -> Op.OP_BIT_NOT,
    And -> Op.OP_BIT_AND,
    Or -> Op.OP_BIT_OR,
    Xor -> Op.OP_BIT_XOR,
    Andr -> Op.OP_AND_REDUCE,
    Orr -> Op.OP_OR_REDUCE,
    Xorr -> Op.OP_XOR_REDUCE,
    Cat -> Op.OP_CONCAT,
    Bits -> Op.OP_EXTRACT_BITS,
    Head -> Op.OP_HEAD,
    Tail -> Op.OP_TAIL,
    BPShl -> Op.OP_SHIFT_BINARY_POINT_LEFT,
    BPShr -> Op.OP_SHIFT_BINARY_POINT_RIGHT,
    BPSet -> Op.OP_SET_BINARY_POINT
  )

  def convertToIntegerLiteral(value: BigInt): Firrtl.Expression.IntegerLiteral.Builder = {
    Firrtl.Expression.IntegerLiteral.newBuilder()
      .setValue(value.toString)
  }

  def convertToBigInt(value: BigInt): Firrtl.BigInt.Builder = {
    Firrtl.BigInt.newBuilder()
      .setValue(com.google.protobuf.ByteString.copyFrom(value.toByteArray))
  }

  def convert(info: ir.Info): Firrtl.SourceInfo.Builder = {
    val ib = Firrtl.SourceInfo.newBuilder()
    info match {
      case ir.NoInfo =>
        ib.setNone(Firrtl.SourceInfo.None.newBuilder)
      case ir.FileInfo(ir.StringLit(text)) =>
        ib.setText(text)
      // TODO properly implement MultiInfo
      case ir.MultiInfo(infos) =>
        val x = if (infos.nonEmpty) infos.head else ir.NoInfo
        convert(x)
    }
  }

  def convert(expr: ir.Expression): Firrtl.Expression.Builder = {
    val eb = Firrtl.Expression.newBuilder()
    expr match {
      case ir.Reference(name, _) =>
        val rb = Firrtl.Expression.Reference.newBuilder()
          .setId(name)
        eb.setReference(rb)
      case ir.SubField(e, name, _) =>
        val sb = Firrtl.Expression.SubField.newBuilder()
          .setExpression(convert(e))
          .setField(name)
        eb.setSubField(sb)
      case ir.SubIndex(e, value, _) =>
        val sb = Firrtl.Expression.SubIndex.newBuilder()
          .setExpression(convert(e))
          .setIndex(convertToIntegerLiteral(value))
        eb.setSubIndex(sb)
      case ir.SubAccess(e, index, _) =>
        val sb = Firrtl.Expression.SubAccess.newBuilder()
          .setExpression(convert(e))
          .setIndex(convert(index))
        eb.setSubAccess(sb)
      case ir.UIntLiteral(value, width) =>
        val ub = Firrtl.Expression.UIntLiteral.newBuilder()
          .setValue(convertToIntegerLiteral(value))
        convert(width).foreach(ub.setWidth)
        eb.setUintLiteral(ub)
      case ir.SIntLiteral(value, width) =>
        val sb = Firrtl.Expression.SIntLiteral.newBuilder()
          .setValue(convertToIntegerLiteral(value))
        convert(width).foreach(sb.setWidth)
        eb.setSintLiteral(sb)
      case ir.FixedLiteral(value, width, point) =>
        val fb = Firrtl.Expression.FixedLiteral.newBuilder()
          .setValue(convertToBigInt(value))
        convert(width).foreach(fb.setWidth)
        convert(point).foreach(fb.setPoint)
        eb.setFixedLiteral(fb)
      case ir.DoPrim(op, args, consts, _) =>
        val db = Firrtl.Expression.PrimOp.newBuilder()
          .setOp(convert(op))
        consts.foreach(c => db.addConst(convertToIntegerLiteral(c)))
        args.foreach(a => db.addArg(convert(a)))
        eb.setPrimOp(db)
      case ir.Mux(cond, tval, fval, _) =>
        val mb = Firrtl.Expression.Mux.newBuilder()
          .setCondition(convert(cond))
          .setTValue(convert(tval))
          .setFValue(convert(fval))
        eb.setMux(mb)
    }
  }

  def convert(dir: MPortDir): Firrtl.Statement.MemoryPort.Direction = {
    import Firrtl.Statement.MemoryPort.Direction._
    dir match {
      case MInfer => MEMORY_PORT_DIRECTION_INFER
      case MRead => MEMORY_PORT_DIRECTION_READ
      case MWrite => MEMORY_PORT_DIRECTION_WRITE
      case MReadWrite => MEMORY_PORT_DIRECTION_READ_WRITE
    }
  }

  def convert(stmt: ir.Statement): Seq[Firrtl.Statement.Builder] = {
    stmt match {
      case ir.Block(stmts) => stmts.flatMap(convert(_))
      case ir.EmptyStmt => Seq.empty
      case other =>
        val sb = Firrtl.Statement.newBuilder()
        other match {
          case ir.DefNode(_, name, expr) =>
            val nb = Firrtl.Statement.Node.newBuilder()
              .setId(name)
              .setExpression(convert(expr))
            sb.setNode(nb)
          case ir.DefWire(_, name, tpe) =>
            val wb = Firrtl.Statement.Wire.newBuilder()
              .setId(name)
              .setType(convert(tpe))
            sb.setWire(wb)
          case ir.DefRegister(_, name, tpe, clock, reset, init) =>
            val rb = Firrtl.Statement.Register.newBuilder()
              .setId(name)
              .setType(convert(tpe))
              .setClock(convert(clock))
              .setReset(convert(reset))
              .setInit(convert(init))
            sb.setRegister(rb)
          case ir.DefInstance(_, name, module) =>
            val ib = Firrtl.Statement.Instance.newBuilder()
              .setId(name)
              .setModuleId(module)
            sb.setInstance(ib)
          case ir.Connect(_, loc, expr) =>
            val cb = Firrtl.Statement.Connect.newBuilder()
              .setLocation(convert(loc))
              .setExpression(convert(expr))
            sb.setConnect(cb)
          case ir.PartialConnect(_, loc, expr) =>
            val cb = Firrtl.Statement.PartialConnect.newBuilder()
              .setLocation(convert(loc))
              .setExpression(convert(expr))
            sb.setPartialConnect(cb)
          case ir.Conditionally(_, pred, conseq, alt) =>
            val cs = convert(conseq)
            val as = convert(alt)
            val wb = Firrtl.Statement.When.newBuilder()
              .setPredicate(convert(pred))
            cs.foreach(wb.addConsequent)
            as.foreach(wb.addOtherwise)
            sb.setWhen(wb)
          case ir.Print(_, string, args, clk, en) =>
            val pb = Firrtl.Statement.Printf.newBuilder()
              .setValue(string.string)
              .setClk(convert(clk))
              .setEn(convert(en))
            args.foreach(a => pb.addArg(convert(a)))
            sb.setPrintf(pb)
          case ir.Stop(_, ret, clk, en) =>
            val stopb = Firrtl.Statement.Stop.newBuilder()
              .setReturnValue(ret)
              .setClk(convert(clk))
              .setEn(convert(en))
            sb.setStop(stopb)
          case ir.IsInvalid(_, expr) =>
            val ib = Firrtl.Statement.IsInvalid.newBuilder()
              .setExpression(convert(expr))
            sb.setIsInvalid(ib)
          case ir.DefMemory(_, name, dtype, depth, wlat, rlat, rs, ws, rws, _) =>
            val mem = Firrtl.Statement.Memory.newBuilder()
              .setId(name)
              .setType(convert(dtype))
              .setDepth(depth)
              .setWriteLatency(wlat)
              .setReadLatency(rlat)
            mem.addAllReaderId(rs.asJava)
            mem.addAllWriterId(ws.asJava)
            mem.addAllReadwriterId(rws.asJava)
            sb.setMemory(mem)
          case CDefMemory(_, name, tpe, size, seq) =>
            val tpeb = convert(ir.VectorType(tpe, size))
            val mb = Firrtl.Statement.CMemory.newBuilder()
              .setId(name)
              .setType(tpeb)
              .setSyncRead(seq)
            sb.setCmemory(mb)
          case CDefMPort(_, name, _, mem, exprs, dir) =>
            val pb = Firrtl.Statement.MemoryPort.newBuilder()
              .setId(name)
              .setMemoryId(mem)
              .setMemoryIndex(convert(exprs.head))
              .setExpression(convert(exprs(1)))
              .setDirection(convert(dir))
            sb.setMemoryPort(pb)
          case ir.Attach(_, exprs) =>
            val ab = Firrtl.Statement.Attach.newBuilder()
            exprs.foreach(e => ab.addExpression(convert(e)))
            sb.setAttach(ab)
        }
        stmt match {
          case hasInfo: ir.HasInfo => sb.setSourceInfo(convert(hasInfo.info))
          case _ => // Do nothing
        }
        Seq(sb)
    }
  }

  def convert(field: ir.Field): Firrtl.Type.BundleType.Field.Builder = {
    val b = Firrtl.Type.BundleType.Field.newBuilder()
      .setId(field.name)
      .setIsFlipped(field.flip == ir.Flip)
      .setType(convert(field.tpe))
    b
  }

  /** Converts a Width to a ProtoBuf Width Builder
    *
    * @param width Input width
    * @return Option width where None means the width field should be cleared in the parent object
    */
  def convert(width: ir.Width): Option[Firrtl.Width.Builder] = width match {
    case ir.IntWidth(w) => Some(Firrtl.Width.newBuilder().setValue(w.toInt))
    case ir.UnknownWidth => None
  }

  def convert(vtpe: ir.VectorType): Firrtl.Type.VectorType.Builder =
    Firrtl.Type.VectorType.newBuilder()
      .setType(convert(vtpe.tpe))
      .setSize(vtpe.size)

  def convert(tpe: ir.Type): Firrtl.Type.Builder = {
    val tb = Firrtl.Type.newBuilder()
    tpe match {
      case ir.UIntType(width) =>
        val ut = Firrtl.Type.UIntType.newBuilder()
        convert(width).foreach(ut.setWidth)
        tb.setUintType(ut)
      case ir.SIntType(width) =>
        val st = Firrtl.Type.SIntType.newBuilder()
        convert(width).foreach(st.setWidth)
        tb.setSintType(st)
      case ir.FixedType(width, point) =>
        val ft = Firrtl.Type.FixedType.newBuilder()
        convert(width).foreach(ft.setWidth)
        convert(point).foreach(ft.setPoint)
        tb.setFixedType(ft)
      case ir.ClockType =>
        val ct = Firrtl.Type.ClockType.newBuilder()
        tb.setClockType(ct)
      case ir.AnalogType(width) =>
        val at = Firrtl.Type.AnalogType.newBuilder()
          convert(width).foreach(at.setWidth)
        tb.setAnalogType(at)
      case ir.BundleType(fields) =>
        val bt = Firrtl.Type.BundleType.newBuilder()
        fields.foreach(f => bt.addField(convert(f)))
        tb.setBundleType(bt)
      case vtpe: ir.VectorType =>
        val vtb = convert(vtpe)
        tb.setVectorType(vtb)
    }
  }

  def convert(direction: ir.Direction): Firrtl.Port.Direction = direction match {
    case ir.Input => Firrtl.Port.Direction.PORT_DIRECTION_IN
    case ir.Output => Firrtl.Port.Direction.PORT_DIRECTION_OUT
  }

  def convert(port: ir.Port): Firrtl.Port.Builder = {
    Firrtl.Port.newBuilder()
      .setId(port.name)
      .setDirection(convert(port.direction))
      .setType(convert(port.tpe))
  }

  def convert(param: ir.Param): Firrtl.Module.ExternalModule.Parameter.Builder = {
    import Firrtl.Module.ExternalModule._
    val pb = Parameter.newBuilder()
      .setId(param.name)
    param match {
      case ir.IntParam(_, value) =>
        pb.setInteger(convertToBigInt(value))
      case ir.DoubleParam(_, value) =>
        pb.setDouble(value)
      case ir.StringParam(_, ir.StringLit(value)) =>
        pb.setString(value)
      case ir.RawStringParam(_, value) =>
        pb.setRawString(value)
    }
  }

  def convert(module: ir.DefModule): Firrtl.Module.Builder = {
    val ports = module.ports.map(convert(_))
    val b = Firrtl.Module.newBuilder()
    module match {
      case mod: ir.Module =>
        val stmts = convert(mod.body)
        val mb = Firrtl.Module.UserModule.newBuilder()
          .setId(mod.name)
        ports.foreach(mb.addPort)
        stmts.foreach(mb.addStatement)
        b.setUserModule(mb)
      case ext: ir.ExtModule =>
        val eb = Firrtl.Module.ExternalModule.newBuilder()
          .setId(ext.name)
          .setDefinedName(ext.defname)
        ports.foreach(eb.addPort)
        val params = ext.params.map(convert(_))
        params.foreach(eb.addParameter)
        b.setExternalModule(eb)
    }
  }

  def convert(circuit: ir.Circuit): Firrtl = {
    val moduleBuilders = circuit.modules.map(convert(_))
    val cb = Firrtl.Circuit.newBuilder
      .addTop(Firrtl.Top.newBuilder().setName(circuit.main))
    for (m <- moduleBuilders) {
      cb.addModule(m)
    }
    Firrtl.newBuilder()
      .addCircuit(cb.build())
      .build()
  }
}
