// See LICENSE for license details.

package firrtlTests

import firrtl.FirrtlProtos.Firrtl
import firrtl._
import firrtl.ir._
import firrtl.testutils._
import firrtl.Utils.BoolType

class ProtoBufSpec extends FirrtlFlatSpec {

  /** Tests in src/test/resource/ in .fir format
    *
    * @note These tests rely on the ANTLR Parser
    */
  case class FirrtlResourceTest(name: String, resourceDir: String)

  val firrtlResourceTests = List(
    FirrtlResourceTest("GCDTester", "/integration"),
    FirrtlResourceTest("RightShiftTester", "/integration"),
    FirrtlResourceTest("MemTester", "/integration"),
    FirrtlResourceTest("PipeTester", "/integration"),
    FirrtlResourceTest("Rob", "/regress"),
    FirrtlResourceTest("RocketCore", "/regress"),
    FirrtlResourceTest("ICache", "/regress"),
    FirrtlResourceTest("FPU", "/regress"),
    FirrtlResourceTest("AsyncResetTester", "/features")
  )

  for (FirrtlResourceTest(name, dir) <- firrtlResourceTests) {
    s"$name" should "work with Protobuf serialization and deserialization" in {
      val circuit = parse(FileUtils.getTextResource(s"$dir/$name.fir"))

      // Test ToProto and FromProto
      val protobuf = proto.ToProto.convert(circuit)
      val pcircuit = proto.FromProto.convert(protobuf)
      canonicalize(circuit).serialize should equal(canonicalize(pcircuit).serialize)

      // Test ProtoBuf generated serialization and deserialization
      val ostream = new java.io.ByteArrayOutputStream()
      protobuf.writeTo(ostream)
      val istream = new java.io.ByteArrayInputStream(ostream.toByteArray)
      val cistream = com.google.protobuf.CodedInputStream.newInstance(istream)
      cistream.setRecursionLimit(Integer.MAX_VALUE)
      val protobuf2 = firrtl.FirrtlProtos.Firrtl.parseFrom(cistream)
      protobuf2 should equal (protobuf)

      // Test that our faster serialization matches generated serialization
      val ostream2 = new java.io.ByteArrayOutputStream
      proto.ToProto.writeToStream(ostream2, circuit)
      ostream2.toByteArray.toList should equal (ostream.toByteArray.toList)
    }
  }

  // ********** Focused Tests **********
  // The goal is to fill coverage holes left after the above

  behavior of "ProtoBuf serialization and deserialization"
  import firrtl.proto._

  it should "support UnknownWidth" in {
    // Note that this has to be handled in the parent object so we need to test everything that has a width
    val uint = ir.UIntType(ir.UnknownWidth)
    FromProto.convert(ToProto.convert(uint).build) should equal (uint)

    val sint = ir.SIntType(ir.UnknownWidth)
    FromProto.convert(ToProto.convert(sint).build) should equal (sint)

    val ftpe = ir.FixedType(ir.UnknownWidth, ir.UnknownWidth)
    FromProto.convert(ToProto.convert(ftpe).build) should equal (ftpe)

    val atpe = ir.AnalogType(ir.UnknownWidth)
    FromProto.convert(ToProto.convert(atpe).build) should equal (atpe)

    val ulit = ir.UIntLiteral(123, ir.UnknownWidth)
    FromProto.convert(ToProto.convert(ulit).build) should equal (ulit)

    val slit = ir.SIntLiteral(-123, ir.UnknownWidth)
    FromProto.convert(ToProto.convert(slit).build) should equal (slit)

    val flit = ir.FixedLiteral(-123, ir.UnknownWidth, ir.UnknownWidth)
    FromProto.convert(ToProto.convert(flit).build) should equal (flit)
  }

  it should "support all Primops" in {
    val builtInOps = PrimOps.listing.map(PrimOps.fromString(_))
    for (op <- builtInOps) {
      val expr = DoPrim(op, List.empty, List.empty, ir.UnknownType)
      FromProto.convert(ToProto.convert(expr).build) should equal (expr)
    }
  }

  it should "support all ExtModule features (except info which isn't yet supported by Chisel)" in {
    val ports = Seq(
      Port(ir.NoInfo, "port1", ir.Input, ir.UIntType(ir.IntWidth(8))),
      Port(ir.NoInfo, "port2", ir.Output, ir.SIntType(ir.IntWidth(8)))
    )
    val params = Seq(
      IntParam("param1", BigInt(Long.MinValue)),
      DoubleParam("param2", Double.NegativeInfinity),
      StringParam("param3", ir.StringLit("quite the string!")),
      RawStringParam("param4", "get some raw strings")
    )
    val ext = ir.ExtModule(ir.NoInfo, "MyModule", ports, "DefNameHere", params)
    FromProto.convert(ToProto.convert(ext).build) should equal (ext)
  }

  it should "support FixedType" in {
    val ftpe = ir.FixedType(IntWidth(8), IntWidth(4))
    FromProto.convert(ToProto.convert(ftpe).build) should equal (ftpe)
  }

  it should "support FixedLiteral" in {
    val flit = ir.FixedLiteral(3, IntWidth(8), IntWidth(4))
    FromProto.convert(ToProto.convert(flit).build) should equal (flit)
  }

  it should "support Analog and Attach" in {
    val analog = ir.AnalogType(IntWidth(8))
    FromProto.convert(ToProto.convert(analog).build) should equal (analog)

    val attach = ir.Attach(ir.NoInfo, Seq(Reference("hi", ir.UnknownType)))
    FromProto.convert(ToProto.convert(attach).head.build) should equal (attach)
  }

  // Regression tests were generated before Chisel could emit else
  it should "support whens with elses" in {
    val expr = Reference("hi", ir.UnknownType)
    val stmt = Connect(ir.NoInfo, expr, expr)
    val when = ir.Conditionally(ir.NoInfo, expr, stmt, stmt)
    FromProto.convert(ToProto.convert(when).head.build) should equal (when)
  }

  it should "support SIntLiteral with a width" in {
    val slit = ir.SIntLiteral(-123)
    FromProto.convert(ToProto.convert(slit).build) should equal (slit)
  }

  // Backwards compatibility
  it should "support mems using old uint32 and new BigInt" in {
    val size = 128
    val mem = DefMemory(NoInfo, "m", UIntType(IntWidth(8)), size, 1, 1, List("r"), List("w"), List("rw"))
    val builder = ToProto.convert(mem).head
    val defaultProto = builder.build()
    val oldProto = Firrtl.Statement.newBuilder().setMemory(
      builder.getMemoryBuilder.clearDepth().setUintDepth(size)
    ).build()
    // These Proto messages are not the same
    defaultProto shouldNot equal (oldProto)

    val defaultMem = FromProto.convert(defaultProto)
    val oldMem = FromProto.convert(oldProto)

    // But they both deserialize to the original!
    defaultMem should equal (mem)
    oldMem should equal (mem)
  }

  // Backwards compatibility
  it should "support cmems using old VectorType and new TypeAndDepth" in {
    val size = 128
    val cmem = CDefMemory(NoInfo, "m", UIntType(IntWidth(8)), size, true)
    val vtpe = ToProto.convert(VectorType(UIntType(IntWidth(8)), size))
    val builder = ToProto.convert(cmem).head
    val defaultProto = builder.build()
    val oldProto = Firrtl.Statement.newBuilder().setCmemory(
      builder.getCmemoryBuilder.clearTypeAndDepth().setVectorType(vtpe)
    ).build()
    // These Proto messages are not the same
    defaultProto shouldNot equal (oldProto)

    val defaultCMem = FromProto.convert(defaultProto)
    val oldCMem = FromProto.convert(oldProto)

    // But they both deserialize to the original!
    defaultCMem should equal (cmem)
    oldCMem should equal (cmem)
  }

  // readunderwrite support
  it should "support readunderwrite parameters" in {
    val m1 = DefMemory(NoInfo, "m", UIntType(IntWidth(8)), 128, 1, 1, List("r"), List("w"), Nil, ir.ReadUnderWrite.Old)
    FromProto.convert(ToProto.convert(m1).head.build) should equal (m1)

    val m2 = m1.copy(readUnderWrite = ir.ReadUnderWrite.New)
    FromProto.convert(ToProto.convert(m2).head.build) should equal (m2)

    val cm1 = CDefMemory(NoInfo, "m", UIntType(IntWidth(8)), 128, true, ir.ReadUnderWrite.Old)
    FromProto.convert(ToProto.convert(cm1).head.build) should equal (cm1)

    val cm2 = cm1.copy(readUnderWrite = ir.ReadUnderWrite.New)
    FromProto.convert(ToProto.convert(cm2).head.build) should equal (cm2)
  }

  it should "support AsyncResetTypes" in {
    val port = ir.Port(ir.NoInfo, "reset", ir.Input, ir.AsyncResetType)
    FromProto.convert(ToProto.convert(port).build) should equal (port)
  }

  it should "support ResetTypes" in {
    val port = ir.Port(ir.NoInfo, "reset", ir.Input, ir.ResetType)
    FromProto.convert(ToProto.convert(port).build) should equal (port)
  }

  it should "support ValidIf" in {
    val en = ir.Reference("en", BoolType, PortKind, SourceFlow)
    val value = ir.Reference("x", UIntType(IntWidth(8)), PortKind, SourceFlow)
    val vi = ir.ValidIf(en, value, value.tpe)
    // Deserialized has almost nothing filled in
    val expected = ir.ValidIf(ir.Reference("en"), ir.Reference("x"), UnknownType)
    FromProto.convert(ToProto.convert(vi).build) should equal (expected)
  }
}
