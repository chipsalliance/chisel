// See LICENSE for license details.

package firrtlTests

import java.io.{ByteArrayInputStream, ByteArrayOutputStream}

import firrtl.FirrtlProtos.Firrtl
import firrtl._
import firrtl.ir._
import firrtl.Mappers._

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
    FirrtlResourceTest("FPU", "/regress")
  )

  for (FirrtlResourceTest(name, dir) <- firrtlResourceTests) {
    s"$name" should "work with Protobuf serialization and deserialization" in {
      val stream = getClass.getResourceAsStream(s"$dir/$name.fir")
      val circuit = parse(scala.io.Source.fromInputStream(stream).getLines.mkString("\n"))

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

  it should "supported FixedType" in {
    val ftpe = ir.FixedType(IntWidth(8), IntWidth(4))
    FromProto.convert(ToProto.convert(ftpe).build) should equal (ftpe)
  }

  it should "supported FixedLiteral" in {
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
}
