// SPDX-License-Identifier: Apache-2.0

package chiselTests.properties

import chisel3._
import chisel3.properties.{Class, Path, Property, PropertyType}
import chiselTests.{ChiselFlatSpec, MatchesAndOmits}
import circt.stage.ChiselStage
import chisel3.properties.ClassType
import chisel3.properties.AnyClassType

class PropertySpec extends ChiselFlatSpec with MatchesAndOmits {
  behavior.of("Property")

  it should "fail to compile with unsupported Property types" in {
    assertTypeError("""
      class MyThing
      val badProp = Property[MyThing]()
    """)
  }

  it should "fail to compile with unsupported Property literals" in {
    assertTypeError("""
      class MyThing
      val badProp = Property(new MyThing)
    """)
  }

  it should "support Int as a Property type" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val intProp = IO(Input(Property[Int]()))
    })

    matchesAndOmits(chirrtl)(
      "input intProp : Integer"
    )()
  }

  it should "support Int as a Property literal" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val propOut = IO(Output(Property[Int]()))
      propOut := Property(123)
    })

    matchesAndOmits(chirrtl)(
      "propassign propOut, Integer(123)"
    )()
  }

  it should "support Long as a Property type" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val longProp = IO(Input(Property[Long]()))
    })

    matchesAndOmits(chirrtl)(
      "input longProp : Integer"
    )()
  }

  it should "support Long as a Property literal" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val propOut = IO(Output(Property[Long]()))
      propOut := Property[Long](123)
    })

    matchesAndOmits(chirrtl)(
      "propassign propOut, Integer(123)"
    )()
  }

  it should "support BigInt as a Property type" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val bigIntProp = IO(Input(Property[BigInt]()))
    })

    matchesAndOmits(chirrtl)(
      "input bigIntProp : Integer"
    )()
  }

  it should "support BigInt as a Property literal" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val propOut = IO(Output(Property[BigInt]()))
      propOut := Property[BigInt](123)
    })

    matchesAndOmits(chirrtl)(
      "propassign propOut, Integer(123)"
    )()
  }

  it should "support Double as a Property type" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val doubleProp = IO(Input(Property[Double]()))
    })

    matchesAndOmits(chirrtl)(
      "input doubleProp : Double"
    )()
  }

  it should "support Double as a Property literal" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val propOut = IO(Output(Property[Double]()))
      propOut := Property[Double](123.456)
    })

    matchesAndOmits(chirrtl)(
      "propassign propOut, Double(123.456)"
    )()
  }

  it should "support String as a Property type" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val stringProp = IO(Input(Property[String]()))
    })

    matchesAndOmits(chirrtl)(
      "input stringProp : String"
    )()
  }

  it should "support String as a Property literal" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val propOut = IO(Output(Property[String]()))
      propOut := Property("fubar")
    })

    matchesAndOmits(chirrtl)(
      "propassign propOut, String(\"fubar\")"
    )()
  }

  it should "support Boolean as a Property type" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val boolProp = IO(Input(Property[Boolean]()))
    })

    matchesAndOmits(chirrtl)(
      "input boolProp : Bool"
    )()
  }

  it should "support Boolean as a Property literal" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val propOut = IO(Output(Property[Boolean]()))
      propOut := Property(false)
    })

    matchesAndOmits(chirrtl)(
      "propassign propOut, Bool(false)"
    )()
  }

  it should "support paths as a Property type" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val pathProp = IO(Input(Property[Path]()))
    })

    matchesAndOmits(chirrtl)(
      "input pathProp : Path"
    )()
  }

  it should "support path as a Property literal" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val propOutA = IO(Output(Property[Path]()))
      val propOutB = IO(Output(Property[Path]()))
      val propOutC = IO(Output(Property[Path]()))
      val propOutD = IO(Output(Property[Path]()))
      override def desiredName = "Top"
      val inst = Module(new RawModule {
        val data = WireInit(false.B)
        val mem = SyncReadMem(1, Bool())
        override def desiredName = "Foo"
      })
      propOutA := Property(inst)
      propOutB := Property(inst.data)
      propOutC := Property(inst.mem)
      propOutD := Property(this)
    })

    matchesAndOmits(chirrtl)(
      """propassign propOutA, path("OMInstanceTarget:~Top|Top/inst:Foo")""",
      """propassign propOutB, path("OMReferenceTarget:~Top|Top/inst:Foo>data")""",
      """propassign propOutC, path("OMReferenceTarget:~Top|Top/inst:Foo>mem")""",
      """propassign propOutD, path("OMInstanceTarget:~Top|Top")"""
    )()
  }

  it should "support member path target types when requested" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val propOutA = IO(Output(Property[Path]()))
      val propOutB = IO(Output(Property[Path]()))
      val propOutC = IO(Output(Property[Path]()))
      override def desiredName = "Top"
      val inst = Module(new RawModule {
        val data = WireInit(false.B)
        val mem = SyncReadMem(1, Bool())
        override def desiredName = "Foo"
      })
      propOutA := Property(Path(inst, true))
      propOutB := Property(Path(inst.data, true))
      propOutC := Property(Path(inst.mem, true))
    })

    matchesAndOmits(chirrtl)(
      """propassign propOutA, path("OMMemberInstanceTarget:~Top|Top/inst:Foo")""",
      """propassign propOutB, path("OMMemberReferenceTarget:~Top|Top/inst:Foo>data")""",
      """propassign propOutC, path("OMMemberReferenceTarget:~Top|Top/inst:Foo>mem")"""
    )()
  }

  it should "support Properties on an ExtModule" in {
    // See: https://github.com/chipsalliance/chisel/issues/3509
    class Bar extends experimental.ExtModule {
      val a = IO(Output(Property[Int]()))
    }

    class Foo extends RawModule {
      val bar = Module(new Bar)
    }

    val chirrtl = ChiselStage.emitCHIRRTL(new Foo)
    matchesAndOmits(chirrtl)(
      "output a : Integer"
    )()
  }

  it should "support connecting Property types of the same type" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val propIn = IO(Input(Property[Int]()))
      val propOut = IO(Output(Property[Int]()))
      propOut := propIn
    })

    matchesAndOmits(chirrtl)(
      "propassign propOut, propIn"
    )()
  }

  it should "fail to compile when connectable connecting Property types of different types" in {
    assertTypeError("""new RawModule {
      val propIn = IO(Input(Property[Int]()))
      val propOut = IO(Output(Property[BigInt]()))
      propOut :#= propIn
    }""")
  }

  it should "support Seq[Int], Vector[Int], and List[Int] as a Property type" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val seqProp1 = IO(Input(Property[Seq[Int]]()))
      val seqProp2 = IO(Input(Property[Vector[Int]]()))
      val seqProp3 = IO(Input(Property[List[Int]]()))
    })

    matchesAndOmits(chirrtl)(
      "input seqProp1 : List<Integer>",
      "input seqProp2 : List<Integer>",
      "input seqProp3 : List<Integer>"
    )()
  }

  it should "support nested Seqs as a Property type" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val nestedSeqProp = IO(Input(Property[Seq[Seq[Seq[Int]]]]()))
    })

    matchesAndOmits(chirrtl)(
      "input nestedSeqProp : List<List<List<Integer>>>"
    )()
  }

  it should "support Seq[BigInt] as Property values" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val propOut = IO(Output(Property[Seq[BigInt]]()))
      propOut := Property(Seq[BigInt](123, 456)) // The Int => BigInt implicit conversion fails here
    })

    matchesAndOmits(chirrtl)(
      "propassign propOut, List<Integer>(Integer(123), Integer(456))"
    )()
  }

  it should "support mixed Seqs of Integer literal and ports as Seq Property values" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val propIn = IO(Input(Property[BigInt]()))
      val propOut = IO(Output(Property[Seq[BigInt]]()))
      // Use connectable to show that Property[Seq[Property[A]]]
      propOut :#= Property(Seq(propIn, Property(BigInt(123))))
    })

    matchesAndOmits(chirrtl)(
      "propassign propOut, List<Integer>(propIn, Integer(123))"
    )()
  }

  it should "support nested collections without nested Property[_] values" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val a = IO(Output(Property[Seq[Seq[Property[Int]]]]()))
      val b = IO(Output(Property[Seq[Property[Seq[Int]]]]()))
      a := Property(Seq[Seq[Int]](Seq(123)))
      b := Property(Seq[Seq[Int]](Seq(123)))
    })

    assertTypeError {
      "Property[Property[Property[Int]]]()"
    }

    matchesAndOmits(chirrtl)(
      "output a : List<List<Integer>>",
      "output b : List<List<Integer>>",
      """propassign a, List<List<Integer>>(List<Integer>(Integer(123)))""",
      """propassign b, List<List<Integer>>(List<Integer>(Integer(123)))"""
    )()
  }

  it should "not support types with nested Property[_]" in {
    assertTypeError("Property[Property[Property[Int]]]()")
    assertTypeError("Property[Property[Seq[Property[Int]]]]()")
    assertTypeError("Property[Property[Seq[Property[Seq[Property[Int]]]]]]()")
  }

  it should "be supported as a field of a Bundle" in {
    class MyBundle extends Bundle {
      val foo = UInt(8.W)
      val bar = Property[BigInt]()
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val propOut = IO(Output(new MyBundle))
      propOut.foo := 123.U
      propOut.bar := Property(3)
    })
    matchesAndOmits(chirrtl)(
      "output propOut : { foo : UInt<8>, bar : Integer}",
      "connect propOut.foo, UInt<7>(0h7b)",
      "propassign propOut.bar, Integer(3)"
    )()
  }

  it should "being a flipped field of a Bundle" in {
    class MyBundle extends Bundle {
      val foo = UInt(8.W)
      val bar = Flipped(Property[BigInt]())
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val aligned = IO(new MyBundle)
      val flipped = IO(Flipped(new MyBundle))
      aligned.foo := flipped.foo
      flipped.bar := aligned.bar
    })
    matchesAndOmits(chirrtl)(
      "output aligned : { foo : UInt<8>, flip bar : Integer}",
      "input flipped : { foo : UInt<8>, flip bar : Integer}",
      "propassign flipped.bar, aligned.bar",
      "connect aligned.foo, flipped.foo"
    )()
  }

  it should "support connectable operators when nested in a Bundle" in {
    class MyBundle extends Bundle {
      val foo = Property[String]()
      val bar = Flipped(Property[BigInt]())
    }
    abstract class MyBaseModule extends RawModule {
      val aligned = IO(new MyBundle)
      val flipped = IO(Flipped(new MyBundle))
    }
    val chirrtl1 = ChiselStage.emitCHIRRTL(new MyBaseModule {
      aligned :<>= flipped
    })
    matchesAndOmits(chirrtl1)(
      "output aligned : { foo : String, flip bar : Integer}",
      "input flipped : { foo : String, flip bar : Integer}",
      "propassign flipped.bar, aligned.bar",
      "propassign aligned.foo, flipped.foo"
    )()

    val chirrtl2 = ChiselStage.emitCHIRRTL(new MyBaseModule {
      aligned :<= flipped
    })
    matchesAndOmits(chirrtl2)(
      "output aligned : { foo : String, flip bar : Integer}",
      "input flipped : { foo : String, flip bar : Integer}",
      "propassign aligned.foo, flipped.foo"
    )("propassign flipped.bar, aligned.bar")

    val chirrtl3 = ChiselStage.emitCHIRRTL(new MyBaseModule {
      aligned :>= flipped
    })
    matchesAndOmits(chirrtl3)(
      "output aligned : { foo : String, flip bar : Integer}",
      "input flipped : { foo : String, flip bar : Integer}",
      "propassign flipped.bar, aligned.bar"
    )("propassign aligned.foo, flipped.foo")

    val chirrtl4 = ChiselStage.emitCHIRRTL(new RawModule {
      val out = IO(Output(new MyBundle))
      val in = IO(Input(new MyBundle))
      out :#= in
    })
    matchesAndOmits(chirrtl4)(
      "output out : { foo : String, bar : Integer}",
      "input in : { foo : String, bar : Integer}",
      "propassign out.bar, in.bar",
      "propassign out.foo, in.foo"
    )()
  }

  it should "NOT support <>" in {
    class MyBundle extends Bundle {
      val foo = Property[String]()
      val bar = Flipped(Property[BigInt]())
    }
    val e = the[ChiselException] thrownBy ChiselStage.emitCHIRRTL(
      new RawModule {
        val aligned = IO(new MyBundle)
        val flipped = IO(Flipped(new MyBundle))
        aligned <> flipped
      },
      Array("--throw-on-first-error")
    )
    e.getMessage should include("Field '_.bar' of type Property[Integer] does not support <>, use :<>= instead")
  }

  it should "support being nested in a Bundle in a wire" in {
    class MyBundle extends Bundle {
      val foo = Property[String]()
      val bar = Flipped(Property[BigInt]())
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val outgoing = IO(new MyBundle)
      val incoming = IO(Flipped(new MyBundle))
      val wire = Wire(new MyBundle)
      wire :<>= incoming
      outgoing :<>= wire
    })
    matchesAndOmits(chirrtl)(
      "output outgoing : { foo : String, flip bar : Integer}",
      "input incoming : { foo : String, flip bar : Integer}",
      "wire wire : { foo : String, flip bar : Integer}",
      "propassign incoming.bar, wire.bar",
      "propassign wire.foo, incoming.foo",
      "propassign wire.bar, outgoing.bar",
      "propassign outgoing.foo, wire.foo"
    )()
  }

  it should "have None litOption" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val propOut = IO(Output(Property[BigInt]()))
      val propLit = Property[BigInt](123)
      propOut.litOption should be(None)
      // Even though we could technically return a value for Property[Int|Long|BigInt], it's misleading
      // since the typical litOption is for hardware bits values
      // If we want an API to get literal values out of Properties, we should add a different API that returns type T
      propLit.litOption should be(None)
    })
  }

  it should "give a decent error when .asUInt is called on it" in {
    class MyBundle extends Bundle {
      val foo = UInt(8.W)
      val bar = Property[BigInt]()
    }

    val e1 = the[ChiselException] thrownBy (ChiselStage.emitCHIRRTL(
      new RawModule {
        val in = IO(Input(Property[String]()))
        in.asUInt
      },
      Array("--throw-on-first-error")
    ))
    e1.getMessage should include("Property[String] does not support .asUInt.")

    val e2 = the[ChiselException] thrownBy (ChiselStage.emitCHIRRTL(
      new RawModule {
        val in = IO(Input(new MyBundle))
        in.asUInt
      },
      Array("--throw-on-first-error")
    ))
    e2.getMessage should include("Field '_.bar' of type Property[Integer] does not support .asUInt")
  }

  it should "give a decent error when used in a printf" in {
    class MyBundle extends Bundle {
      val foo = UInt(8.W)
      val bar = Property[BigInt]()
    }
    val e = the[ChiselException] thrownBy (ChiselStage.emitCHIRRTL(
      new RawModule {
        val in = IO(Input(new MyBundle))
        printf(cf"in = $in\n")
      },
      Array("--throw-on-first-error")
    ))
    e.getMessage should include("Properties do not support hardware printing: 'in_bar', in module 'PropertySpec_Anon'")
  }

  it should "not be able to get the ClassType of a Property[T] if T != ClassType" in {
    assertTypeError("""
      val intProp = Property[Int](123)
      val foobar = Property[intProp.ClassType]()
    """)
  }

  it should "emit correct types for all the ways of creating class references and properties" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val cls = ClassType.unsafeGetClassTypeByName("MyClass")
      val a = IO(Input(Property[cls.Type]()))
      val b = IO(Output(Property[Seq[cls.Type]]()))
      val c = IO(Output(a.cloneType))

      val myClass = Class.unsafeGetDynamicObject("MyClass")

      // collections of Property[ClassType] need to be cast to a specific ClassType
      b :#= Property(Seq(a.as(cls), myClass.getReference.as(cls)))
      assertTypeError("""
        b :#= Property(Seq(a, myClass.getReference))
      """)

      // this is still fine because we already know the ClassType of a
      c :#= a

      // note that assigning to a property of the wrong ClassType is still possible, because everything has type Property[ClassType]
      c :#= Class.unsafeGetDynamicObject("SomeOtherClass").getReference

      val obj = Class.unsafeGetDynamicObject("FooBar")
      val objRef = obj.getReference
      val d = IO(Output(objRef.cloneType))
      val e = IO(Output(Property[objRef.ClassType]()))
      val f = IO(Output(Class.unsafeGetReferenceType(obj.className.name)))

      d :#= obj.getReference
      e :#= obj.getReference
      f :#= obj.getReference

      // AnyRef
      val g = IO(Output(Property[AnyClassType]()))
      val h = IO(Output(Property[Seq[AnyClassType]]()))
      g :#= objRef
      g :#= myClass.getReference
      h :#= Property(Seq(objRef.asAnyClassType, myClass.getReference.asAnyClassType))

      // should work with methods
      def connectAB(cls: ClassType) = {
        val a = IO(Output(Property[cls.Type]())).suggestName(cls.name + "A")
        val b = IO(Output(Property[cls.Type]())).suggestName(cls.name + "B")
        val obj = Class.unsafeGetDynamicObject(cls.name).suggestName(cls.name + "Obj")
        a := obj.getReference
        b := obj.getReference
      }

      connectAB(ClassType.unsafeGetClassTypeByName("foo"))
      connectAB(ClassType.unsafeGetClassTypeByName("bar"))
    })

    matchesAndOmits(chirrtl)(
      "input a : Inst<MyClass>",
      "output b : List<Inst<MyClass>>",
      "output c : Inst<MyClass>",
      "object myClass of MyClass",
      "propassign b, List<Inst<MyClass>>(a, myClass)",
      "propassign c, a",
      "object obj of FooBar",
      "output e : Inst<FooBar>",
      "output d : Inst<FooBar>",
      "output f : Inst<FooBar>",
      "propassign d, obj",
      "propassign e, obj",
      "propassign f, obj",
      "output g : AnyRef",
      "output h : List<AnyRef>",
      "propassign g, obj",
      "propassign g, myClass",
      "propassign h, List<AnyRef>(obj, myClass)",
      "output fooA : Inst<foo>",
      "output fooB : Inst<foo>",
      "object fooObj of foo",
      "propassign fooA, fooObj",
      "propassign fooB, fooObj",
      "output barA : Inst<bar>",
      "output barB : Inst<bar>",
      "object barObj of bar",
      "propassign barA, barObj",
      "propassign barB, barObj"
    )()
  }
}
