// SPDX-License-Identifier: Apache-2.0

package chiselTests.properties

import chisel3._
import chisel3.properties.{AnyClassType, Class, ClassType, DynamicObject, Path, Property, PropertyType}
import chisel3.testing.scalatest.FileCheck
import chisel3.util.experimental.BoringUtils
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class PropertySpec extends AnyFlatSpec with Matchers with FileCheck {
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

    chirrtl should include("input intProp : Integer")
  }

  it should "support Int as a Property literal" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val propOut = IO(Output(Property[Int]()))
      propOut := Property(123)
    })

    chirrtl should include("propassign propOut, Integer(123)")
  }

  it should "support Long as a Property type" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val longProp = IO(Input(Property[Long]()))
    })

    chirrtl should include("input longProp : Integer")
  }

  it should "support Long as a Property literal" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val propOut = IO(Output(Property[Long]()))
      propOut := Property[Long](123)
    })

    chirrtl should include("propassign propOut, Integer(123)")
  }

  it should "support BigInt as a Property type" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val bigIntProp = IO(Input(Property[BigInt]()))
    })

    chirrtl should include("input bigIntProp : Integer")
  }

  it should "support BigInt as a Property literal" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val propOut = IO(Output(Property[BigInt]()))
      propOut := Property[BigInt](123)
    })

    chirrtl should include("propassign propOut, Integer(123)")
  }

  it should "support Double as a Property type" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val doubleProp = IO(Input(Property[Double]()))
    })

    chirrtl should include("input doubleProp : Double")
  }

  it should "support Double as a Property literal" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val propOut = IO(Output(Property[Double]()))
      propOut := Property[Double](123.456)
    })

    chirrtl should include("propassign propOut, Double(123.456)")
  }

  it should "support String as a Property type" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val stringProp = IO(Input(Property[String]()))
    })

    chirrtl should include("input stringProp : String")
  }

  it should "support String as a Property literal" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val propOut = IO(Output(Property[String]()))
      propOut := Property("fubar")
    })

    chirrtl should include("propassign propOut, String(\"fubar\")")
  }

  it should "escape special characters in Property String literals" in {
    val input = "foo\"\n\t\\bar"
    val expected = """foo\"\n\t\\bar"""
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val propOut = IO(Output(Property[String]()))
      propOut := Property(input)
    })

    chirrtl should include(s"""propassign propOut, String("$expected")""")
  }

  it should "not escape characters that do not need escaping in Property String literals" in {
    val input = "foo!@#$%^&*()_+bar"
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val propOut = IO(Output(Property[String]()))
      propOut := Property(input)
    })

    chirrtl should include(s"""propassign propOut, String("$input")""")
  }

  it should "support Boolean as a Property type" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val boolProp = IO(Input(Property[Boolean]()))
    })

    chirrtl should include("input boolProp : Bool")
  }

  it should "support Boolean as a Property literal" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val propOut = IO(Output(Property[Boolean]()))
      propOut := Property(false)
    })

    chirrtl should include("propassign propOut, Bool(false)")
  }

  it should "support paths as a Property type" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val pathProp = IO(Input(Property[Path]()))
    })

    chirrtl should include("input pathProp : Path")
  }

  it should "support path as a Property literal" in {
    ChiselStage.emitCHIRRTL {
      new Module {
        val propOutA = IO(Output(Property[Path]()))
        val propOutB = IO(Output(Property[Path]()))
        val propOutC = IO(Output(Property[Path]()))
        val propOutD = IO(Output(Property[Path]()))
        val propOutE = IO(Output(Property[Path]()))
        val propOutF = IO(Output(Property[Path]()))
        val propOutG = IO(Output(Property[Path]()))
        override def desiredName = "Top"
        val inst = Module(new Module {
          val localPropOut = IO(Output(Property[Path]()))
          val data = WireInit(false.B)
          val mem = SyncReadMem(1, Bool())
          val sram = chisel3.util.SRAM(1, Bool(), 1, 1, 0)
          localPropOut := Property(Path(data))
          override def desiredName = "Foo"
        })
        propOutA := Property(inst)
        propOutB := Property(inst.data)
        propOutC := Property(inst.mem)
        propOutD := Property(this)
        propOutE := inst.localPropOut
        propOutF := Property(Path(inst.sram.underlying.get))
        propOutG := Property(Path(inst.sram.underlying.get, true))
      }
    }.fileCheck()(
      """|CHECK-LABEL: module Foo :
         |CHECK:         propassign localPropOut, path("OMReferenceTarget:~|Foo>data")
         |CHECK-LABEL: public module Top :
         |CHECK:         propassign propOutA, path("OMInstanceTarget:~|Top/inst:Foo")
         |CHECK:         propassign propOutB, path("OMReferenceTarget:~|Top/inst:Foo>data")
         |CHECK:         propassign propOutC, path("OMReferenceTarget:~|Top/inst:Foo>mem")
         |CHECK:         propassign propOutD, path("OMInstanceTarget:~|Top")
         |CHECK:         propassign propOutE, inst.localPropOut
         |CHECK:         propassign propOutF, path("OMReferenceTarget:~|Top/inst:Foo>sram_sram")
         |CHECK:         propassign propOutG, path("OMMemberReferenceTarget:~|Top/inst:Foo>sram_sram")
         |""".stripMargin
    )
  }

  it should "support member path target types when requested" in {
    ChiselStage.emitCHIRRTL {
      new RawModule {
        val propOutA = IO(Output(Property[Path]()))
        val propOutB = IO(Output(Property[Path]()))
        val propOutC = IO(Output(Property[Path]()))
        val propOutD = IO(Output(Property[Path]()))
        override def desiredName = "Top"
        val inst = Module(new RawModule {
          val localPropOut = IO(Output(Property[Path]()))
          val data = WireInit(false.B)
          val mem = SyncReadMem(1, Bool())
          localPropOut := Property(Path(data, true))
          override def desiredName = "Foo"
        })
        propOutA := Property(Path(inst, true))
        propOutB := Property(Path(inst.data, true))
        propOutC := Property(Path(inst.mem, true))
        propOutD := inst.localPropOut
      }
    }.fileCheck()(
      """|CHECK-LABEL: module Foo :
         |CHECK:         propassign localPropOut, path("OMMemberReferenceTarget:~|Foo>data")
         |CHECK-LABEL: public module Top :
         |CHECK:         propassign propOutA, path("OMMemberInstanceTarget:~|Top/inst:Foo")
         |CHECK:         propassign propOutB, path("OMMemberReferenceTarget:~|Top/inst:Foo>data")
         |CHECK:         propassign propOutC, path("OMMemberReferenceTarget:~|Top/inst:Foo>mem")
         |CHECK:         propassign propOutD, inst.localPropOut
         |""".stripMargin
    )
  }

  // These are rejected by firtool anyway
  it should "reject Paths created in modules that are not direct ancestors of the referenced target" in {
    def runTest[A](mkTarget: => A, mkPath: A => Path): Unit = {
      val e = the[ChiselException] thrownBy {
        ChiselStage.emitCHIRRTL(new RawModule {
          val child1 = Module(new RawModule {
            override def desiredName = "Child1"
            val target = mkTarget
          })
          val captured = child1.target
          val child2 = Module(new RawModule {
            override def desiredName = "Child2"
            val out = IO(Output(Property[Path]()))
            out := Property(mkPath(captured))
          })
        })
      }
      e.getMessage should include("Requested .toRelativeTarget relative to Child2, but it is not an ancestor")
    }
    runTest[Data](WireInit(false.B), Path.apply)
    runTest[RawModule](Module(new RawModule {}), Path.apply)
    runTest[SyncReadMem[_]](SyncReadMem(1, Bool()), Path.apply)
  }

  it should "support deleted paths when requested" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val propOut = IO(Output(Property[Path]()))
      propOut := Property(Path.deleted)
    })

    chirrtl should include("""propassign propOut, path("OMDeleted:")""")
  }

  it should "support Properties on an ExtModule" in {
    // See: https://github.com/chipsalliance/chisel/issues/3509
    class Bar extends ExtModule {
      val a = IO(Output(Property[Int]()))
    }

    class Foo extends RawModule {
      val bar = Module(new Bar)
    }

    ChiselStage.emitCHIRRTL(new Foo) should include("output a : Integer")
  }

  it should "support connecting Property types of the same type" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val propIn = IO(Input(Property[Int]()))
      val propOut = IO(Output(Property[Int]()))
      propOut := propIn
    })

    chirrtl should include("propassign propOut, propIn")
  }

  it should "fail to compile when connectable connecting Property types of different types" in {
    assertTypeError("""new RawModule {
      val propIn = IO(Input(Property[Int]()))
      val propOut = IO(Output(Property[BigInt]()))
      propOut :#= propIn
    }""")
  }

  it should "support Seq[Int], Vector[Int], and List[Int] as a Property type" in {
    ChiselStage.emitCHIRRTL {
      new RawModule {
        val seqProp1 = IO(Input(Property[Seq[Int]]()))
        val seqProp2 = IO(Input(Property[Vector[Int]]()))
        val seqProp3 = IO(Input(Property[List[Int]]()))
      }
    }.fileCheck()(
      """|CHECK: input seqProp1 : List<Integer>
         |CHECK: input seqProp2 : List<Integer>
         |CHECK: input seqProp3 : List<Integer>
         |""".stripMargin
    )
  }

  it should "support nested Seqs as a Property type" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val nestedSeqProp = IO(Input(Property[Seq[Seq[Seq[Int]]]]()))
    })

    chirrtl should include("input nestedSeqProp : List<List<List<Integer>>>")
  }

  it should "support Seq[BigInt] as Property values" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val propOut = IO(Output(Property[Seq[BigInt]]()))
      propOut := Property(Seq[BigInt](123, 456)) // The Int => BigInt implicit conversion fails here
    })

    chirrtl should include("propassign propOut, List<Integer>(Integer(123), Integer(456))")
  }

  it should "support mixed Seqs of Integer literal and ports as Seq Property values" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val propIn = IO(Input(Property[BigInt]()))
      val propOut = IO(Output(Property[Seq[BigInt]]()))
      // Use connectable to show that Property[Seq[Property[A]]]
      propOut :#= Property(Seq(propIn, Property(BigInt(123))))
    })

    chirrtl should include("propassign propOut, List<Integer>(propIn, Integer(123))")
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

    chirrtl.fileCheck()(
      """|CHECK: output a : List<List<Integer>>
         |CHECK: output b : List<List<Integer>>
         |CHECK: propassign a, List<List<Integer>>(List<Integer>(Integer(123)))
         |CHECK: propassign b, List<List<Integer>>(List<Integer>(Integer(123)))
         |""".stripMargin
    )
  }

  it should "support concatenation of Property[Seq[Int]]" in {
    ChiselStage.emitCHIRRTL {
      new RawModule {
        val seqProp1 = IO(Input(Property[Seq[Int]]()))
        val seqProp2 = IO(Input(Property[Seq[Int]]()))
        val seqProp3 = IO(Output(Property[Seq[Int]]()))
        seqProp3 := seqProp1 ++ seqProp2
      }
    }.fileCheck()(
      """|CHECK: input seqProp1 : List<Integer>
         |CHECK: input seqProp2 : List<Integer>
         |CHECK: output seqProp3 : List<Integer>
         |CHECK: wire _seqProp3_propExpr : List<Integer>
         |CHECK: propassign _seqProp3_propExpr, list_concat(seqProp1, seqProp2)
         |""".stripMargin
    )
  }

  it should "support concatenation of Property[List[Int]]" in {
    ChiselStage.emitCHIRRTL {
      new RawModule {
        val listProp1 = IO(Input(Property[List[Int]]()))
        val listProp2 = IO(Input(Property[List[Int]]()))
        val listProp3 = IO(Output(Property[List[Int]]()))
        listProp3 := listProp1 ++ listProp2
      }
    }.fileCheck()(
      """|CHECK: input listProp1 : List<Integer>
         |CHECK: input listProp2 : List<Integer>
         |CHECK: output listProp3 : List<Integer>
         |CHECK: wire _listProp3_propExpr : List<Integer>
         |CHECK: propassign _listProp3_propExpr, list_concat(listProp1, listProp2)
         |""".stripMargin
    )
  }

  it should "not support types with nested Property[_]" in {
    assertTypeError("Property[Property[Property[Int]]]()")
    assertTypeError("Property[Property[Seq[Property[Int]]]]()")
    assertTypeError("Property[Property[Seq[Property[Seq[Property[Int]]]]]]()")
  }

  it should "be supported as a field of a Bundle" in {
    ChiselStage.emitCHIRRTL {
      class MyBundle extends Bundle {
        val foo = UInt(8.W)
        val bar = Property[BigInt]()
      }
      new RawModule {
        val propOut = IO(Output(new MyBundle))
        propOut.foo := 123.U
        propOut.bar := Property(3)
      }
    }.fileCheck()(
      """|CHECK: output propOut : { foo : UInt<8>, bar : Integer}
         |CHECK: connect propOut.foo, UInt<7>(0h7b)
         |CHECK: propassign propOut.bar, Integer(3)
         |""".stripMargin
    )
  }

  it should "being a flipped field of a Bundle" in {
    ChiselStage.emitCHIRRTL {
      class MyBundle extends Bundle {
        val foo = UInt(8.W)
        val bar = Flipped(Property[BigInt]())
      }
      new RawModule {
        val aligned = IO(new MyBundle)
        val flipped = IO(Flipped(new MyBundle))
        aligned.foo := flipped.foo
        flipped.bar := aligned.bar
      }
    }.fileCheck()(
      """|CHECK: output aligned : { foo : UInt<8>, flip bar : Integer}
         |CHECK: input flipped : { foo : UInt<8>, flip bar : Integer}
         |CHECK: connect aligned.foo, flipped.foo
         |CHECK: propassign flipped.bar, aligned.bar
         |""".stripMargin
    )
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

    ChiselStage.emitCHIRRTL {
      new MyBaseModule {
        aligned :<>= flipped
      }
    }.fileCheck()(
      """|CHECK: output aligned : { foo : String, flip bar : Integer}
         |CHECK: input flipped : { foo : String, flip bar : Integer}
         |CHECK: propassign flipped.bar, aligned.bar
         |CHECK: propassign aligned.foo, flipped.foo
         |""".stripMargin
    )

    ChiselStage.emitCHIRRTL {
      new MyBaseModule {
        aligned :<= flipped
      }
    }.fileCheck()(
      """|CHECK: output aligned : { foo : String, flip bar : Integer}
         |CHECK: input flipped : { foo : String, flip bar : Integer}
         |CHECK-NOT: propassign
         |CHECK: propassign aligned.foo, flipped.foo
         |CHECK-NOT: propassign
         |""".stripMargin
    )

    ChiselStage.emitCHIRRTL {
      new MyBaseModule {
        aligned :>= flipped
      }
    }.fileCheck()(
      """|CHECK: output aligned         : { foo : String, flip bar : Integer}
         |CHECK: input flipped  : { foo : String, flip bar : Integer}
         |CHECK-NOT: propassign
         |CHECK: propassign flipped.bar, aligned.bar
         |CHECK-NOT: propassign
         |""".stripMargin
    )

    ChiselStage.emitCHIRRTL {
      new RawModule {
        val out = IO(Output(new MyBundle))
        val in = IO(Input(new MyBundle))
        out :#= in
      }
    }.fileCheck()(
      """|CHECK: output out : { foo : String, bar : Integer}
         |CHECK: input in : { foo : String, bar : Integer}
         |CHECK: propassign out.bar, in.bar
         |CHECK: propassign out.foo, in.foo
         |""".stripMargin
    )
  }

  it should "support being nested in a Bundle in a wire" in {
    ChiselStage.emitCHIRRTL {
      class MyBundle extends Bundle {
        val foo = Property[String]()
        val bar = Flipped(Property[BigInt]())
      }
      new RawModule {
        val outgoing = IO(new MyBundle)
        val incoming = IO(Flipped(new MyBundle))
        val wire = Wire(new MyBundle)
        wire :<>= incoming
        outgoing :<>= wire
      }
    }.fileCheck()(
      """|CHECK: output outgoing : { foo : String, flip bar : Integer}
         |CHECK: input incoming : { foo : String, flip bar : Integer}
         |CHECK: wire wire : { foo : String, flip bar : Integer}
         |CHECK: propassign incoming.bar, wire.bar
         |CHECK: propassign wire.foo, incoming.foo
         |CHECK: propassign wire.bar, outgoing.bar
         |CHECK: propassign outgoing.foo, wire.foo
         |""".stripMargin
    )
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
    ChiselStage.emitCHIRRTL {
      new RawModule {
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
      }
    }.fileCheck()(
      """|CHECK:      input a : Inst<MyClass>
         |CHECK-NEXT: output b : List<Inst<MyClass>>
         |CHECK-NEXT: output c : Inst<MyClass>
         |CHECK-NEXT: output d : Inst<FooBar>
         |CHECK-NEXT: output e : Inst<FooBar>
         |CHECK-NEXT: output f : Inst<FooBar>
         |CHECK-NEXT: output g : AnyRef
         |CHECK-NEXT: output h : List<AnyRef>
         |CHECK-NEXT: output fooA : Inst<foo>
         |CHECK-NEXT: output fooB : Inst<foo>
         |CHECK-NEXT: output barA : Inst<bar>
         |CHECK-NEXT: output barB : Inst<bar>
         |
         |CHECK:      object myClass of MyClass
         |CHECK-NEXT: propassign b, List<Inst<MyClass>>(a, myClass)
         |CHECK-NEXT: propassign c, a
         |CHECK-NEXT: object obj of FooBar
         |CHECK-NEXT: propassign d, obj
         |CHECK-NEXT: propassign e, obj
         |CHECK-NEXT: propassign f, obj
         |CHECK-NEXT: propassign g, obj
         |CHECK-NEXT: propassign g, myClass
         |CHECK-NEXT: propassign h, List<AnyRef>(obj, myClass)
         |CHECK-NEXT: object fooObj of foo
         |CHECK-NEXT: propassign fooA, fooObj
         |CHECK-NEXT: propassign fooB, fooObj
         |CHECK-NEXT: object barObj of bar
         |CHECK-NEXT: propassign barA, barObj
         |CHECK-NEXT: propassign barB, barObj
         |""".stripMargin
    )
  }

  it should "support FlatIO" in {
    ChiselStage.emitCHIRRTL {
      new RawModule {
        val flatModule = Module(new RawModule {
          val io = FlatIO(new Bundle {
            val bool = Input(Bool())
            val prop = Input(Property[Int]())
          })
        })

        flatModule.io.bool := true.B
        flatModule.io.prop := Property(1)
      }
    }.fileCheck()(
      """|CHECK: connect flatModule.bool, UInt<1>(0h1)
         |CHECK: propassign flatModule.prop, Integer(1)
         |""".stripMargin
    )
  }

  it should "support FlatIO when used in a Bundle" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      class PropBundle extends Bundle {
        val int = Property[Int]()
      }

      val flatModule = Module(new RawModule {
        val io = FlatIO(new Bundle {
          val prop = Input(new PropBundle)
        })
      })

      flatModule.io.prop.int := Property(1)
    })

    chirrtl should include("propassign flatModule.prop.int, Integer(1)")
  }

  it should "support isLit" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val port = IO(Input(Property[Int]()))
      val lit = Property(1)

      port.isLit shouldBe false
      lit.isLit shouldBe true
    })
  }

  behavior.of("PropertyArithmeticOps")

  it should "support expressions in temporaries, wires, and ports" in {
    ChiselStage.emitCHIRRTL {
      new RawModule {
        val a = IO(Input(Property[Int]()))
        val b = IO(Input(Property[Int]()))
        val c = IO(Output(Property[Int]()))
        val d = IO(Output(Property[Int]()))
        val e = IO(Output(Property[Int]()))

        val t = a + b

        val w = WireInit(t)

        c := t
        d := t + a
        e := w + (a + b)
      }
    }.fileCheck()(
      """|CHECK: wire t : Integer
         |CHECK: propassign t, integer_add(a, b)
         |CHECK: wire w : Integer
         |CHECK: propassign w, t
         |CHECK: propassign c, t
         |CHECK: wire _d_propExpr : Integer
         |CHECK: propassign _d_propExpr, integer_add(t, a)
         |CHECK: propassign d, _d_propExpr
         |CHECK: wire _e_propExpr
         |CHECK: propassign _e_propExpr, integer_add(a, b)
         |CHECK: wire _e_propExpr_1
         |CHECK: propassign _e_propExpr_1, integer_add(w, _e_propExpr)
         |CHECK: propassign e, _e_propExpr_1
         |""".stripMargin
    )
  }

  it should "support boring from expressions" in {
    ChiselStage.emitCHIRRTL {
      new RawModule {
        val child = Module(new RawModule {
          val a = IO(Input(Property[Int]()))
          val b = IO(Input(Property[Int]()))
          val c = a + b
        })

        val a = IO(Input(Property[Int]()))
        val b = IO(Input(Property[Int]()))
        val c = IO(Output(Property[Int]()))

        child.a := a
        child.b := a
        c := BoringUtils.bore(child.c)
      }
    }.fileCheck()(
      """|CHECK: output c_bore : Integer
         |CHECK: wire c : Integer
         |CHECK: propassign c, integer_add(a, b)
         |CHECK: propassign c_bore, c
         |CHECK: propassign c, child.c_bore
         |""".stripMargin
    )
  }

  it should "support targeting the result of expressions" in {
    ChiselStage.emitCHIRRTL {
      new RawModule {
        override def desiredName = "Top"

        val mod = Module(new RawModule {
          override def desiredName = "Foo"
          val a = IO(Input(Property[Int]()))
          val b = IO(Input(Property[Int]()))
          val c = a + b
        })

        mod.c.toTarget.toString should equal("~|Foo>c")
      }
    }.fileCheck()(
      """|CHECK: wire c : Integer
         |CHECK: propassign c, integer_add(a, b)
         |""".stripMargin
    )
  }

  it should "not support expressions involving Property types that don't provide a typeclass instance" in {
    assertTypeError("""
      val a = Property[String]()
      val b = Property[String]()
      a + b
    """)
  }

  it should "not support expressions in Classes, and give a nice error" in {
    val e = the[ChiselException] thrownBy (ChiselStage.emitCHIRRTL(new RawModule {
      DynamicObject(new Class {
        val a = IO(Input(Property[BigInt]()))
        val b = IO(Input(Property[BigInt]()))
        val c = IO(Output(Property[BigInt]()))
        c := a + b
      })
    }))

    e.getMessage should include(
      "Property expressions are currently only supported in RawModules @[src/test/scala-2/chiselTests/properties/PropertySpec.scala"
    )
  }

  it should "support addition" in {
    ChiselStage.emitCHIRRTL {
      new RawModule {
        val a = IO(Input(Property[BigInt]()))
        val b = IO(Input(Property[BigInt]()))
        val c = IO(Output(Property[BigInt]()))
        c := a + b
      }
    }.fileCheck()(
      """|CHECK: wire _c_propExpr : Integer
         |CHECK: propassign _c_propExpr, integer_add(a, b)
         |CHECK: propassign c, _c_propExpr
         |""".stripMargin
    )
  }

  it should "support multiplication" in {
    ChiselStage.emitCHIRRTL {
      new RawModule {
        val a = IO(Input(Property[BigInt]()))
        val b = IO(Input(Property[BigInt]()))
        val c = IO(Output(Property[BigInt]()))
        c := a * b
      }
    }.fileCheck()(
      """|CHECK: wire _c_propExpr : Integer
         |CHECK: propassign _c_propExpr, integer_mul(a, b)
         |CHECK: propassign c, _c_propExpr
         |""".stripMargin
    )
  }

  it should "support shift right" in {
    ChiselStage.emitCHIRRTL {
      new RawModule {
        val a = IO(Input(Property[BigInt]()))
        val b = IO(Input(Property[BigInt]()))
        val c = IO(Output(Property[BigInt]()))
        c := a >> b
      }
    }.fileCheck()(
      """|CHECK: wire _c_propExpr : Integer
         |CHECK: propassign _c_propExpr, integer_shr(a, b)
         |CHECK: propassign c, _c_propExpr
         |""".stripMargin
    )
  }

  it should "support shift left" in {
    ChiselStage.emitCHIRRTL {
      new RawModule {
        val a = IO(Input(Property[BigInt]()))
        val b = IO(Input(Property[BigInt]()))
        val c = IO(Output(Property[BigInt]()))
        c := a << b
      }
    }.fileCheck()(
      """|CHECK: wire _c_propExpr : Integer
         |CHECK: propassign _c_propExpr, integer_shl(a, b)
         |CHECK: propassign c, _c_propExpr
         |""".stripMargin
    )
  }

  behavior.of("PropertySeqOps")

  it should "not support expressions involving Property types that don't provide a typeclass instance" in {
    assertTypeError("""
      val a = Property[String]()
      val b = Property[String]()
      a ++ b
    """)
  }

  it should "not support expressions in Classes, and give a nice error" in {
    val e = the[ChiselException] thrownBy (ChiselStage.emitCHIRRTL(new RawModule {
      DynamicObject(new Class {
        val a = IO(Input(Property[Seq[Int]]()))
        val b = IO(Input(Property[Seq[Int]]()))
        val c = IO(Output(Property[Seq[Int]]()))
        c := a ++ b
      })
    }))

    e.getMessage should include(
      "Property expressions are currently only supported in RawModules @[src/test/scala-2/chiselTests/properties/PropertySpec.scala"
    )
  }

  it should "support concatenation for Property[Seq[Int]]" in {
    ChiselStage.emitCHIRRTL {
      new RawModule {
        val a = IO(Input(Property[Seq[Int]]()))
        val b = IO(Input(Property[Seq[Int]]()))
        val c = IO(Output(Property[Seq[Int]]()))
        c := a ++ b
      }
    }.fileCheck()(
      """|CHECK: wire _c_propExpr : List<Integer>
         |CHECK: propassign _c_propExpr, list_concat(a, b)
         |CHECK: propassign c, _c_propExpr
         |""".stripMargin
    )
  }

  it should "support concatenation for Property[Seq[ClassType]]" in {
    ChiselStage.emitCHIRRTL {
      new RawModule {
        val a = IO(Input(Property[Seq[AnyClassType]]()))
        val b = IO(Input(Property[Seq[AnyClassType]]()))
        val c = IO(Output(Property[Seq[AnyClassType]]()))
        c := a ++ b
      }
    }.fileCheck()(
      """|CHECK: wire _c_propExpr : List<AnyRef>
         |CHECK: propassign _c_propExpr, list_concat(a, b)
         |CHECK: propassign c, _c_propExpr
         |""".stripMargin
    )
  }
}
