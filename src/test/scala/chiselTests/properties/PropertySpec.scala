// SPDX-License-Identifier: Apache-2.0

package chiselTests.properties

import chisel3._
import chisel3.properties.Property
import chiselTests.{ChiselFlatSpec, MatchesAndOmits}
import circt.stage.ChiselStage
import scala.collection.immutable.{ListMap, SeqMap, VectorMap}

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

  it should "fail to compile when connecting Property types of different types" in {
    assertTypeError("""
      new RawModule {
        val propIn = IO(Input(Property[Int]()))
        val propOut = IO(Output(Property[BigInt]()))
        propOut := propIn
      }
    """)
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
      propOut := Property(Seq(propIn, Property(BigInt(123))))
    })

    matchesAndOmits(chirrtl)(
      "propassign propOut, List<Integer>(propIn, Integer(123))"
    )()
  }

  it should "support SeqMap[Int], VectorMap[Int], and ListMap[Int] as a Property type" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val mapProp1 = IO(Input(Property[SeqMap[String, Int]]()))
      val mapProp2 = IO(Input(Property[VectorMap[String, Int]]()))
      val mapProp3 = IO(Input(Property[ListMap[String, Int]]()))
    })

    matchesAndOmits(chirrtl)(
      "input mapProp1 : Map<Integer>",
      "input mapProp2 : Map<Integer>",
      "input mapProp3 : Map<Integer>"
    )()
  }

  it should "support nested Maps as a Property type" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val nestedMapProp = IO(Input(Property[SeqMap[String, SeqMap[String, SeqMap[String, Int]]]]()))
    })

    matchesAndOmits(chirrtl)(
      "input nestedMapProp : Map<Map<Map<Integer>>>"
    )()
  }

  it should "support SeqMap[String, BigInt] as Property values" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val propOut = IO(Output(Property[SeqMap[String, BigInt]]()))
      propOut := Property(
        SeqMap[String, BigInt]("foo" -> 123, "bar" -> 456)
      ) // The Int => BigInt implicit conversion fails here
    })

    matchesAndOmits(chirrtl)(
      """propassign propOut, Map<Integer>("foo" -> Integer(123), "bar" -> Integer(456))"""
    )()
  }

  it should "support mixed Maps of Integer literal and ports as Map Property values" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val propIn = IO(Input(Property[BigInt]()))
      val propOut = IO(Output(Property[SeqMap[String, BigInt]]()))
      propOut := Property(SeqMap("foo" -> propIn, "bar" -> Property(BigInt(123))))
    })

    matchesAndOmits(chirrtl)(
      """propassign propOut, Map<Integer>("foo" -> propIn, "bar" -> Integer(123))"""
    )()
  }

  it should "support nested collections without nested Property[_] values" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new RawModule {
      val a = IO(Output(Property[Seq[SeqMap[String, Seq[Property[Int]]]]]()))
      val b = IO(Output(Property[Seq[SeqMap[String, Property[Seq[Int]]]]]()))
      val c = IO(Output(Property[Seq[Property[SeqMap[String, Seq[Int]]]]]()))
      val d = IO(Output(Property[Property[Seq[SeqMap[String, Seq[Int]]]]]()))
      a := Property(Seq[SeqMap[String, Seq[Int]]](SeqMap("foo" -> Seq(123))))
      b := Property(Seq[SeqMap[String, Seq[Int]]](SeqMap("foo" -> Seq(123))))
      c := Property(Seq[SeqMap[String, Seq[Int]]](SeqMap("foo" -> Seq(123))))
      d := Property(Seq[SeqMap[String, Seq[Int]]](SeqMap("foo" -> Seq(123))))
    })

    assertTypeError {
      "Property[Property[Property[Int]]]()"
    }

    matchesAndOmits(chirrtl)(
      "output a : List<Map<List<Integer>>>",
      "output b : List<Map<List<Integer>>>",
      "output c : List<Map<List<Integer>>>",
      "output d : List<Map<List<Integer>>>",
      """propassign a, List<Map<List<Integer>>>(Map<List<Integer>>("foo" -> List<Integer>(Integer(123))))""",
      """propassign b, List<Map<List<Integer>>>(Map<List<Integer>>("foo" -> List<Integer>(Integer(123))))""",
      """propassign c, List<Map<List<Integer>>>(Map<List<Integer>>("foo" -> List<Integer>(Integer(123))))""",
      """propassign d, List<Map<List<Integer>>>(Map<List<Integer>>("foo" -> List<Integer>(Integer(123))))"""
    )()
  }

  it should "not support types with nested Property[_]" in {
    assertTypeError("Property[Property[Property[Int]]]()")
    assertTypeError("Property[Property[Seq[Property[Int]]]]()")
    assertTypeError("Property[Property[SeqMap[String, Property[Int]]]]()")
    assertTypeError("Property[Property[Seq[Property[Seq[Property[Int]]]]]]()")
    assertTypeError("Property[Property[SeqMap[String, Property[Seq[Property[Int]]]]]]()")
  }
}
