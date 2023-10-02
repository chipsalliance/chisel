package chisel3.util.experimental.decode

import chisel3._
import chisel3.util.BitPat

import scala.collection.immutable.SeqMap

/**
  * Input pattern a DecoderField should match, e.g. an instruction
  */
trait DecodePattern {
  def bitPat: BitPat
}

/**
  * One output field of a decoder bundle
  *
  * @tparam T pattern this field should match
  * @tparam D output type of this field
  */
trait DecodeField[T <: DecodePattern, D <: Data] {
  def name: String

  def chiselType: D

  def default: BitPat = dc

  final def width: Int = chiselType.getWidth

  def dc: BitPat = BitPat.dontCare(width)

  def genTable(op: T): BitPat

  require(width == default.width)
}

/**
  * Special case when output type is a single Bool
  *
  * @tparam T pattern this field should match
  */
trait BoolDecodeField[T <: DecodePattern] extends DecodeField[T, Bool] {
  def chiselType: Bool = Bool()

  def y: BitPat = BitPat.Y(1)

  def n: BitPat = BitPat.N(1)
}

/**
  * Output of DecoderTable
  * @param fields all fields to be decoded
  */
class DecodeBundle(fields: Seq[DecodeField[_, _ <: Data]]) extends Record {
  require(fields.map(_.name).distinct.size == fields.size, "Field names must be unique")
  val elements: SeqMap[String, Data] = SeqMap(fields.map(k => k.name -> k.chiselType): _*)

  /**
    * Get result of each field in decoding result
    *
    * @param field field to be queried
    * @tparam D type of field
    * @return hardware value of decoded output
    */
  def apply[D <: Data](field: DecodeField[_, D]): D = elements(field.name).asInstanceOf[D]
}

/**
  * A structured way of generating large decode tables, often found in CPU instruction decoders
  *
  * Each field is a `DecoderPattern`, its genTable method will be called for each possible pattern and gives expected
  * output for this field as a `BitPat`.
  *
  * @param patterns all possible input patterns, should implement trait DecoderPattern
  * @param fields   all fields as decoded output
  * @tparam I concrete type of input patterns trait
  */
class DecodeTable[I <: DecodePattern](patterns: Seq[I], fields: Seq[DecodeField[I, _ <: Data]]) {
  require(patterns.map(_.bitPat.getWidth).distinct.size == 1, "All instructions must have the same width")

  def bundle: DecodeBundle = new DecodeBundle(fields)

  val table: TruthTable = TruthTable(
    patterns.map { op => op.bitPat -> fields.reverse.map(field => field.genTable(op)).reduce(_ ## _) },
    fields.reverse.map(_.default).reduce(_ ## _)
  )

  def decode(input: UInt): DecodeBundle = chisel3.util.experimental.decode.decoder(input, table).asTypeOf(bundle)
}
