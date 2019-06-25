// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.experimental.{Analog, DataMirror, RawModule}

import scala.collection.immutable.HashMap
import scala.collection.mutable
import java.io._

class VerifyConnectionProperties extends ChiselPropSpec {
  val test0Width = false  // true to test 0-width wires
  val test0Elements = true  // true to test structures with 0 elements
  val targetDir = "."

  val testObjects = Table(
    "testObject",
    "UInt",
    "SInt",
    "Bool",
//    "BareBundle",
//    "InputBundle",
//    "OutputBundle",
//    "MixedBundle",
//    "NestedBundle",
//    "Vec",
    "DontCare",
    "Analog"
  )

  val partnerObjects = Table(
    "partnerObject",
    "UInt",
    "SInt",
    "Bool",
//    "BareBundle",
//    "InputBundle",
//    "OutputBundle",
//    "MixedBundle",
//    "NestedBundle",
//    "Vec",
    "DontCare",
    "Analog"

  )

  val binding_directions = Table(
    "direction",
    "winput",
    "woutput",
    "wnone",
    "pinput",
    "poutput"
  )

  def connectionOk(l: Data, r: Data): (Int, String) = (0, "")
  def dontCareNotSink(l: Data, r: Data): (Int, String) = (1, "DontCare cannot be a connection sink (LHS)")
  def differentTypes(l: Data, r: Data): (Int, String) = (2, "have different types")
  def illegalAnalogMono(l: Data, r: Data) = (11, "Analog cannot participate in a mono connection (source and sink)")
  def illegalAnalogMonoSink(l: Data, r: Data) = (12, "Analog cannot participate in a mono connection (sink - LHS)")
  def illegalAnalogMonoSource(l: Data, r: Data) = (13, "Analog cannot participate in a mono connection (source - RHS)")
  def differentWidths(l: Data, r: Data): (Int, String) = {
    (l, r) match {
      case (_:Analog, _:Analog) =>
        (11, "Analog cannot participate in a mono connection (source and sink)")
      case _ =>
        if (l.getWidth != r.getWidth) {
          (4, "have different types")
        } else {
          (0, "")
        }
    }
  }
  def locallyUnclearBothInternal(l: Data, r: Data): (Int, String) = (5, "Locally unclear whether Left or Right (both internal)")
  def missingFields(l: Data, r: Data): (Int, String) = (6, "Record missing field")
  def differentLengths(l: Data, r: Data): (Int, String) = {
    (l, r) match {
      case (lv :Vec[_], rv:Vec[_]) =>
        if (lv.length != rv.length)
          (7, "different length Vecs")
        else if (!(l.getClass.isAssignableFrom(r.getClass)))
          (3, "have different types")
        else
          (0, "")
      case _ =>
        if (l.getWidth != r.getWidth) {
          (4, "have different types")
        } else {
          (0, "")
        }
    }
  }
  def differentLengthsOrlocallyUnclearBothInternal(l: Data, r: Data): (Int, String) = {
    (l, r) match {
      case (lv :Vec[_], rv:Vec[_]) =>
        if (lv.length != rv.length)
          (7, "different length Vecs")
        else if (!(l.getClass.isAssignableFrom(r.getClass)))
          (3, "have different types")
        else
          (5, "Locally unclear whether Left or Right (both internal)")
      case _ =>
        if (l.getWidth != r.getWidth) {
          (4, "have different types")
        } else {
          (5, "Locally unclear whether Left or Right (both internal)")
        }
    }
  }
  def locallyUnclearBothInternalOrOKFor0LengthVec(l: Data, r: Data): (Int, String) = {
    (l, r) match {
      case (lv :Vec[_], rv:Vec[_]) =>
        if (lv.length != rv.length)
          (7, "different length Vecs")
        else if (!(l.getClass.isAssignableFrom(r.getClass)))
          (3, "have different types")
        else if (lv.length == 0)
          (0, "")
        else
          (5, "Locally unclear whether Left or Right (both internal)")
      case (lv :Vec[_], DontCare) =>
        if (lv.length == 0)
          (0, "")
        else
          (5, "Locally unclear whether Left or Right (both internal)")
      case (_, rv: Vec[_]) =>
        if (true || rv.length == 0 || l.getWidth != r.getWidth) {
          (4, "have different types")
        } else {
          (5, "Locally unclear whether Left or Right (both internal)")
        }
      case _ =>
        if (l.getWidth != r.getWidth) {
          (4, "have different types")
        } else {
          (5, "Locally unclear whether Left or Right (both internal)")
        }
    }
  }

  val codeMap = HashMap[Int, (String, String)](
    (0, ("", "ok")),
    (1, ("DontCare cannot be a connection sink (LHS)", "")),
    (2, ("have different types", " (basic)")),
    (3, ("have different types", " (!isAssignableFrom)")),
    (4, ("have different types", " (different widths or lengths")),
    (5, ("Locally unclear whether Left or Right (both internal)", "")),
    (6, ("Record missing field", "")),
    (7, ("different length Vecs", "")),
    (8, ("Sink is unwriteable", "")),
    (9, ("Neither Left nor Right is a driver", "")),
    (10, ("Both Left and Right are drivers", "")),
    (11, ("Analog cannot participate in a mono connection (source and sink)", "")),
    (12, ("Analog cannot participate in a mono connection (sink - LHS)", "")),
    (13, ("Analog cannot participate in a mono connection (source - RHS)", ""))
  )
  class CMap {
    val cmap = new mutable.HashMap[String, mutable.HashMap[String, mutable.HashMap[String, Int]]]
    def insert(e: String, l: String, r: String, code: Int): Unit = {
      if (!cmap.contains(e))
        cmap(e) = new mutable.HashMap[String, mutable.HashMap[String, Int]]
      if (!cmap(e).contains(l))
        cmap(e)(l) = new mutable.HashMap[String, Int]
      if (!cmap(e)(l).contains(r))
        cmap(e)(l)(r) = code
      else
        println(s"dup ($e)($l)($r)")
      if (cmap(e)(l)(r) != code) {
        println(s"map($e)($l)($r) ${cmap(e)(l)(r)} != $code")
      } else {
        println(s"map(e)($l)($r) == $code")
      }
    }
    def dump(directoryName: String, fileName: String, tag: String): Unit = {
      for (widthKey <- cmap.keys) {
        val eName = widthKey match {
          case "==" => "eq"
          case "!=" => "ne"
          case _ => widthKey
        }
        val file = new File(directoryName, eName + fileName)
        val bw = new BufferedWriter(new FileWriter(file))
        val emap = cmap(widthKey)
        val sortedKeys = emap.keys.toList.sorted
        val sortedValueKeys = emap.values.flatMap(_.keys).toSet.toList.sorted
        val useTab = true
        val fieldDelimiter = if (useTab) "\t" else ","
        val lineDelimiter = "\n"

        bw.write(s"$tag $widthKey" + fieldDelimiter + sortedValueKeys.mkString(fieldDelimiter) + lineDelimiter)
        for (k <- sortedKeys) {
          val leftMap = emap(k)
          bw.write(k + fieldDelimiter + sortedValueKeys.map(leftMap.getOrElse(_, "")).mkString(fieldDelimiter) + lineDelimiter)
        }
        // Get all the values in these maps
        val values = emap.values.flatMap(_.values).toSet
        // Output the legend for the values appearing in this map
        for (k <- values.toList.sorted) {
          val s = codeMap(k)
          bw.write(lineDelimiter + k + fieldDelimiter + s._1 + s._2 + lineDelimiter)
        }
        bw.close()
      }
    }
  }
  val connectionExceptions = HashMap[String, HashMap[String, HashMap[String, (Data, Data) => (Int, String)]]](
    ":=" -> HashMap(
      "DontCare" -> HashMap(
        "UInt" -> dontCareNotSink,
        "SInt" -> dontCareNotSink,
        "Bool" -> dontCareNotSink,
        "BareBundle" -> dontCareNotSink,
        "InputBundle" -> dontCareNotSink,
        "OutputBundle" -> dontCareNotSink,
        "MixedBundle" -> dontCareNotSink,
        "NestedBundle" -> dontCareNotSink,
        "Vec" -> dontCareNotSink,
        "Analog" -> connectionOk,
        "DontCare" -> connectionOk  // The one exception to the rule that DontCare can't be a sync.
      ),
      "Analog" -> HashMap(
        "UInt" -> illegalAnalogMonoSink,
        "SInt" -> illegalAnalogMonoSink,
        "Bool" -> illegalAnalogMonoSink,
        "BareBundle" -> illegalAnalogMonoSink,
        "InputBundle" -> illegalAnalogMonoSink,
        "OutputBundle" -> illegalAnalogMonoSink,
        "MixedBundle" -> illegalAnalogMonoSink,
        "NestedBundle" -> illegalAnalogMonoSink,
        "Vec" -> illegalAnalogMonoSink,
        "Analog" -> illegalAnalogMono,
        "DontCare" -> connectionOk
      ),
      "UInt" -> HashMap(
        "UInt" -> connectionOk,
        "SInt" -> differentTypes,
        "Bool" -> connectionOk,
        "BareBundle" -> differentTypes,
        "InputBundle" -> differentTypes,
        "OutputBundle" -> differentTypes,
        "MixedBundle" -> differentTypes,
        "NestedBundle" -> differentTypes,
        "Vec" -> differentTypes,
        "Analog" -> illegalAnalogMonoSource,
        "DontCare" -> connectionOk
      ),
      "SInt" -> HashMap(
        "UInt" -> differentTypes,
        "SInt" -> connectionOk,
        "Bool" -> differentTypes,
        "BareBundle" -> differentTypes,
        "InputBundle" -> differentTypes,
        "OutputBundle" -> differentTypes,
        "MixedBundle" -> differentTypes,
        "NestedBundle" -> differentTypes,
        "Vec" -> differentTypes,
        "Analog" -> illegalAnalogMonoSource,
        "DontCare" -> connectionOk
      ),
      "Bool" -> HashMap(
        "UInt" -> connectionOk,
        "SInt" -> differentTypes,
        "Bool" -> connectionOk,
        "BareBundle" -> differentTypes,
        "InputBundle" -> differentTypes,
        "OutputBundle" -> differentTypes,
        "MixedBundle" -> differentTypes,
        "NestedBundle" -> differentTypes,
        "Vec" -> differentTypes,
        "Analog" -> illegalAnalogMonoSource,
        "DontCare" -> connectionOk
      ),
      "BareBundle" -> HashMap(
        "UInt" -> differentTypes,
        "SInt" -> differentTypes,
        "Bool" -> differentTypes,
        "BareBundle" -> connectionOk,
        "InputBundle" -> missingFields,
        "OutputBundle" -> missingFields,
        "MixedBundle" -> missingFields,
        "NestedBundle" -> missingFields,
        "Vec" -> differentTypes,
        "Analog" -> illegalAnalogMonoSource,
        "DontCare" -> connectionOk
      ),
      "InputBundle" -> HashMap(
        "UInt" -> differentTypes,
        "SInt" -> differentTypes,
        "Bool" -> differentTypes,
        "BareBundle" -> missingFields,
        "InputBundle" -> connectionOk,
        "OutputBundle" -> missingFields,
        "MixedBundle" -> missingFields,
        "NestedBundle" -> missingFields,
        "Vec" -> differentTypes,
        "Analog" -> illegalAnalogMonoSource,
        "DontCare" -> connectionOk
      ),
      "OutputBundle" -> HashMap(
        "UInt" -> differentTypes,
        "SInt" -> differentTypes,
        "Bool" -> differentTypes,
        "BareBundle" -> missingFields,
        "InputBundle" -> missingFields,
        "OutputBundle" -> connectionOk,
        "MixedBundle" -> missingFields,
        "NestedBundle" -> missingFields,
        "Vec" -> differentTypes,
        "Analog" -> illegalAnalogMonoSource,
        "DontCare" -> connectionOk
      ),
      "MixedBundle" -> HashMap(
        "UInt" -> differentTypes,
        "SInt" -> differentTypes,
        "Bool" -> differentTypes,
        "BareBundle" -> missingFields,
        "InputBundle" -> missingFields,
        "OutputBundle" -> missingFields,
        "MixedBundle" -> connectionOk,
        "NestedBundle" -> missingFields,
        "Vec" -> differentTypes,
        "Analog" -> illegalAnalogMonoSource,
        "DontCare" -> connectionOk
      ),
      "NestedBundle" -> HashMap(
        "UInt" -> differentTypes,
        "SInt" -> differentTypes,
        "Bool" -> differentTypes,
        "BareBundle" -> missingFields,
        "InputBundle" -> missingFields,
        "OutputBundle" -> missingFields,
        "MixedBundle" -> missingFields,
        "NestedBundle" -> connectionOk,
        "Vec" -> differentTypes,
        "Analog" -> illegalAnalogMonoSource,
        "DontCare" -> connectionOk
      ),
      "Vec" -> HashMap(
        "UInt" -> differentTypes,
        "SInt" -> differentTypes,
        "Bool" -> differentTypes,
        "BareBundle" -> differentTypes,
        "InputBundle" -> differentTypes,
        "OutputBundle" -> differentTypes,
        "MixedBundle" -> differentTypes,
        "NestedBundle" -> differentTypes,
        "Vec" -> differentLengths,
        "Analog" -> illegalAnalogMonoSource,
        "DontCare" -> connectionOk
      )
    ),
    "<>" -> HashMap(
      "DontCare" -> HashMap(
        "UInt" -> dontCareNotSink,
        "SInt" -> dontCareNotSink,
        "Bool" -> dontCareNotSink,
        "BareBundle" -> dontCareNotSink,
        "InputBundle" -> dontCareNotSink,
        "OutputBundle" -> dontCareNotSink,
        "MixedBundle" -> dontCareNotSink,
        "NestedBundle" -> dontCareNotSink,
        "Vec" -> dontCareNotSink,
        "Analog" -> connectionOk,
        "DontCare" -> dontCareNotSink
      ),
      "Analog" -> HashMap(
        "UInt" -> locallyUnclearBothInternal,
        "SInt" -> locallyUnclearBothInternal,
        "Bool" -> locallyUnclearBothInternal,
        "BareBundle" -> differentTypes,
        "InputBundle" -> differentTypes,
        "OutputBundle" -> differentTypes,
        "MixedBundle" -> differentTypes,
        "NestedBundle" -> differentTypes,
        "Vec" -> locallyUnclearBothInternalOrOKFor0LengthVec,
        "Analog" -> connectionOk,
        "DontCare" -> connectionOk
      ),
      "UInt" -> HashMap(
        "UInt" -> locallyUnclearBothInternal,
        "SInt" -> locallyUnclearBothInternal,
        "Bool" -> locallyUnclearBothInternal,
        "BareBundle" -> differentTypes,
        "InputBundle" -> differentTypes,
        "OutputBundle" -> differentTypes,
        "MixedBundle" -> differentTypes,
        "NestedBundle" -> differentTypes,
        "Vec" -> differentTypes,
        "Analog" -> locallyUnclearBothInternal,
        "DontCare" -> locallyUnclearBothInternal
      ),
      "SInt" -> HashMap(
        "UInt" -> locallyUnclearBothInternal,
        "SInt" -> locallyUnclearBothInternal,
        "Bool" -> locallyUnclearBothInternal,
        "BareBundle" -> differentTypes,
        "InputBundle" -> differentTypes,
        "OutputBundle" -> differentTypes,
        "MixedBundle" -> differentTypes,
        "NestedBundle" -> differentTypes,
        "Vec" -> differentTypes,
        "Analog" -> locallyUnclearBothInternal,
        "DontCare" -> locallyUnclearBothInternal
      ),
      "Bool" -> HashMap(
        "UInt" -> locallyUnclearBothInternal,
        "SInt" -> locallyUnclearBothInternal,
        "Bool" -> locallyUnclearBothInternal,
        "BareBundle" -> differentTypes,
        "InputBundle" -> differentTypes,
        "OutputBundle" -> differentTypes,
        "MixedBundle" -> differentTypes,
        "NestedBundle" -> differentTypes,
        "Vec" -> differentTypes,
        "Analog" -> locallyUnclearBothInternal,
        "DontCare" -> locallyUnclearBothInternal
      ),
      "BareBundle" -> HashMap(
        "UInt" -> differentTypes,
        "SInt" -> differentTypes,
        "Bool" -> differentTypes,
        "BareBundle" -> locallyUnclearBothInternal,
        "InputBundle" -> missingFields,
        "OutputBundle" -> missingFields,
        "MixedBundle" -> missingFields,
        "NestedBundle" -> missingFields,
        "Vec" -> differentTypes,
        "Analog" -> differentTypes,
        "DontCare" -> locallyUnclearBothInternal
      ),
      "InputBundle" -> HashMap(
        "UInt" -> differentTypes,
        "SInt" -> differentTypes,
        "Bool" -> differentTypes,
        "BareBundle" -> missingFields,
        "InputBundle" -> locallyUnclearBothInternal,
        "OutputBundle" -> missingFields,
        "MixedBundle" -> missingFields,
        "NestedBundle" -> missingFields,
        "Vec" -> differentTypes,
        "Analog" -> differentTypes,
        "DontCare" -> locallyUnclearBothInternal
      ),
      "OutputBundle" -> HashMap(
        "UInt" -> differentTypes,
        "SInt" -> differentTypes,
        "Bool" -> differentTypes,
        "BareBundle" -> missingFields,
        "InputBundle" -> missingFields,
        "OutputBundle" -> locallyUnclearBothInternal,
        "MixedBundle" -> missingFields,
        "NestedBundle" -> missingFields,
        "Vec" -> differentTypes,
        "Analog" -> differentTypes,
        "DontCare" -> locallyUnclearBothInternal
      ),
      "MixedBundle" -> HashMap(
        "UInt" -> differentTypes,
        "SInt" -> differentTypes,
        "Bool" -> differentTypes,
        "BareBundle" -> missingFields,
        "InputBundle" -> missingFields,
        "OutputBundle" -> missingFields,
        "MixedBundle" -> locallyUnclearBothInternal,
        "NestedBundle" -> missingFields,
        "Vec" -> differentTypes,
        "Analog" -> differentTypes,
        "DontCare" -> locallyUnclearBothInternal
      ),
      "NestedBundle" -> HashMap(
        "UInt" -> differentTypes,
        "SInt" -> differentTypes,
        "Bool" -> differentTypes,
        "BareBundle" -> missingFields,
        "InputBundle" -> missingFields,
        "OutputBundle" -> missingFields,
        "MixedBundle" -> missingFields,
        "NestedBundle" -> locallyUnclearBothInternal,
        "Vec" -> differentTypes,
        "Analog" -> differentTypes,
        "DontCare" -> locallyUnclearBothInternal
      ),
      "Vec" -> HashMap(
        "UInt" -> differentTypes,
        "SInt" -> differentTypes,
        "Bool" -> differentTypes,
        "BareBundle" -> differentTypes,
        "InputBundle" -> differentTypes,
        "OutputBundle" -> differentTypes,
        "MixedBundle" -> differentTypes,
        "NestedBundle" -> differentTypes,
        "Vec" -> locallyUnclearBothInternalOrOKFor0LengthVec,
        "Analog" -> differentTypes,
        "DontCare" -> locallyUnclearBothInternalOrOKFor0LengthVec
      )
    )
  )
  class ConnectModule(val cMap: CMap) extends RawModule {
    def randomWidth = safeUIntWidth.sample.getOrElse(32)
    class BareBundle(val w1: Int, val w2: Int) extends Bundle {
      val uInt = UInt(w1.W)
      val sInt = SInt(w2.W)
    }
    class InputBundle(val w1: Int, val w2: Int) extends Bundle {
      val uIntInput = Input(UInt(w1.W))
      val sIntInput = Input(SInt(w2.W))
    }
    class OutputBundle(val w1: Int, val w2: Int) extends Bundle {
      val uIntOutput = Output(UInt(w1.W))
      val sIntOutput = Output(SInt(w2.W))
    }
    class MixedBundle(val w1: Int, val w2: Int) extends Bundle {
      val uIntInput = Input(UInt(w1.W))
      val sIntOutput = Output(SInt(w2.W))
    }
    class BundleBundle(val w1: Int, val w2: Int) extends Bundle {
      val bareBundle = new BareBundle(w1, w2)
    }
    class IOBundle[T <: Data](v: T) extends Bundle {
      val i = Input(v)
      val o = Output(v)

    }
    class IOModule[T <: Data](element: T) extends RawModule {
      val io = IO(new IOBundle(element))
    }
    class IModule[T <: Data](element: T) extends RawModule {
      val io = IO(Input(element))
    }
    class OModule[T <: Data](element: T) extends RawModule {
      val io = IO(Output(element))
    }
    class UModule[T <: Data](element: T) extends RawModule {
      val io = IO(element)
    }

    def tryConnect(sinkType: String, sourceType: String, connector: String): Unit = {
      // For those objects that have a defined width, we'll generate a pair of objects:
      //  those with the same width (==),
      //  and those with different widths (!=)
      val wc = if (sinkType == "Bool" || sourceType == "Bool") 1 else randomWidth // common width is 1 if any type is Bool
      val w1 = randomWidth
      val w2 = if (wc != w1) wc else w1 + 3
      val nc = 1 + vecSizes.sample.getOrElse(7)
      val n1 = 1 + vecSizes.sample.getOrElse(7)
      val n2 = if (nc != n1) nc else n1 + 1

      /** Return a tuple indicating the parameters required to construct an object of a speficied type:
        *
        * @param typeId a String representing the object type,
        * @return a tuple representing (hasElements, hasWidth, complexity)
        */
      def typeToParams(typeId: String): (Boolean, Boolean, Int) = {
        typeId match {
          case "DontCare" => (false, false, 0)
          case "Analog" => (false, true, 0)
          case "SInt" => (false, true, 1)
          case "Bool" => (false, true, 2)
          case "BareBundle" => (true, true, 3)
          case "InputBundle" => (true, true, 3)
          case "OutputBundle" => (true, true, 3)
          case "MixedBundle" => (true, true, 3)
          case "NestedBundle" => (true, true, 3)
          case "Vec" => (true, true, 3)
          case "UInt" => (false, true, 1)
        }
      }
      def typeToObject(typeId: String, nElements: Int, w1: Int, w2: Int): Data = {
        typeId match {
          case "DontCare" => DontCare
          case "Analog" => Wire(Analog(w1.W)) // We need to wrap this in a Wire() for correct usage.
          case "SInt" => SInt(w1.W)
          case "Bool" => Bool()
          case "BareBundle" => new BareBundle(w1, w2)
          case "InputBundle" => new InputBundle(w1, w2)
          case "OutputBundle" => new OutputBundle(w1, w2)
          case "MixedBundle" => new MixedBundle(w1, w2)
          case "NestedBundle" => new BundleBundle(w1, w2)
          case "Vec" => Vec(nElements, UInt(w1.W))
          case _ => UInt(w1.W)
        }
      }
      def connect(l: Data, r: Data): Unit = {
        connector match {
          case ":=" => l := r
          case "<>" => l <> r
        }
      }
      /** Given a sequence of undirectioned objects, return a sequence with those objects wrapped in directioned (and not) wires.
        *
        * @param ss sequence of bare chisel objects
        * @return sequence of Wire() wrapped, directioned objects
        */
      def addDirections(ss: Seq[Data]): Seq[Data] = {
        val list = new scala.collection.mutable.ListBuffer[Data]
        for (s <- ss) {
          forAll(binding_directions) { sinkDirection =>
            sinkDirection match {
              case "winput" => list += Wire(Input(s))
              case "woutput" => list += Wire(Output(s))
              case "wnone" => list += Wire(s)
              case "pinput" => list += Module(new IModule(s)).io
              case "poutput" =>list += Module(new OModule(s)).io
              case "pnone" => list += Module(new UModule(s)).io
            }
          }
        }
        list.toList
      }
      // We'll generate test connections based on the possible sink and source parameters.
      val (sinkHasElements, sinkHasWidth, sinkComplexity) = typeToParams(sinkType)
      val (sourceHasElements, sourceHasWidth, sourceComplexity) = typeToParams(sourceType)
      val widths = new scala.collection.mutable.ListBuffer[(Array[Data] => Int, Array[Data] => Int)]
      // Some functions to generate widths, either from constants, or the width of another object.
      // We'll create curried versions of these and store them as tuples in our widths list.
      def iw(w: Int)(a: Array[Data]) = w
      def ow_eq(i: Int)(a: Array[Data]) = a(i).getWidth
      def ow_ne(i: Int)(a: Array[Data]) = a(i).getWidth + 1
      val elements = new scala.collection.mutable.ListBuffer[(Int, Int)]
      val objectTypes = new scala.collection.mutable.ListBuffer[(String, Int)]
      // We need to determine which of the two objects (source or sink) is the "dependent" object.
      // This is required for those tests where we want some sort of object equivalence, and one of the objects
      //  is dependent on the parameters used to construct the other.
      // Sink (left) is by convention object 0. Source (right) is objct 1.
      // We'll construct the more complicated (independent) object first, then use its parameters to construct the second.
      if (sourceComplexity > sinkComplexity) {
        objectTypes += ((sourceType, 1))
        objectTypes += ((sinkType, 0))
      } else {
        objectTypes += ((sinkType, 0))
        objectTypes += ((sourceType, 1))
      }
      // Generate curried width functions.
      def genWidthFuncs(w1: Int, w2: Int): (Array[Data] => Int, Array[Data] => Int) = {
        if (w1 == w2)
          if (sourceComplexity == sinkComplexity)
            (iw(w1), iw(w2))  // independent widths
          else if (sourceComplexity > sinkComplexity)
            (ow_eq(1), iw(w2)) // sink width should be equal to source (other) width
          else
            (iw(w1), ow_eq(0))  // source width should be equal to sink (other) width
        else
          if (sourceComplexity == sinkComplexity)
            (iw(w1), iw(w2))  // independent widths
          else if (sourceComplexity > sinkComplexity)
            (ow_ne(1), iw(w2)) // sink width should be NOT equal to source (other) width
          else
            (iw(w1), ow_ne(0))  // source width should be NOT equal to sink (other) width
      }
      if (sinkHasWidth && sourceHasWidth) {
        // Generate two connection tests: one with equal widths, one with unequal widths
        widths += genWidthFuncs(w1, w2)
        widths += genWidthFuncs(wc, wc)
        if (test0Width) {
          widths += genWidthFuncs(w1, 0)
          widths += genWidthFuncs(0, w2)
          widths += genWidthFuncs(0, 0)
        }
      } else if (sinkHasWidth) {
        widths += genWidthFuncs(w1, 0)
        if (test0Width) {
          widths += genWidthFuncs(0, 0)
        }
      } else if (sourceHasWidth) {
        widths += genWidthFuncs(0, w2)
        if (test0Width) {
          widths += genWidthFuncs(0, 0)
        }
      } else {
        widths += genWidthFuncs(0, 0)
      }
      if (sinkHasElements && sourceHasElements) {
        // Generate two connection tests: one with equal lengths, one with unequal lengths
        elements += ((n1, n2))
        elements += ((nc, nc))
        if (test0Elements) {
          elements += ((n1, 0))
          elements += ((0, n2))
          elements += ((0, 0))
        }
      } else if (sinkHasElements) {
        elements += ((n1, 0))
        if (test0Elements) {
          elements += ((0, 0))
        }
      } else if (sourceHasElements) {
        elements += ((0, n2))
        if (test0Elements) {
          elements += ((0, 0))
        }
      } else {
        elements += ((0, 0))
      }
      case class Connection(val sinkType: String, val sinkObject: Data, val sourceType: String, val sourceObject: Data)
      val connections = new scala.collection.mutable.ListBuffer[Connection]
      for (elementTuple <- elements) {
        val elements = Array(elementTuple._1, elementTuple._2)
        for (widthTuple <- widths) {
          val widths = Array(widthTuple._1, widthTuple._2)
          val objects = new Array[Data](2)
          for ((t, i) <- objectTypes) {
            objects(i) = typeToObject(t, elements(i), widths(i)(objects), widths(i)(objects))
          }
          val sink = objects(0)
          val source = objects(1)

          val lefts = sinkType match {
            case "DontCare" | "Analog" => Seq(sink)
            case _ => addDirections(Seq(sink))
          }
          val rights = sourceType match {
            case "DontCare" | "Analog" => Seq(source)
            case _ => addDirections(Seq(source))
          }
          for (left <- lefts) {
            for (right <- rights) {
              connections += Connection(sinkType, left, sourceType, right)
            }
          }
        }
      }

      for (c <- connections) {
        val left = c.sinkObject
        val right = c.sourceObject
        val sinkType = c.sinkType
        val sourceType = c.sourceType
        println(s"connecting $left:${DataMirror.specifiedDirectionOf(left).toString} $connector $right::${DataMirror.specifiedDirectionOf(right).toString}")
        // Are we expecting an exception for this combination?
        val widthMark = (left, right) match {
          case (lv: Vec[_], rv: Vec[_]) => (lv.length, rv.length) match {
            case (0, 0) => "zz"
            case (0, _) => "zx"
            case (_, 0) => "xz"
            case (_, _) => if (lv.length == rv.length) "==" else "!="
          }
          case (DontCare, _) => "dc"
          case (_, DontCare) => "dc"
          case (v: Vec[_], _) => if (v.length == 0) "zx" else "!="
          case (_, v: Vec[_]) => if (v.length == 0) "xz" else "!="
          case (_, _) => if (sinkHasWidth && sourceHasWidth && left.getWidth == right.getWidth) "==" else "!="
        }
        def objectToBindingDirection(o: Data): (Char, Char) = {
          val binding = if (o.toString.contains("(IO")) 'P' else 'W'
          val direction = DataMirror.specifiedDirectionOf(o).toString.charAt(0)
          (binding, direction)
        }
        val (leftBinding, leftDirection) = objectToBindingDirection(left)
        val (rightBinding, rightDirection) = objectToBindingDirection(right)
        val lCell = sinkType.filter(_.isUpper) + "_" + leftDirection + leftBinding
        val rCell = sourceType.filter(_.isUpper) + "_" + rightDirection + rightBinding
        // Get the possible port-independent exception
        val (tid, _) = connectionExceptions(connector)(sinkType)(sourceType)(left, right)
        val id = (tid, leftBinding, leftDirection, rightBinding, rightDirection) match {
          case (0, 'P', 'I', 'P', 'I') => if (connector == ":=") 0 else 9
          case (5, 'P', 'I', 'P', 'I') => if (connector == ":=") 0 else 9
          case (5, 'P', 'O', 'P', 'O') => 10
          case (0, 'P', 'O', _, _) if sourceType != "DontCare" => 8
          case (5, _, _, 'P', _) => 0
          case (5, 'P', _, _, _) => 0
          case _ => tid
        }
        val expectedMessage = codeMap(id)._1
        cMap.insert(widthMark, lCell, rCell, id)
        assert(cMap.cmap(widthMark)(lCell)(rCell) == id)
        if (expectedMessage == "") {
          try {
            connect(left, right)
          } catch {
            case e: Throwable =>
              println("Bang!")
          }
        } else {
          val e = intercept[ChiselException] {
            println("expecting: " + expectedMessage)
            connect(left, right)
          }
          e.getMessage should include (expectedMessage)
        }
      }
    }
  }

  property("objects should mono-connect correctly") {
    elaborate(new RawModule {
      val cMap = new CMap
      val module = Module(new ConnectModule(cMap))
      val connector = ":="
      forAll(testObjects) { testObject =>
        forAll(partnerObjects) { partnerObject =>
          module.tryConnect(testObject, partnerObject, connector)
          module.tryConnect(partnerObject, testObject, connector)
        }
      }
      cMap.dump(targetDir, "connections.mono.txt", connector)
    })
  }

  property("objects should bulk-connect correctly") {
    elaborate(new RawModule {
      val cMap = new CMap
      val module = Module(new ConnectModule(cMap))
      val connector = "<>"
      forAll(testObjects) { testObject =>
        forAll(partnerObjects) { partnerObject =>
          module.tryConnect(testObject, partnerObject, connector)
          module.tryConnect(partnerObject, testObject, connector)
        }
      }
      cMap.dump(targetDir, "connections.bi.txt", connector)
    })
  }
}
