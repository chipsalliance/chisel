package Chisel
import scala.math.log
import scala.math.abs
import scala.math.ceil
import scala.math.max
import scala.math.min
import Literal._
import ChiselError._

/* Factory for literal values to be used by Bits and SInt factories. */
object Lit {
  def apply[T <: Bits](n: String)(gen: => T): T = {
    makeLit(Literal(n, -1))(gen)
  }

  def apply[T <: Bits](n: String, width: Int)(gen: => T): T = {
    makeLit(Literal(n, width))(gen)
  }

  def apply[T <: Bits](n: String, base: Char)(gen: => T): T = {
    makeLit(Literal(-1, base, n))(gen)
  }

  def apply[T <: Bits](n: String, base: Char, width: Int)(gen: => T): T = {
    makeLit(Literal(width, base, n))(gen)
  }

  def apply[T <: Bits](n: BigInt)(gen: => T): T = {
    makeLit(Literal(n, signed = gen.isInstanceOf[SInt]))(gen)
  }

  def apply[T <: Bits](n: BigInt, width: Int)(gen: => T): T = {
    val lit = Literal(n, width, signed = gen.isInstanceOf[SInt])
    makeLit(lit)(gen)
  }

  def makeLit[T <: Bits](x: Literal)(gen: => T): T = {
    gen.makeLit(x.value, x.width)
  }
}

class Literal(val value: BigInt, val width: Int) { }

object Literal {

  private def bigMax(x: BigInt, y: BigInt): BigInt = if (x > y) x else y;
  def sizeof(x: BigInt): Int = {
    val y = bigMax(BigInt(1), x.abs).toDouble;
    val res = max(1, (ceil(log(y + 1)/log(2.0))).toInt);
    res
   }

  private def sizeof(base: Char, x: String): Int = {
    var res = 0;
    var first = true;
    val size =
      if(base == 'b') {
        1
      } else if(base == 'h') {
        4
      } else if(base == 'o') {
        3
      } else {
        -1
      }
    for(c <- x)
      if (c == '_') {

      } else if(first) {
        first = false;
        res += sizeof(c.asDigit);
      } else if (c != '_') {
        res += size;
      }
    res
  }
  val hexNibbles = "0123456789abcdef";
  def toHexNibble(x: String, off: Int): Char = {
    var res = 0;
    for (i <- 0 until 4) {
      val idx = off + i;
      val c   = if (idx < 0) '0' else x(idx);
      res     = 2 * res + (if (c == '1') 1 else 0);
    }
    hexNibbles(res)
  }
  val pads = Vector(0, 3, 2, 1);
  def toHex(x: String): String = {
    var res = "";
    val numNibbles = (x.length-1) / 4 + 1;
    val pad = pads(x.length % 4);
    for (i <- 0 until numNibbles) {
      res += toHexNibble(x, i*4 - pad);
    }
    res
  }
  def toLitVal(x: String): BigInt = {
    BigInt(x.substring(2, x.length), 16)
  }

  def toLitVal(x: String, shamt: Int): BigInt = {
    var res = BigInt(0);
    for(c <- x)
      if(c != '_'){
        if(!(hexNibbles + "?").contains(c.toLower)) ChiselError.error({"Literal: " + x + " contains illegal character: " + c});
        res = res * shamt + c.asDigit;
      }
    res
  }

  def removeUnderscore(x: String): String = {
    var res = ""
    for(c <- x){
      if(c != '_'){
        res = res + c
      }
    }
    res
  }

  def parseLit(x: String): (String, String, Int) = {
    var bits = "";
    var mask = "";
    var width = 0;
    for (d <- x) {
      if (d != '_') {
        if(!"01?".contains(d)) ChiselError.error({"Literal: " + x + " contains illegal character: " + d});
        width += 1;
        mask   = mask + (if (d == '?') "0" else "1");
        bits   = bits + (if (d == '?') "0" else d.toString);
      }
    }
    (bits, mask, width)
  }
  def stringToVal(base: Char, x: String): BigInt = {
    if(base == 'x') {
      toLitVal(x, 16);
    } else if(base == 'd') {
      BigInt(x.toInt)
    } else if(base == 'h') {
      toLitVal(x, 16)
    } else if(base == 'b') {
      toLitVal(x, 2)
    } else if(base == 'o') {
      toLitVal(x, 8)
    } else {
      BigInt(-1)
    }
  }

  /** Derive the bit length for a Literal
   *  
   */
  def bitLength(b: BigInt): Int = {
    // Check for signedness
    // We have seen unexpected values (one too small) when using .bitLength on negative BigInts,
    // so use the positive value instead.
    val usePositiveValueForBitLength = false
    (if (usePositiveValueForBitLength && b < 0) {
      -b
    } else {
      b
    }).bitLength
  }
  /** Creates a *Literal* instance from a scala integer.
    */
  def apply(x: BigInt, width: Int = -1, signed: Boolean = false): Literal = {
    // Check for signedness
    // We get unexpected values (one too small) when using .bitLength on negative BigInts,
    // so use the positive value instead.
    val bl = bitLength(x)
    val xWidth = if (signed) {
      bl + 1
    } else {
      max(bl, 1)
    }
    val w = if(width == -1) xWidth else width
    val xString = (if (x >= 0) x else (BigInt(1) << w) + x).toString(16)

    if(xWidth > width && width != -1) {
      // Is this a zero-width wire with value 0
      if (!(x == 0 && width == 0 && Driver.isSupportW0W)) {
        ChiselError.error({"width " + width + " is too small for literal " + x + ". Smallest allowed width is " + xWidth});
      }
    }
    apply("h" + xString, w)
  }
  def apply(n: String, width: Int): Literal =
    apply(width, n(0), n.substring(1, n.length));

  def apply(width: Int, base: Char, literal: String): Literal = {
    if (!"dhbo".contains(base)) {
      ChiselError.error("no base specified");
    }
    new Literal(stringToVal(base, literal), width)
  }
}


