package Chisel

object Literal {
  def sizeof(x: BigInt): Int = x.bitLength

  def decodeBase(base: Char): Int = base match {
    case 'x' | 'h' => 16
    case 'd' => 10
    case 'o' => 8
    case 'b' => 2
    case _ => ChiselError.error("Invalid base " + base); 2
  }

  def stringToVal(base: Char, x: String): BigInt =
    BigInt(x, decodeBase(base))
}
