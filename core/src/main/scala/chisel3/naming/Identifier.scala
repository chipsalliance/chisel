package chisel3.naming

trait IdentifierProposer[T] {
  def propose(obj: T): String
}

object IdentifierProposer {

  // Any proposal needs valid characters
  // E.g. `chisel3.internal.Blah@123412` -> `chisel3_internal_Blah`
  // E.g. `(Foo)` -> `Foo`
  // E.g. `Foo(1)` -> `Foo_1`
  def filterProposal(s: String): String = {
    def legalStartOrEnd(c: Char) = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')
    def legal(c:           Char) = legalStartOrEnd(c) || c == '_'
    def terminate(c:       Char) = c == '@'
    var firstOkChar: Int = -1
    var lastOkChar:  Int = 0
    var finalChar:   Int = -1
    for (i <- (0 until s.length)) {
      if (finalChar != -1 && finalChar < i) {} else {
        if (terminate(s(i))) finalChar = i
        else {
          if (legalStartOrEnd(s(i))) {
            lastOkChar = i
            if (firstOkChar == -1) firstOkChar = i
          }
        }
      }
    }
    s.substring(firstOkChar, if (finalChar == -1) lastOkChar + 1 else finalChar).map { x => if (!legal(x)) '_' else x }
  }

  // Summons correct IdentifierProposer to generate a proposal
  def getProposal[T](obj: T)(implicit ip: IdentifierProposer[T]): String = filterProposal(ip.propose(obj))

  // Algorithm to create an identifier proposal derived from a list of proposals
  def makeProposal(proposals: String*): String = proposals.mkString("_")

  // Catch-all proposer is to call toString, maybe use Java reflection instead?
  implicit def proposerAll[T] = new IdentifierProposer[T] {
    def propose(o: T): String = o.toString
  }

  implicit val proposerString = proposerAll[String]
  implicit val proposerInt = proposerAll[Int]

  import scala.language.higherKinds // Required to avoid warning for proposerIterable type parameter
  implicit def proposerIterable[T, F[_] <: Iterable[_]](implicit ip: IdentifierProposer[T]) =
    new IdentifierProposer[F[T]] {
      def propose(o: F[T]): String = makeProposal(o.toList.map { x => ip.propose(x.asInstanceOf[T]) }: _*)
    }

}
