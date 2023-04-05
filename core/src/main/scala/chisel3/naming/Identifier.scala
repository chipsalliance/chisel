package chisel3.naming

trait IdentifierProposer[T] {
  def propose(obj: T): String
}

object IdentifierProposer {

  // Any proposal needs valid characters
  def filterProposal(s: String): String = {
    s.split('@')
      .head
      .replaceAll("\\s", "")
      .replaceAll("[^a-zA-Z0-9]", "_")
      .dropWhile(_ == '_')
      .reverse
      .dropWhile(_ == '_')
      .reverse
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
