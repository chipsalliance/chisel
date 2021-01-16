package chisel3.util.experimental.decoder

import chisel3._
import chisel3.util.{BitPat, Cat}

import scala.annotation.tailrec
import scala.collection.mutable

object QMCDecoder {
  /** Construct a [[QMCDecoder]].
    *
    * @return A [[QMCDecoder]]
    */
  def apply(): QMCDecoder = new QMCDecoder()

  /** decoder cache during a chisel elaboration. */
  private val caches: mutable.Map[UInt, mutable.Map[Term, Bool]] = mutable.Map[UInt, mutable.Map[Term, Bool]]()
}

class QMCDecoder extends Decoder {
  /** Simplify a multi-input multi-output logic function given by the truth table `mapping`, with function output values
    * on unspecified inputs treated as `default`, and return the output values of the function on input values `addr`.
    *
    * Each bit of `addr` and `mapping[]._1` means one 1-bit input variable of the logic function, and each bit of
    * `default` and `mapping[]_2` represents one 1-bit output value of the function.
    *
    * @param addr     Input values
    * @param default  Default output values, can have don't cares
    * @param mapping  Truth table, can have don't cares in both inputs and outputs, specified as [(inputs, outputs), ...]
    * @return         Output values of the logic function when inputs are `addr`
    */
  def decode(addr: UInt, default: BitPat, mapping: Iterable[(BitPat, BitPat)]): UInt = {
    /** Construct a 1-bit output value out of given [[Seq]] of minterms by ORing them together, equity check between
      * inputs and one minterm is only calculated once using cache mechanism.
      *
      * @param addr       Input values
      * @param addrWidth  Number of 1-bit input values
      * @param cache      Cache of already constructed outputs on a given term
      * @param terms      List of minterms
      * @return           1 bit output value
      */
    def logic(addr: UInt, addrWidth: Int, cache: mutable.Map[Term, Bool], terms: Seq[Term]): Bool = {
      terms.map { t =>
        cache
          .getOrElseUpdate(
            t, (
              if (t.mask == 0)
                addr
              else {
                /** {{{BigInt(2).pow(addrWidth) - (t.mask + 1)}}} calculates the `addrWidth` wide inversion of `t.mask` */
                addr & (BigInt(2).pow(addrWidth) - (t.mask + 1)).U(addrWidth.W)
              }
              ) === t.value.U(addrWidth.W)  // equity check effectively ANDed all inputs together
          )
      }.foldLeft(false.B)(_ || _)  // produce the final SOP, Sum of Products output
    }

    /** Construct a [[Term]] out of a given [[BitPat]].
      *
      * @param lit  [[BitPat]] literature
      * @return     A [[Term]] representing the same value and mask as `lit`
      */
    def term(lit: BitPat) = new Term(lit.value, BigInt(2).pow(lit.getWidth) - (lit.mask + 1))

    /** Calculate essential prime implicants based on previously calculated prime implicants and all implicants.
      *
      * @param prime    Prime implicants
      * @param minterms All implicants
      * @return (a, b, c)
      *         a: essential prime implicants
      *         b: nonessential prime implicants
      *         c: implicants that are not cover by any of the essential prime implicants
      */
    def getEssentialPrimeImplicants(prime: Seq[Term], minterms: Seq[Term]): (Seq[Term], Seq[Term], Seq[Term]) = {
      // primeCovers(i): implicants that `prime(i)` covers
      val primeCovers = prime.map(p => minterms.filter(p.covers))
      // eliminate prime implicants that can be covered by other prime implicants
      for (((icover, pi), i) <- (primeCovers zip prime).zipWithIndex) {
        for (((jcover, pj), _) <- (primeCovers zip prime).zipWithIndex.drop(i + 1)) {
          // we prefer prime implicants with wider implicants coverage
          if (icover.size > jcover.size && jcover.forall(pi.covers)) {
            // calculate essential prime implicants with `pj` eliminated from prime implicants table
            return getEssentialPrimeImplicants(prime.filter(_ != pj), minterms)
          }
        }
      }

      // implicants that only one prime implicant covers
      val essentiallyCovered = minterms.filter(t => prime.count(_.covers(t)) == 1)
      // essential prime implicants, prime implicants that covers only one implicant
      val essential = prime.filter(p => essentiallyCovered.exists(p.covers))
      // {nonessential} = {prime implicants} - {essential prime implicants}
      val nonessential = prime.filterNot(essential contains _)
      // implicants that no essential prime implicants covers
      val uncovered = minterms.filterNot(t => essential.exists(_.covers(t)))
      if (essential.isEmpty || uncovered.isEmpty)
        (essential, nonessential, uncovered)
      else {
        // now there are implicants (`uncovered`) that are covered by multiple nonessential prime implicants (`nonessential`)
        // need to reduce prime implicants
        val (a, b, c) = getEssentialPrimeImplicants(nonessential, uncovered)
        (essential ++ a, b, c)
      }
    }

    /** Use Petrick's method to select a [[Seq]] of nonessential prime implicants that covers all implicants that are not
      * covered by essential prime implicants.
      *
      * @param implicants Nonessential prime implicants
      * @param minterms   Implicants that are not covered by essential prime implicants
      * @param bits       Number of input variables
      * @return           Selected nonessential prime implicants
      */
    def getCover(implicants: Seq[Term], minterms: Seq[Term], bits: Int): Seq[Term] = {
      /** Calculate the implementation cost (using comparators) of a list of implicants, more don't cares is cheaper
        * @param cover  Implicant list
        * @param bits   Number of input variables
        * @return       How many comparators need to implement this list of implicants
        */
      def getCost(cover: Seq[Term], bits: Int) = cover.map(bits - _.mask.bitCount).sum

      /** Determine if one combination of prime implicants is cheaper when implementing as comparators.
        * Shorter term list is cheaper, term list with more don't cares is cheaper (less comparators)
        *
        * @param a    Operand a
        * @param b    Operand b
        * @param bits Number of input values
        * @return
        */
      def cheaper(a: List[Term], b: List[Term], bits: Int) = {
        val ca = getCost(a, bits)
        val cb = getCost(b, bits)

        /** If `a` < `b`
          *
          * Like comparing the dictionary order of two strings.
          * Define `a` < `b` if both `a` and `b` are empty.
          * @param a Operand a
          * @param b Operand b
          * @return `a` < `b`
          */
        @tailrec
        def listLess(a: List[Term], b: List[Term]): Boolean = b.nonEmpty && (a.isEmpty || a.head < b.head || a.head == b.head && listLess(a.tail, b.tail))

        ca < cb || ca == cb && listLess(a.sortWith(_ < _), b.sortWith(_ < _))
      }

      // if there are no implicant that is not covered by essential prime implicants, which means all implicants are
      // covered by essential prime implicants, so no need to apply Petrick's method
      if (minterms.nonEmpty) {
        // cover(i): nonessential prime implicants that covers `minterms(i)`
        val cover = minterms.map(m => implicants.filter(_.covers(m)))
        // apply [[Petrick's method https://en.wikipedia.org/wiki/Petrick%27s_method]]
        val all = cover.tail.foldLeft(cover.head.map(Set(_)))((c0, c1) => c0.flatMap(a => c1.map(a + _)))
        all.map(_.toList).reduceLeft((a, b) => if (cheaper(a, b, bits)) a else b)
      } else
        Seq[Term]()
    }

    /** Check all implicants are covered, and no implicants from the forbidden list are intersected with.
      *
      * @param cover    Cover set need to be checked
      * @param minterms Implicants must be covered
      * @param maxterms Implicants must not be intersected with
      */
    def verify(cover: Seq[Term], minterms: Seq[Term], maxterms: Seq[Term]): Unit = {
      assert(minterms.forall(t => cover.exists(_.covers(t))))
      assert(maxterms.forall(t => !cover.exists(_ intersects t)))
    }

    /** Same as [[simplify]] but implicants that are not specified by `minterms` or `dontcares` make the output either
      * `0` or `1`.
      *
      * @param minterms Each one of these implicants makes output to be `1`
      * @param maxterms Each one of these implicants makes output to be `0`
      * @param bits     Number of input values
      * @return Minimal (measured by implementation cost) set of implicants that each of them will result a `1` as output
      */
    def simplifyDC(minterms: Seq[Term], maxterms: Seq[Term], bits: Int): Seq[Term] = {
      /** Get prime implicants from all implicants.
        * Automatically infer don't cares from ({all possible inputs} - {minterm} - {maxterm}) for implicants merging
        *
        * @param minterms All implicants
        * @param maxterms Forbidden list of inferring
        * @param bits     Number of 1-bit input values
        * @return         Prime implicants
        */
      def getPrimeImplicants(minterms: Seq[Term], maxterms: Seq[Term], bits: Int): Seq[Term] = {
        /** Search for implicit don't cares of `term`. The new implicant must NOT intersect with any of the implicants from `maxterm`.
          *
          * @param maxterms The forbidden list of searching
          * @param term     The implicant we want to search implicit dc for
          * @param bits     Number of input values
          * @param above    Are we searching for implicants with one more `1` in value than `term`? (or search for implicants with one less `1`)
          * @return         The implicants that we found or `null`
          */
        def getImplicitDC(maxterms: Seq[Term], term: Term, bits: Int, above: Boolean): Term = {
          // foreach input values in implicant `term`
          for (i <- 0 until bits) {
            var t: Term = null
            if (above && ((term.value | term.mask) & (BigInt(1) << i)) == 0)  // the i-th input of this implicant is cared and is `0`
              t = new Term(term.value | (BigInt(1) << i), term.mask)  // generate a new implicant with i-th input being `1` and other inputs the same as `term`
            else if (!above && (term.value & (BigInt(1) << i)) != 0)  // the i-th input of this implicant is cared and is `1`
              t = new Term(term.value & ~(BigInt(1) << i), term.mask)  // generate a new implicant with i-th input being `0` and other inputs the same as `term`
            if (t != null && !maxterms.exists(_.intersects(t)))  // make sure we are not using one implicant from the forbidden list
              return t
          }
          null
        }

        // container for prime implicants
        var prime = List[Term]()
        // first set all implicants as prime
        minterms.foreach(_.prime = true)
        val mint = minterms.map(t => new Term(t.value, t.mask))
        val cols = (0 to bits).map(b => mint.filter(b == _.mask.bitCount))
        // table(i)(j) : implicants with `i` '1's in mask and `j` '1's in value, 0 <= i <= bits, 0 <= j <= bits
        val table = cols.map(c => (0 to bits).map(b => mutable.Set(c.filter(b == _.value.bitCount): _*)))

        for (i <- 0 to bits) {
          for (j <- 0 until bits - i) {
            table(i)(j).foreach(a => table(i + 1)(j) ++= table(i)(j + 1).filter(_ similar a).map(_ merge a))
          }
          for (j <- 0 until bits - i) {
            for (a <- table(i)(j).filter(_.prime)) {
              // if we found a dc, it will have `i` '1's in mask and `j + 1` '1's in value
              val dc = getImplicitDC(maxterms, a, bits, above = true)
              if (dc != null) {
                // merge the new dc
                table(i + 1)(j) += dc merge a
              }
            }
            for (a <- table(i)(j + 1).filter(_.prime)) {
              // if we found a dc, it will have `i` '1's in mask and `j` '1's in value
              val dc = getImplicitDC(maxterms, a, bits, above = false)
              if (dc != null)
                // merge the new dc
                table(i + 1)(j) += a merge dc
            }
          }
          for (r <- table(i))
            for (p <- r; if p.prime)  // if an implicants cannot be merged with others, then it's a prime implicant
              prime = p :: prime  // collect all prime implicants
        }
        prime.sortWith(_ < _)
      }

      // the same as [[simplify]]
      val prime = getPrimeImplicants(minterms, maxterms, bits)
      val (eprime, prime2, uncovered) = getEssentialPrimeImplicants(prime, minterms)
      val cover = eprime ++ getCover(prime2, uncovered, bits)
      verify(cover, minterms, maxterms)  // sanity check, now we should get all implicants covered, and did not violate the forbidden list
      cover
    }

    /** Simplify implicants so that returned implicants make the output value to be `1` and use the least comparators to
      * implement. Implicants that are not specified by `minterms` or `dontcares` make the output a `0`.
      *
      * @param minterms   Each one of these implicants makes output to be `1`
      * @param dontcares  Each one of these implicants makes output to be either `0` or `1`
      * @param bits       Number of input values
      * @return           Minimal (measured by implementation cost) set of implicants that each of them will result a `1` as output
      */
    def simplify(minterms: Seq[Term], dontcares: Seq[Term], bits: Int): Seq[Term] = {
      /** Get prime implicants from all implicants.
        *
        * @param implicants All implicants
        * @param bits       Number of 1-bit input values
        * @return           Prime implicants
        */
      def getPrimeImplicants(implicants: Seq[Term], bits: Int): Seq[Term] = {
        // container for prime implicants
        var prime = List[Term]()
        // first set all implicants as prime
        implicants.foreach(_.prime = true)

        /** Implicants grouped by mask '1' count
          * [
          *   [implicants with 0 '1' in mask], [implicants with 1 '1' in mask], [implicants with 2 '1's in mask], ...
          * ]
          *
          * col(i) : implicants with `i` '1's in mask, 0 <= i <= bits
          */
        val cols = (0 to bits).map(b => implicants.filter(b == _.mask.bitCount))

        /** Implicants grouped by mask '1' count and value '1' count
          * [
          *   [ [implicants with 0 '1' in mask and 0 '1' in value], [implicants with 0 '1' in mask and 1 '1' in value], ...],
          *   [ [implicants with 1 '1' in mask and 0 '1' in value], [implicants with 1 '1' in mask and 1 '1' in value], ...],
          *   ...,
          * ]
          *
          * table(i)(j) : implicants with `i` '1's in mask and `j` '1's in value, 0 <= i <= bits, 0 <= j <= bits
          */
        val table = cols.map(c => (0 to bits).map(b => mutable.Set(c.filter(b == _.value.bitCount): _*)))

        for (i <- 0 to bits) {  // 0 <= i <= bits, mask bit count
          for (j <- 0 until bits-i) {  // 0 <= j < bits - i, because i + j <= bits, and for implicants that i + j == bits, they cannot be merged with other implicants
            // for each implicants, merge similar implicants, similar implicants can only be found in table(i+1)(j) or table(i)(j+1)
            table(i)(j).foreach(a => table(i+1)(j) ++= table(i)(j+1).filter(_.similar(a)).map(_.merge(a)))
          }
          for (r <- table(i))
            for (p <- r; if p.prime)  // if an implicants cannot be merged with others, then it's a prime implicant
              prime = p :: prime  // collect all prime implicants
        }
        prime.sortWith(_ < _)
      }

      if (dontcares.isEmpty) {
        // As an elaboration performance optimization, don't be too clever if
        // there are no don't-cares; synthesis can figure it out.
        minterms
      } else {
        val prime = getPrimeImplicants(minterms ++ dontcares, bits)
        // make sure prime implicants cover all implicants
        minterms.foreach(t => assert(prime.exists(_.covers(t))))
        val (eprime, prime2, uncovered) = getEssentialPrimeImplicants(prime, minterms)
        // select nonessential prime implicants to cover the left uncovered implicants
        val cover = eprime ++ getCover(prime2, uncovered, bits)
        minterms.foreach(t => assert(cover.exists(_.covers(t)))) // sanity check, now we should get all implicants covered
        cover
      }
    }

    // get existing cache or create an empty one
    val cache = QMCDecoder.caches.getOrElseUpdate(addr, mutable.Map[Term, Bool]())
    val defaultTerm = term(default)
    // keys - inputs, values - outputs
    val (keys, values) = mapping.unzip
    // Number of inputs
    val addrWidth = keys.map(_.getWidth).max
    // input Terms
    val terms = keys.toList.map(k => term(k))
    // (inputTerm, outputTerm) pairs
    val termvalues = terms.zip(values.toList.map(term))

    // Check no two input patterns specified in the truth table intersect
    for (t <- keys.zip(terms).tails; if t.nonEmpty)
      for (u <- t.tail)
        assert(
          !t.head._2.intersects(u._2),
          "DecodeLogic: keys " + t.head + " and " + u + " overlap"
        )

    Cat(
      // Foreach 1-bit value in outputs
      (0 until default.getWidth.max(values.map(_.getWidth).max))
        .map({ i: Int =>
          // Min terms, implicants that makes the output to be 1
          // k - term, t - value
          val mint: Seq[Term] =
            termvalues.filter { case (_, t) => ((t.mask >> i) & 1) == 0 && ((t.value >> i) & 1) == 1 }.map(_._1)
          // Max terms, implicants that makes the output to be 0
          val maxt: Seq[Term] =
            termvalues.filter { case (_, t) => ((t.mask >> i) & 1) == 0 && ((t.value >> i) & 1) == 0 }.map(_._1)
          // Don't cares, implicants that can produce either 0 or 1 as output
          val dc: Seq[Term] = termvalues.filter { case (_, t) => ((t.mask >> i) & 1) == 1 }.map(_._1)

          // This bit is default to don't care
          if (((defaultTerm.mask >> i) & 1) != 0) {
            // Use simplifyDC method
            logic(addr, addrWidth, cache, simplifyDC(mint, maxt, addrWidth))
          } else {
            // Default output of this bit
            val defbit = (defaultTerm.value >> i) & 1
            // 0 -> mint, 1 -> maxt, the following lines are used to make sure we produce best result when truth table are not
            // fully specified.
            val t = if (defbit == 0) mint else maxt
            val bit = logic(addr, addrWidth, cache, simplify(t, dc, addrWidth))
            // Because `simplify` method in previous line expect mint (ORed together produce `1` as output), but when
            // `defbit == 1`, we actually provided `simplify` maxt, which would compose the "inverted" version of the
            // truth table, so we need to invert one more time to produce the correct output.
            if (defbit == 0) bit else ~bit
          }
        })
        .reverse
    )
  }
}