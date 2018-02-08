// See LICENSE for license details.

package firrtl.graph

import scala.collection.mutable

/** Euler Tour companion object */
object EulerTour {
  /** Create an Euler Tour of a `DiGraph[T]` */
  def apply[T](diGraph: DiGraph[T], start: T): EulerTour[Seq[T]] = {
    val r = mutable.Map[Seq[T], Int]()
    val e = mutable.ArrayBuffer[Seq[T]]()
    val h = mutable.ArrayBuffer[Int]()

    def tour(u: T, parent: Vector[T], height: Int): Unit = {
      val id = parent :+ u
      r.getOrElseUpdate(id, e.size)
      e += id
      h += height
      diGraph.getEdges(id.last).foreach { v =>
        tour(v, id, height + 1)
        e += id
        h += height
      }
    }

    tour(start, Vector.empty, 0)
    new EulerTour(r.toMap, e, h)
  }
}

/** A class that represents an Euler Tour of a directed graph from a
  * given root. This requires `O(n)` preprocessing time to generate
  * the initial Euler Tour.
  *
  * @constructor Create a new EulerTour from the specified data
  * @param r A map of a node to its first index
  * @param e A representation of the EulerTour as a `Seq[T]`
  * @param h The depths of the Euler Tour represented as a `Seq[Int]`
  */
class EulerTour[T](r: Map[T, Int], e: Seq[T], h: Seq[Int]) {
  private def lg(x: Double): Double = math.log(x) / math.log(2)

  /** Range Minimum Query of an Euler Tour using a naive algorithm.
    *
    * @param x The first query bound
    * @param y The second query bound
    * @return The minimum between the first and second query
    * @note The order of '''x''' and '''y''' does not matter
    * @note '''Performance''':
    *   - preprocessing: `O(1)`
    *   - query: `O(n)`
    */
  def rmqNaive(x: T, y: T): T = {
    val Seq(i, j) = Seq(r(x), r(y)).sorted
    e.zip(h).slice(i, j + 1).minBy(_._2)._1
  }

  // n: the length of the Euler Tour
  // m: the size of blocks the Euler Tour is split into
  private val n = h.size
  private val m = math.max(1, math.ceil(lg(n) / 2).toInt)

  /** Split up the tour into blocks of size m, padding the last block to
    * be a multiple of m. Compute the minimum of each block, a, and
    * the index of that minimum in each block, b.
    */
  private lazy val blocks = (h ++ (1 to (m - n % m))).grouped(m).toArray
  private lazy val a = blocks map (_.min)
  private lazy val b = blocks map (b => b.indexOf(b.min))

  /** Construct a Sparse Table (ST) representation for the minimum index
    * of a sequence of integers. Data in the returned array is indexed
    * as: [base, power of 2 range]
    */
  private def constructSparseTable(x: Seq[Int]): Array[Array[Int]] = {
    val tmp = Array.ofDim[Int](x.size + 1, math.ceil(lg(x.size)).toInt)
    for (i <- 0 to x.size - 1; j <- 0 to math.ceil(lg(x.size)).toInt - 1) {
      tmp(i)(j) = -1
    }

    def tableRecursive(base: Int, size: Int): Int = {
      if (size == 0) {
        tmp(base)(size) = base
        base
      } else {
        val (a, b, c) = (base, base + (1 << (size - 1)), size - 1)

        val l = if (tmp(a)(c) != -1) { tmp(a)(c)            }
        else                         { tableRecursive(a, c) }

        val r = if (tmp(b)(c) != -1) { tmp(b)(c)            }
        else                         { tableRecursive(b, c) }

        val min = if (x(l) < x(r)) l else r
        tmp(base)(size) = min
        assert(min >= base)
        min
      }
    }

    for (i <- (0 to x.size - 1);
         j <- (0 to math.ceil(lg(x.size)).toInt - 1);
         if i + (1 << j) - 1 < x.size) {
      tableRecursive(i, j)
    }
    tmp
  }
  private lazy val st = constructSparseTable(a)

  /** Precompute all possible RMQs for an array of size `n where each
    * entry in the range is different from the last by only +-1
    */
  private def constructTableLookups(n: Int): Array[Array[Array[Int]]] = {
    def sortSeqSeq[T <: Int](x: Seq[T], y: Seq[T]): Boolean = {
      if (x(0) != y(0)) x(0) < y(0) else sortSeqSeq(x.tail, y.tail)
    }

    val size = m - 1
    val out = Seq.fill(size)(Seq(-1, 1))
      .flatten.combinations(m - 1).flatMap(_.permutations).toList
      .sortWith(sortSeqSeq)
      .map(_.foldLeft(Seq(0))((h, pm) => (h.head + pm) +: h).reverse)
      .map{ a =>
        var tmp = Array.ofDim[Int](m, m)
        for (i <- 0 to size; j <- i to size) yield {
          val window = a.slice(i, j + 1)
          tmp(i)(j) = window.indexOf(window.min) + i }
        tmp }.toArray
    out
  }
  private lazy val tables = constructTableLookups(m)

  /** Compute the precomputed table index of a given block */
  private def mapBlockToTable(block: Seq[Int]): Int = {
    var index = 0
    var power = block.size - 2
    for (Seq(l, r) <- block.sliding(2)) {
      if (l < r) { index += 1 << power }
      power -= 1
    }
    index
  }

  /** Precompute a mapping of all blocks to their precomputed RMQ table
    * indices
    */
  private def mapBlocksToTables(blocks: Seq[Seq[Int]]): Array[Int] = {
    val out = blocks.map(mapBlockToTable(_)).toArray
    out
  }
  private lazy val tableIdx = mapBlocksToTables(blocks)

  /** Range Minimum Query using the Berkman--Vishkin algorithm with the
    * simplifications of Bender--Farach-Colton.
    *
    * @param x The first query bound
    * @param y The second query bound
    * @return The minimum between the first and second query
    * @note The order of '''x''' and '''y''' does not matter
    * @note '''Performance''':
    *   - preprocessing: `O(n)`
    *   - query: `O(1)`
    */
  def rmqBV(x: T, y: T): T = {
    val Seq(i, j) = Seq(r(x), r(y)).sorted

    // Compute block and word indices
    val (block_i, block_j) = (i / m, j / m)
    val (word_i,  word_j)  = (i % m, j % m)

    /** Up to four possible minimum indices are then computed based on the
      * following conditions:
      *   1. `i` and `j` are in the same block:
      *     - one precomputed RMQ from `i` to `j`
      *   2. `i` and `j` are in adjacent blocks:
      *     - one precomputed RMQ from `i` to the end of its block
      *     - one precomputed RMQ from `j` to the beginning of its block
      *   3. `i` and `j` have blocks between them:
      *     - one precomputed RMQ from `i` to the end of its block
      *     - one precomputed RMQ from `j` to the beginning of its block
      *     - two sparse table lookups to fully cover all blocks
      *       between `i` and `j`
      */
    val minIndices = (block_i, block_j) match {
      case (bi, bj) if (block_i == block_j) =>
        val min_i = block_i * m + tables(tableIdx(block_i))(word_i)(word_j)
        Seq(min_i)
      case (bi, bj) if (block_i == block_j - 1) =>
        val min_i = block_i * m + tables(tableIdx(block_i))(word_i)( m - 1)
        val min_j = block_j * m + tables(tableIdx(block_j))(     0)(word_j)
        Seq(min_i, min_j)
      case _ =>
        val min_i = block_i * m + tables(tableIdx(block_i))(word_i)( m - 1)
        val min_j = block_j * m + tables(tableIdx(block_j))(     0)(word_j)
        val (min_between_l, min_between_r) = {
          val range = math.floor(lg(block_j - block_i - 1)).toInt
          val base_0 = block_i + 1
          val base_1 = block_j - (1 << range)

          val (idx_0, idx_1) = (st(base_0)(range), st(base_1)(range))
          val (min_0, min_1) = (b(idx_0) + idx_0 * m, b(idx_1) + idx_1 * m)
          (min_0, min_1) }
        Seq(min_i, min_between_l, min_between_r, min_j)
    }

    // Return the minimum of all possible minimum indices
    e(minIndices.minBy(h(_)))
  }

  /** Range Minimum Query of the Euler Tour.
    *
    * Use this for typical queries.
    *
    * @param x The first query bound
    * @param y The second query bound
    * @return The minimum between the first and second query
    * @note This currently maps to `rmqBV`, but may choose to map to
    * either `rmqBV` or `rmqNaive`
    * @note The order of '''x''' and '''y''' does not matter
    */
  def rmq(x: T, y: T): T = rmqBV(x, y)
}
