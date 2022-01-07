package chisel3

/** This comparator compares the new and old elements accessor of a [[Bundle]]'s fields
  * It lives here at the present to make working on the compiler plugin easier to
  * test without having to recompile all tests in order to test the plugin.
  * It would be nice to have tests in the plugin but that is hard to do
  */
object BundleComparator {

  /**
    * @param bundle         the bundle that is being tested
    * @param showComparison shows the new and old fields side by side
    * @return               returns true if a discrepancy is found
    */
  def apply(bundle: Bundle, showComparison: Boolean = false): Boolean = {
    val header = s"=== Bundle Comparator ${bundle.className} " + "=" * 40
    if (showComparison) {
      println(header)
      println(f"${"New Field Name"}%30s ${"id"}%6s ${"Old Field Name"}%30s ${"id"}%6s")
    }

    val newElements = bundle.elements.toList
    val oldElements = bundle.oldElementsNoChecks.toList

    var discrepancyFound = false
    newElements.zipAll(oldElements, "Oops" -> Bool(), "Oops" -> Bool()).foreach {
      case ((a, b), (c, d)) =>
        val color = if (a == c) { Console.RESET }
        else { discrepancyFound = true; Console.RED }
        if (showComparison) {
          println(f"$color$a%30s (${b._id}%06x) $c%30s (${d._id}%06x) ${Console.RESET}")
        }
    }
    if (showComparison) {
      println("=" * header.length)
    }
    discrepancyFound
  }
}
