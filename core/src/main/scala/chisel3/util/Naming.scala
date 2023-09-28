package chisel3.util

/* Generates a safe 'simple class name' from the given class, avoiding `Malformed class name` exceptions from `getClass.getSimpleName`
 * when Java 8 is used.
 */
object simpleClassName {

  def apply[T](clazz: Class[T]): String = {
    /* The default class name is derived from the Java reflection derived class name. */
    val baseName = clazz.getName

    /* A sequence of string filters applied to the name */
    val filters: Seq[String => String] =
      Seq(((a: String) => raw"\$$+anon".r.replaceAllIn(a, "_Anon")) // Merge the "$$anon" name with previous name
      )

    filters
      .foldLeft(baseName) { case (str, filter) => filter(str) } // 1. Apply filters to baseName
      .split("\\.|\\$") // 2. Split string at '.' or '$'
      .filterNot(_.forall(_.isDigit)) // 3. Drop purely numeric names
      .last // 4. Use the last name
  }
}
