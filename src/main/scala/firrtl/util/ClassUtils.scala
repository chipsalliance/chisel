// SPDX-License-Identifier: Apache-2.0

package firrtl.util

object ClassUtils {

  /** Determine if a named class is loaded.
    *
    * @param name - name of the class: "foo.bar" or "org.foo.bar"
    * @return true if the class has been loaded (is accessible), false otherwise.
    */
  def isClassLoaded(name: String): Boolean = {
    val found =
      try {
        Class.forName(name, false, getClass.getClassLoader) != null
      } catch {
        case e: ClassNotFoundException => false
        case x: Throwable              => throw x
      }
//    println(s"isClassLoaded: %s $name".format(if (found) "found" else "didn't find"))
    found
  }
}
