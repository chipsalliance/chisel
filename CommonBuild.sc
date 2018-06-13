def scalacOptionsVersion(scalaVersion: String): Seq[String] = {
  Seq() ++ {
    // If we're building with Scala > 2.11, enable the compile option
    //  switch to support our anonymous Bundle definitions:
    //  https://github.com/scala/bug/issues/10047
    if (scalaVersion.startsWith("2.11.")) {
      Seq()
    } else {
      Seq(
        "-Xsource:2.11",
        "-Ywarn-unused:imports",
        "-Ywarn-unused:locals"
      )
    }
  }
}

def javacOptionsVersion(scalaVersion: String): Seq[String] = {
  Seq() ++ {
    // Scala 2.12 requires Java 8. We continue to generate
    //  Java 7 compatible code for Scala 2.11
    //  for compatibility with old clients.
    if (scalaVersion.startsWith("2.11.")) {
      Seq("-source", "1.7", "-target", "1.7")
    } else {
      Seq("-source", "1.8", "-target", "1.8")
    }
  }
}
