resolvers += Resolver.url("scalasbt", new URL("https://scalasbt.artifactoryonline.com/scalasbt/sbt-plugin-releases"))(
  Resolver.ivyStylePatterns
)

resolvers += Classpaths.sbtPluginReleases

resolvers += "jgit-repo".at("https://download.eclipse.org/jgit/maven")

addSbtPlugin("org.scoverage" % "sbt-scoverage" % "1.9.3")

addSbtPlugin("com.typesafe.sbt" % "sbt-site" % "1.4.1")

addSbtPlugin("com.eed3si9n" % "sbt-buildinfo" % "0.12.0")

addSbtPlugin("com.github.sbt" % "sbt-unidoc" % "0.5.0")

addSbtPlugin("ch.epfl.scala" % "sbt-scalafix" % "0.12.1")

addSbtPlugin("com.thoughtworks.sbt-api-mappings" % "sbt-api-mappings" % "3.0.2")

addSbtPlugin("org.scalameta" % "sbt-mdoc" % "2.3.7")

addSbtPlugin("com.eed3si9n" % "sbt-sriracha" % "0.1.0")

addSbtPlugin("com.typesafe" % "sbt-mima-plugin" % "1.1.2")

addSbtPlugin("com.github.sbt" % "sbt-ci-release" % "1.5.12")

addSbtPlugin("org.scalameta" % "sbt-scalafmt" % "2.5.2")

// From FIRRTL

addSbtPlugin("com.eed3si9n" % "sbt-assembly" % "2.1.1")

// For generating contributors

libraryDependencies += "com.47deg" %% "github4s" % "0.32.1"

// For firtool version
libraryDependencies += "com.lihaoyi" %% "os-lib" % "0.9.2"
libraryDependencies += "com.lihaoyi" %% "upickle" % "3.1.3"

// This is an older version due to other things depending on older versions of
// scala-xml
libraryDependencies += "io.get-coursier" %% "coursier" % "2.0.16"
