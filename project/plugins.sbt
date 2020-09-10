resolvers += Resolver.url("scalasbt", new URL("https://scalasbt.artifactoryonline.com/scalasbt/sbt-plugin-releases")) (Resolver.ivyStylePatterns)

resolvers += Classpaths.sbtPluginReleases

resolvers += "jgit-repo" at "https://download.eclipse.org/jgit/maven"

addSbtPlugin("org.scoverage" % "sbt-scoverage" % "1.6.1")

addSbtPlugin("com.typesafe.sbt" % "sbt-site" % "1.4.0")

addSbtPlugin("com.eed3si9n" % "sbt-buildinfo" % "0.10.0")

addSbtPlugin("com.eed3si9n" % "sbt-unidoc" % "0.4.3")

addSbtPlugin("ch.epfl.scala" % "sbt-scalafix" % "0.9.15")

addSbtPlugin("com.thoughtworks.sbt-api-mappings" % "sbt-api-mappings" % "3.0.0")

addSbtPlugin("org.scalameta" % "sbt-mdoc" % "2.2.5" )

addSbtPlugin("com.eed3si9n" % "sbt-sriracha" % "0.1.0")

// From FIRRTL for building from source
addSbtPlugin("ch.epfl.scala" % "sbt-scalafix" % "0.9.19")
