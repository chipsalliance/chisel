resolvers += Resolver.url("scalasbt", new URL("https://scalasbt.artifactoryonline.com/scalasbt/sbt-plugin-releases")) (Resolver.ivyStylePatterns)

resolvers += Classpaths.sbtPluginReleases

resolvers += "jgit-repo" at "https://download.eclipse.org/jgit/maven"

addSbtPlugin("org.scoverage" % "sbt-scoverage" % "1.8.0")

addSbtPlugin("com.typesafe.sbt" % "sbt-site" % "1.4.0")

addSbtPlugin("com.eed3si9n" % "sbt-buildinfo" % "0.10.0")

addSbtPlugin("com.eed3si9n" % "sbt-unidoc" % "0.4.3")

addSbtPlugin("ch.epfl.scala" % "sbt-scalafix" % "0.9.15")

addSbtPlugin("com.thoughtworks.sbt-api-mappings" % "sbt-api-mappings" % "3.0.0")

addSbtPlugin("org.scalameta" % "sbt-mdoc" % "2.2.20" )

addSbtPlugin("com.eed3si9n" % "sbt-sriracha" % "0.1.0")

addSbtPlugin("com.typesafe" % "sbt-mima-plugin" % "0.9.0")

addSbtPlugin("com.geirsson" % "sbt-ci-release" % "1.5.7")

// From FIRRTL for building from source
addSbtPlugin("ch.epfl.scala" % "sbt-scalafix" % "0.9.27")
