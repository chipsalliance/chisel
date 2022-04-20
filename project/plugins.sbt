resolvers += Resolver.url("scalasbt", new URL("https://scalasbt.artifactoryonline.com/scalasbt/sbt-plugin-releases"))(
  Resolver.ivyStylePatterns
)

resolvers += Classpaths.sbtPluginReleases

resolvers += "jgit-repo".at("https://download.eclipse.org/jgit/maven")

addSbtPlugin("org.scoverage" % "sbt-scoverage" % "1.9.3")

addSbtPlugin("com.typesafe.sbt" % "sbt-site" % "1.4.1")

addSbtPlugin("com.eed3si9n" % "sbt-buildinfo" % "0.11.0")

addSbtPlugin("com.github.sbt" % "sbt-unidoc" % "0.5.0")

addSbtPlugin("ch.epfl.scala" % "sbt-scalafix" % "0.9.15")

addSbtPlugin("com.thoughtworks.sbt-api-mappings" % "sbt-api-mappings" % "3.0.0")

addSbtPlugin("org.scalameta" % "sbt-mdoc" % "2.3.2")

addSbtPlugin("com.eed3si9n" % "sbt-sriracha" % "0.1.0")

addSbtPlugin("com.typesafe" % "sbt-mima-plugin" % "1.1.0")

addSbtPlugin("com.github.sbt" % "sbt-ci-release" % "1.5.10")

addSbtPlugin("org.scalameta" % "sbt-scalafmt" % "2.4.6")

// From FIRRTL for building from source
addSbtPlugin("ch.epfl.scala" % "sbt-scalafix" % "0.9.33")
