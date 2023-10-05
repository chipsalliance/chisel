resolvers += Resolver.url("scalasbt", new URL("https://scalasbt.artifactoryonline.com/scalasbt/sbt-plugin-releases"))(
  Resolver.ivyStylePatterns
)

resolvers += Classpaths.sbtPluginReleases

resolvers += "jgit-repo".at("https://download.eclipse.org/jgit/maven")

addSbtPlugin("org.scoverage" % "sbt-scoverage" % "1.9.3")

addSbtPlugin("com.typesafe.sbt" % "sbt-site" % "1.4.1")

addSbtPlugin("com.eed3si9n" % "sbt-buildinfo" % "0.11.0")

addSbtPlugin("com.github.sbt" % "sbt-unidoc" % "0.5.0")

addSbtPlugin("ch.epfl.scala" % "sbt-scalafix" % "0.11.0")

addSbtPlugin("com.thoughtworks.sbt-api-mappings" % "sbt-api-mappings" % "3.0.2")

addSbtPlugin("org.scalameta" % "sbt-mdoc" % "2.3.7")

addSbtPlugin("com.eed3si9n" % "sbt-sriracha" % "0.1.0")

addSbtPlugin("com.typesafe" % "sbt-mima-plugin" % "1.1.2")

addSbtPlugin("com.github.sbt" % "sbt-ci-release" % "1.5.12")

addSbtPlugin("org.scalameta" % "sbt-scalafmt" % "2.5.0")

// From FIRRTL

addSbtPlugin("com.eed3si9n" % "sbt-assembly" % "2.1.1")
