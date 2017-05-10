import chiselBuild.ChiselSettings

enablePlugins(BuildInfoPlugin)

buildInfoPackage := ChiselSettings.safeScalaIdentifier(name.value)

buildInfoOptions += BuildInfoOption.BuildTime

buildInfoUsePackageAsPath := true

buildInfoKeys := Seq[BuildInfoKey](buildInfoPackage, version, scalaVersion, sbtVersion)
