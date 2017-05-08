enablePlugins(BuildInfoPlugin)

    buildInfoPackage := name.value

    buildInfoOptions += BuildInfoOption.BuildTime

    buildInfoUsePackageAsPath := true

    buildInfoKeys := Seq[BuildInfoKey](buildInfoPackage, version, scalaVersion, sbtVersion)
