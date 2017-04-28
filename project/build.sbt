// Share everything in the upper level project/chiselBuild directory between build and meta-build
unmanagedSourceDirectories in Compile += baseDirectory.value / "chiselBuild"
