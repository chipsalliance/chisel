final: prev:
{
  mill = (prev.mill.overrideAttrs (oldAttrs: rec {
    version = "0.11.5";
    src = prev.fetchurl {
      url = "https://github.com/com-lihaoyi/mill/releases/download/${version}/${version}-assembly";
      hash = "sha256-sCJMCy4TLRQV3zI28Aydv5a8OV8OHOjLbwhfyIlxOeY=";
    };
  })).override {
    jre = final.openjdk21;
  };

  jextract = (prev.jextract.overrideAttrs (oldAttrs: rec {
    src = final.fetchFromGitHub {
      owner = "openjdk";
      repo = "jextract";
      rev = "jdk21";
      hash = "sha256-jkUCh4oSgilszvvU7RpozFATIghaA9rJRAdUIl5jTHM=";
    };

    env = {
      ORG_GRADLE_PROJECT_llvm_home = final.llvmPackages.libclang.lib;
      ORG_GRADLE_PROJECT_jdk21_home = final.jdk21;
    };
  }));

  espresso = final.callPackage ./nix/espresso.nix { };
  circt = prev.circt.overrideAttrs (old: rec {
    version = "nightly";
    src = final.fetchFromGitHub {
      owner = "llvm";
      repo = "circt";
      rev = "57372957e8365b34ca469299b8c864d830e836a1";
      sha256 = "sha256-gExhWkhVhIpTKRCfF26pZnrcrf//ASQJDxEKbYc570s=";
      fetchSubmodules = true;
    };
  });
}
