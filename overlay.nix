final: prev:
{
  mill = (prev.mill.overrideAttrs (oldAttrs: rec {
    version = "0.12.5";
    src = prev.fetchurl {
      url = "https://github.com/com-lihaoyi/mill/releases/download/${version}/${version}-assembly";
      hash = "sha256-DHslQS/uzwbZVdATQY3pqQgM51W+26x2AckQnDPVoFc=";
    };
  })).override {
    jre = final.openjdk21;
  };
#  This script can be used for override circt for debuging.
#  circt = prev.circt.overrideAttrs (old: rec {
#    version = "nightly";
#    src = final.fetchFromGitHub {
#      owner = "llvm";
#      repo = "circt";
#      rev = "57372957e8365b34ca469299b8c864d830e836a1";
#      sha256 = "sha256-gExhWkhVhIpTKRCfF26pZnrcrf//ASQJDxEKbYc570s=";
#      fetchSubmodules = true;
#    };
#    preConfigure = ''
#      find ./test -name '*.mlir' -exec sed -i 's|/usr/bin/env|${final.coreutils}/bin/env|g' {} \;
#      substituteInPlace cmake/modules/GenVersionFile.cmake --replace "unknown git version" "nightly"
#    '';
#  });
  circt-all = final.symlinkJoin {
    name = "circt-all";
    paths = with final; [
      circt
      circt.dev
      circt.lib
      circt.llvm
      circt.llvm.dev
      circt.llvm.lib
    ];
  };
}
