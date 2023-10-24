circtSrc: final: prev:
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
    version = circtSrc.shortRev;
    cmakeFlags = [
      "-DBUILD_SHARED_LIBS=ON"
      "-DLLVM_ENABLE_BINDINGS=OFF"
      "-DLLVM_ENABLE_OCAMLDOC=OFF"
      "-DLLVM_BUILD_EXAMPLES=OFF"
      "-DLLVM_OPTIMIZED_TABLEGEN=ON"
      "-DLLVM_ENABLE_PROJECTS=mlir"
      "-DLLVM_EXTERNAL_PROJECTS=circt"
      "-DLLVM_EXTERNAL_CIRCT_SOURCE_DIR=.."
      "-DCIRCT_LLHD_SIM_ENABLED=OFF"
    ];
    src = circtSrc;
    preConfigure = ''
      find ./test -name '*.mlir' -exec sed -i 's|/usr/bin/env|${final.coreutils}/bin/env|g' {} \;
      substituteInPlace cmake/modules/GenVersionFile.cmake --replace "unknown git version" "git version ${version}"
    '';
    installPhase = ''
      runHook preInstall
      mkdir -p $out
      CMAKE_INSTALL_PREFIX=$out cmake --build . --target install --config Release
      runHook postInstall
    '';
  });

  llvm-lit = final.callPackage ./nix/llvm-lit.nix { };
}
