{ fetchFromGitHub,
  python3,
  stdenv, cmake, ninja, makeWrapper }:
let
  src = fetchFromGitHub {
    owner = "llvm";
    repo = "llvm-project";
    rev = "llvmorg-17.0.3";
    hash = "sha256-fXkYSYwhMeY3nLuuSlO2b/TpTxa+61zxmsfFg7a2Gbo=";
  };

  llvm-lit = python3.pkgs.buildPythonPackage {
    inherit src;

    pname = "llvm-lit";
    version = "17.0.3";
    format = "setuptools";
    sourceRoot = "source/llvm/utils/lit";
    propagatedBuildInputs = [
      python3.pkgs.setuptools
    ];
    doCheck = false;
  };
in
stdenv.mkDerivation {
  inherit src;

  pname = "llvm-lit";
  version = "17.0.3";
  nativeBuildInputs = [ cmake ninja python3 makeWrapper ];
  cmakeDir = "../llvm";
  cmakeFlags = [
    "-DLLVM_TARGETS_TO_BUILD=X86"
    "-DLLVM_INSTALL_UTILS=ON"
    "-DLLVM_INCLUDE_UTILS=ON"
    "-DLLVM_INCLUDE_RUNTIMES=OFF"
    "-DLLVM_INCLUDE_EXAMPLES=OFF"
    "-DLLVM_INCLUDE_BENCHMARKS=OFF"
    "-DLLVM_ENABLE_OCAMLDOC=OFF"
    "-DLLVM_BUILD_RUNTIME=OFF"
    "-DLLVM_BUILD_TOOLS=OFF"
  ];

  postInstall = ''
    makeWrapper ${llvm-lit}/bin/lit $out/bin/llvm-lit \
      --prefix PATH : $out/bin
    ln -s ${llvm-lit}/lib/python* $out/lib/
    cp -r ${llvm-lit}/nix-support $out/
  '';
}
