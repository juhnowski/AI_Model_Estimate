# /home/ilya/src/AI_Model_Estimate/default.nix
{ pkgs ? import <nixpkgs> {} }:

pkgs.stdenv.mkDerivation {
  pname = "ai-model-estimate";
  version = "1.0.0";
  src = ./.;

  buildInputs = with pkgs; [ python311 ];

  installPhase = ''
    mkdir -p $out/bin $out/share/doc/ai-model-estimate
    cp benchmark.py $out/bin/ai-benchmark
    cp JUSTIFICATION.md $out/share/doc/ai-model-estimate/
    chmod +x $out/bin/ai-benchmark
  '';
}
