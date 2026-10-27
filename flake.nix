# /home/ilya/src/AI_Model_Estimate/flake.nix
{
  description = "Окружение для нагрузочного тестирования LLM моделей и генерации ТЭО";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, utils }:
    utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            python311
            python311Packages.numpy
            python311Packages.tabulate
            curl
            pciutils
          ];

          shellHook = ''
            echo "========================================================"
            echo "🚀 AI_Model_Estimate Dev Environment Loaded"
            echo "Доступен Python 3.11 с модулями математического анализа"
            echo "========================================================"
          '';
        };
      });
}
