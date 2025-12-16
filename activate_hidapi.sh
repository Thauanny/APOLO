#!/bin/bash
# Script para ativar o ambiente virtual com suporte a hidapi no macOS

# Ativa o ambiente virtual
source "$(dirname "${BASH_SOURCE[0]}")/.venv/bin/activate"

# Define o caminho para a biblioteca hidapi instalada via Homebrew
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/hidapi/lib:$DYLD_LIBRARY_PATH"

echo "✓ Ambiente virtual ativado com hidapi configurado!"
