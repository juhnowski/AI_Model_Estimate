#!/usr/bin/env bash
# /root/AI_Model_Estimate/run_quality_tests.sh

set -euo pipefail

MODELS=(
    "qwen3:235b"
    "nemotron-3.5-super:latest"
    "qwen3.5:122b"
    "qwen3.8:27b"
    "gpt-oss:120b"
)

PROMPT="Напиши подробную статью про архитектуру трансформеров на 500 слов"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE}")" && pwd)"
QUALITY_DIR="$SCRIPT_DIR/quality_results"

mkdir -p "$QUALITY_DIR"

echo "========================================================"
echo "📝 Старт конвейера проверки качества через API"
echo "========================================================"

API_ENDPOINT="http://localhost:11434/api/generate"

for MODEL in "${MODELS[@]}"; do
    SAFE_MODEL_NAME="${MODEL//:/-}"
    OUTPUT_FILE="$QUALITY_DIR/${SAFE_MODEL_NAME}_article.txt"

    echo -e "\n🎯 Запуск генерации для модели: $MODEL"

    if ! ollama list | grep -q "$MODEL"; then
        echo "⚠️ Модель $MODEL не найдена. Пропуск."
        continue
    fi

    JSON_PAYLOAD=$(cat <<EOF
{
  "model": "$MODEL",
  "prompt": "$PROMPT",
  "stream": false
}
EOF
)

    echo "🏃 Отправка POST-запроса к Ollama..."
    RESPONSE=$(nix-shell -p curl python3 --run "curl -s --max-time 1200 -X POST $API_ENDPOINT -d '$JSON_PAYLOAD'")

    
    echo "💾 Сохранение чистого текста ответа..."
    echo "$RESPONSE" | nix-shell -p python3 --run "python3 -c \"import sys, json; print(json.load(sys.stdin)['response'])\"" > "$OUTPUT_FILE"

    # Извлекаем статистику
    EVAL_COUNT=$(echo "$RESPONSE" | nix-shell -p python3 --run "python3 -c \"import sys, json; print(json.load(sys.stdin).get('eval_count', 0))\"")
    EVAL_DURATION=$(echo "$RESPONSE" | nix-shell -p python3 --run "python3 -c \"import sys, json; print(json.load(sys.stdin).get('eval_duration', 1))\"")
    
    # Считаем скорость через bc (встроенный калькулятор NixOS) вместо прямого python3
    if [ "$EVAL_COUNT" -gt 0 ]; then
        TPS=$(nix-shell -p bc --run "echo \"scale=2; $EVAL_COUNT / ($EVAL_DURATION / 1000000000)\" | bc")
    else
        TPS="0.00"
    fi

    WORD_COUNT=$(wc -w < "$OUTPUT_FILE")
    echo "🎉 Успешно завершено. Слов: $WORD_COUNT | Скорость: $TPS ток/с"
done
