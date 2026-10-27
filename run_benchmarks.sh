#!/usr/bin/env bash
# /home/ilya/src/AI_Model_Estimate/run_benchmarks.sh

set -euo pipefail

# Список моделей для последовательного тестирования из проекта AI_Model_Estimate
MODELS=(
    "qwen3:235b"
    "nemotron-3.5-super"
    "qwen3.5:122b"
    "qwen3.8:27b"
    "gpt-oss:120b"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE}")" && pwd)"
BENCHMARK_SCRIPT="$SCRIPT_DIR/benchmark.py"
LOG_DIR="$SCRIPT_DIR/logs"

# Увеличиваем таймаут до 45 минут (2700 сек) 
# Тяжелый Qwen3:235b на PCIe-интерконнекте без NVLink обрабатывает очередь медленнее
TIMEOUT_LIMIT=2700 
mkdir -p "$LOG_DIR"

echo "========================================================"
echo "⚡ Старт последовательного конвейера тестирования моделей"
echo "========================================================"

# Проверка наличия основного скрипта тестирования
if [ ! -f "$BENCHMARK_SCRIPT" ]; then
    echo "❌ Ошибка: Файл $BENCHMARK_SCRIPT не найден!"
    exit 1
fi

for MODEL in "${MODELS[@]}"; do
    echo -e "\n--------------------------------------------------------"
    echo "🎯 Подготовка к тестированию модели: $MODEL"
    echo "--------------------------------------------------------"

    # Проверка наличия весов модели в локальном кэше Ollama
    if ! ollama list | grep -q "$MODEL"; then
        echo "📥 Модель $MODEL не найдена локально. Запуск автоматического скачивания..."
        ollama pull "$MODEL"
    else
        echo "✅ Модель $MODEL верифицирована в локальном контуре Ollama."
    fi

    echo "🏃 Запуск нагрузочного теста..."
    
    # Запуск теста с контролем зависания (hang check) через утилиту timeout
    # Если процесс зависнет в deadlock из-за дефицита памяти/очереди, он прервется по таймауту
    if ! timeout --foreground "$TIMEOUT_LIMIT" python3 "$BENCHMARK_SCRIPT" "$MODEL" > "$LOG_DIR/${MODEL//:/-}.log" 2>&1; then
        EXIT_STATUS=$?
        if [ $EXIT_STATUS -eq 124 ]; then
            echo "⚠️ КРИТИЧЕСКИЙ СБОЙ: Тест для $MODEL завершился по таймауту (Hang/Deadlock detected)!"
            echo "Проверьте распределение KV-cache и лимиты памяти в systemctl status ollama"
        else
            echo "❌ Ошибка: Тест для $MODEL завершился с кодом ошибки $EXIT_STATUS"
            echo "Подробности см. в логе: $LOG_DIR/${MODEL//:/-}.log"
        fi
        continue
    fi

    echo "🎉 Тестирование модели $MODEL успешно завершено."
    echo "📋 Сводные метрики сохранены в: $LOG_DIR/${MODEL//:/-}.log"
done

echo -e "\n========================================================"
echo "🏁 Все тесты завершены. Результаты готовы для анализа."
echo "========================================================"
