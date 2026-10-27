#!/usr/bin/env python3
# /home/ilya/src/AI_Model_Estimate/benchmark.py

import asyncio
import time
import json
import urllib.request
import statistics
import sys

# Параметры сетевого подключения к локальной шине инференса Ollama
OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = "11434"
OLLAMA_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate"

PROMPT = "Ответь кратко, в одном предложении, что такое архитектура типа Mixture-of-Experts?"
TOTAL_REQUESTS = 50
CONCURRENCY = 10  # Целевое число одновременных пользователей/агентов по ТЭО

metrics = []

async def send_request(sem, session_id, model_name):
    async with sem:
        payload = {
            "model": model_name,
            "prompt": PROMPT,
            "stream": True,
            "options": {
                "num_ctx": 4096,
                "temperature": 0.3
            }
        }
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(OLLAMA_URL, data=data, headers={'Content-Type': 'application/json'})
        
        start_time = time.time()
        ttft = None
        tokens_count = 0
        
        try:
            loop = asyncio.get_event_loop()
            # Увеличенный таймаут ожидания для тяжелых MoE моделей при работе без NVLink
            response = await loop.run_in_executor(None, lambda: urllib.request.urlopen(req, timeout=600))
            
            for line in response:
                if not line.strip():
                    continue
                chunk = json.loads(line.decode('utf-8'))
                
                if ttft is None:
                    ttft = time.time() - start_time
                
                tokens_count += 1
                if chunk.get("done", False):
                    break
                    
            total_duration = time.time() - start_time
            
            if tokens_count > 0:
                tps = tokens_count / total_duration
                metrics.append({
                    "ttft": ttft,
                    "duration": total_duration,
                    "tps": tps,
                    "tokens": tokens_count
                })
                print(f"  [Поток {session_id}] Успешно: {tokens_count} токенов, TPS: {tps:.2f}")
        except Exception as e:
            print(f"  [Поток {session_id}] Ошибка обработки: {e}")

async def run_benchmark(model_name):
    global metrics
    metrics = []
    sem = asyncio.Semaphore(CONCURRENCY)
    
    print(f"\n🚀 Запуск нагрузочного теста для модели: {model_name}")
    print(f"Параметры: {TOTAL_REQUESTS} запросов, {CONCURRENCY} одновременных сессий")
    
    start_bench = time.time()
    tasks = [send_request(sem, i, model_name) for i in range(TOTAL_REQUESTS)]
    await asyncio.gather(*tasks)
    total_bench_time = time.time() - start_bench
    
    if not metrics:
        print(f"❌ Ошибка: Не удалось получить метрики для {model_name}. Проверьте логи Ollama.")
        return

    ttfts = [m["ttft"] for m in metrics if m["ttft"] is not None]
    durations = [m["duration"] for m in metrics]
    tps_list = [m["tps"] for m in metrics]
    total_tokens = sum([m["tokens"] for m in metrics])
    
    rps = len(metrics) / total_bench_time
    p95_latency = statistics.quantiles(durations, n=20)[18] if len(durations) >= 20 else max(durations)
    
    print(f"\n📊 Агрегированные результаты для {model_name}:")
    print(f"  • Успешных сессий:       {len(metrics)} из {TOTAL_REQUESTS}")
    print(f"  • Среднее время TTFT:    {statistics.mean(ttfts):.3f} сек")
    print(f"  • Средняя скорость TPS:  {statistics.mean(tps_list):.2f} ток/сек на поток")
    print(f"  • Суммарная генерация:   {total_tokens / total_bench_time:.2f} ток/сек")
    print(f"  • Пропускная способность:{rps:.3f} запр/сек (RPS)")
    print(f"  • Задержка p95 Latency:  {p95_latency:.3f} сек")

if __name__ == "__main__":
    target_model = sys.argv[1] if len(sys.argv) > 1 else "qwen3:235b"
    asyncio.run(run_benchmark(target_model))
