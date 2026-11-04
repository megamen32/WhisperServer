// The API key and transcript enter through stdin, never command-line arguments.
import { runByokModel } from '@bezrabotnyi/byok';
import { fallbackByokPresets } from '@bezrabotnyi/byok/catalog';

if (process.argv[2] === 'presets') {
  process.stdout.write(JSON.stringify(fallbackByokPresets));
} else {
  try {
    let input = '';
    for await (const part of process.stdin) input += part;
    const { config, system, prompt } = JSON.parse(input);
    const text = await runByokModel(config, { system, prompt }, { timeoutMs: 240000 });
    process.stdout.write(JSON.stringify({ text }));
  } catch {
    // Provider errors may contain headers or keys; do not log their contents.
    process.stdout.write(JSON.stringify({ error: 'Не удалось получить ответ провайдера. Проверьте ключ, адрес и модель.' }));
    process.exitCode = 1;
  }
}
