// Shared Playwright launcher: real GPU via ANGLE/Metal. The tests read the adapter / UNMASKED renderer string
// and report honestly if the browser fell back to SwiftShader.
import { chromium } from 'playwright';
import { existsSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { homedir } from 'node:os';

function cachedChrome() {
  if (process.env.CHROME_PATH) return process.env.CHROME_PATH;
  const root = join(homedir(), 'Library/Caches/ms-playwright');
  if (!existsSync(root)) return undefined;
  for (const d of readdirSync(root).filter((d) => /^chromium-\d+$/.test(d)).sort().reverse()) {
    const p = join(root, d, 'chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing');
    if (existsSync(p)) return p;
  }
  return undefined;
}

export async function launch() {
  const args = ['--use-angle=metal', '--enable-gpu', '--ignore-gpu-blocklist', '--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU'];
  try { return await chromium.launch({ args, headless: true, channel: 'chromium' }); }
  catch { return chromium.launch({ args, executablePath: cachedChrome() }); }
}

export const BASE = process.env.BLONDE_URL ?? 'http://127.0.0.1:5183/';

export async function open(browser, { query = '', size = 900, dpr = 1, touch = false } = {}) {
  const ctx = await browser.newContext({ viewport: { width: size, height: size }, deviceScaleFactor: dpr, hasTouch: touch });
  const page = await ctx.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error' && !/404|favicon/.test(m.text())) errors.push(m.text()); });
  await page.goto(BASE + query);
  await page.waitForFunction(() => window.__blonde !== undefined, null, { timeout: 90000 });
  const ok = await page.evaluate(() => window.__blonde.ready);
  if (!ok) throw new Error('app failed to boot: ' + (await page.evaluate(() => window.__blonde.error)) + '\n' + errors.join('\n'));
  await page.waitForFunction(() => window.__blonde.stats().frames > 20, null, { timeout: 60000 });
  return { ctx, page, errors };
}
