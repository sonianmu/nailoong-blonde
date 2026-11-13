import { defineConfig } from 'vite';

// COOP/COEP make the page cross-origin isolated so the solver Worker can share its position buffer
// (SharedArrayBuffer). Without the headers the app falls back to copying the buffer per step.
const isolation = { 'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp' };

export default defineConfig({
  base: './',
  server: { host: '127.0.0.1', port: 5183, headers: isolation },
  preview: { host: '127.0.0.1', port: 5184, headers: isolation },
  worker: { format: 'es' },
  build: { target: 'es2022', chunkSizeWarningLimit: 2500 },
});
