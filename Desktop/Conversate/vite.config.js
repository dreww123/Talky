import { defineConfig } from 'vite'
 
export default defineConfig({
  server: {
    proxy: {
      '/api': 'http://localhost:3000',
      '/health':  'http://localhost:3000',
      '/session': 'http://localhost:3000',
      '/stream-audio': {
        target:       'ws://localhost:3000',
        ws:           true,
        changeOrigin: true,
      },
    }
  },
  build: {
    outDir:  'dist',
    target:  'es2015',
  }
})
