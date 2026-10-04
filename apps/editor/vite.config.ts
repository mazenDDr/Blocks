import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// /api is served by the FastAPI control service (see README: uvicorn on 127.0.0.1:8000). Set VOID_API to proxy to another address.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "VOID_");
  return {
    plugins: [react()],
    server: {
      port: 5173,
      // VOID_API_TOKEN (optional) is added here, on the dev server, so the token never ships in the browser bundle.
      proxy: { "/api": { target: env.VOID_API ?? "http://127.0.0.1:8000", changeOrigin: true,
                         headers: env.VOID_API_TOKEN ? { Authorization: `Bearer ${env.VOID_API_TOKEN}` } : undefined } },
    },
  };
});
