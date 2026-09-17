import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],

  server: {
    port: 5173,

    proxy: {
      "/backend": {
        target: "https://localhost:8443",
        changeOrigin: true,
        secure: false,

        rewrite: (path) =>
          path.replace(/^\/backend/, ""),
      },
    },
  },
});