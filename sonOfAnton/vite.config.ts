import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// https://vite.dev/config/
export default defineConfig({
  theme: {
    extend: {
      keyframes: {
        pulseGlow: {
          "0%, 100%": {
            transform: "scale(1)",
            boxShadow: "0 0 0 0 rgba(168, 85, 247, 0.7)",
          },
          "50%": {
            transform: "scale(1.05)",
            boxShadow: "0 0 30px 15px rgba(168, 85, 247, 0)",
          },
        },
      },
      animation: {
        pulseGlow: "pulseGlow 2.5s infinite",
      },
    },
  },
  plugins: [react(), tailwindcss()],
});
