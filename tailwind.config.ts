import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        "v-white": {
          100: "#ffffff",
          200: "#fafafa",
        },
        "v-gray": {
          1000: "#171717",
          900: "#666666",
          800: "#7d7d7d",
          700: "#8f8f8f",
          600: "#a8a8a8",
          500: "#c9c9c9",
          400: "#eaeaea",
        },
        "v-purple": {
          700: "#8e4ec6",
          900: "#7820bc",
        },
        "v-red": {
          700: "#e5484d",
          900: "#cb2a2f",
        },
        "v-amber": {
          700: "#f5b047",
          900: "#a35200",
        },
        "v-green": {
          700: "#45a557",
          900: "#297a3a",
        },
        "v-blue": {
          600: "#52aeff",
          700: "#0072f5",
          900: "#0068d6",
        },
        background: "#282c34",
      },
      typography: {},
    },
  },
  plugins: [require("@tailwindcss/typography")],
};

export default config;
