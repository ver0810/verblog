import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}"],
  darkMode: ["class", '[data-theme="dark"]'],
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
      fontFamily: {
        sans: ["Geist", "LXGWWenKai", "sans-serif"],
        mono: ["GeistMono", "LXGWWenKaiMono", "monospace"],
      },
      typography: () => ({
        DEFAULT: {
          css: {
            "--tw-prose-body": "var(--ctp-text)",
            "--tw-prose-headings": "var(--ctp-mauve)",
            "--tw-prose-h1": "var(--ctp-mauve)",
            "--tw-prose-h2": "var(--ctp-mauve)",
            "--tw-prose-h3": "var(--ctp-pink)",
            "--tw-prose-h4": "var(--ctp-pink)",
            "--tw-prose-h5": "var(--ctp-sky)",
            "--tw-prose-h6": "var(--ctp-sky)",
            "--tw-prose-links": "var(--ctp-blue)",
            "--tw-prose-bold": "var(--ctp-red)",
            "--tw-prose-counters": "var(--ctp-subtext0)",
            "--tw-prose-bullets": "var(--ctp-teal)",
            "--tw-prose-hr": "var(--ctp-surface0)",
            "--tw-prose-quotes": "var(--ctp-subtext0)",
            "--tw-prose-quote-borders": "var(--ctp-surface0)",
            "--tw-prose-captions": "var(--ctp-subtext0)",
            "--tw-prose-code": "var(--ctp-red)",
            "--tw-prose-pre-code": "var(--ctp-text)",
            "--tw-prose-pre-bg": "var(--ctp-mantle)",
            "--tw-prose-th-borders": "var(--ctp-surface0)",
            "--tw-prose-td-borders": "var(--ctp-surface0)",
            "--tw-prose-invert-body": "var(--ctp-text)",
            "--tw-prose-invert-headings": "var(--ctp-mauve)",
            "--tw-prose-invert-links": "var(--ctp-blue)",
            "--tw-prose-invert-bold": "var(--ctp-red)",
            "--tw-prose-invert-counters": "var(--ctp-subtext0)",
            "--tw-prose-invert-bullets": "var(--ctp-teal)",
            "--tw-prose-invert-hr": "var(--ctp-surface0)",
            "--tw-prose-invert-quotes": "var(--ctp-subtext0)",
            "--tw-prose-invert-quote-borders": "var(--ctp-surface0)",
            "--tw-prose-invert-captions": "var(--ctp-subtext0)",
            "--tw-prose-invert-code": "var(--ctp-red)",
            "--tw-prose-invert-pre-code": "var(--ctp-text)",
            "--tw-prose-invert-pre-bg": "var(--ctp-mantle)",
            color: "var(--ctp-text)",
            h1: {
              color: "var(--tw-prose-h1)",
              fontWeight: "700",
            },
            h2: {
              color: "var(--tw-prose-h2)",
              fontWeight: "600",
            },
            h3: {
              color: "var(--tw-prose-h3)",
              fontWeight: "600",
            },
            h4: {
              color: "var(--tw-prose-h4)",
              fontWeight: "500",
            },
            h5: {
              color: "var(--tw-prose-h5)",
              fontWeight: "500",
            },
            h6: {
              color: "var(--tw-prose-h6)",
              fontWeight: "500",
            },
            strong: {
              color: "var(--tw-prose-bold)",
              fontWeight: "700",
            },
            code: {
              color: "var(--tw-prose-code)",
              backgroundColor: "var(--ctp-surface0)",
              padding: "0.125rem 0.25rem",
              borderRadius: "0.25rem",
              fontSize: "0.875em",
            },
            "code::before": {
              content: '""',
            },
            "code::after": {
              content: '""',
            },
            pre: {
              backgroundColor: "var(--tw-prose-pre-bg)",
              color: "var(--tw-prose-pre-code)",
            },
            "pre code": {
              backgroundColor: "transparent",
              color: "inherit",
              fontSize: "inherit",
              padding: "0",
            },
            blockquote: {
              borderLeftColor: "var(--tw-prose-quote-borders)",
              color: "var(--tw-prose-quotes)",
            },
            "ul > li::before": {
              backgroundColor: "var(--tw-prose-bullets)",
            },
            "ol > li::before": {
              color: "var(--tw-prose-counters)",
            },
          },
        },
      }),
    },
  },
  plugins: [require("@tailwindcss/typography")],
};

export default config;
