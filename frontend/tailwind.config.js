/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Inter"', "system-ui", "sans-serif"],
        display: ['"Space Grotesk"', "Inter", "system-ui", "sans-serif"],
      },
      colors: {
        night: "#0b1221",
        "night-soft": "#11182a",
        accent: "#36e0f8",
        "accent-2": "#6b7bff",
      },
      boxShadow: {
        glow: "0 10px 55px rgba(54, 224, 248, 0.18)",
        "glow-strong": "0 18px 80px rgba(107, 123, 255, 0.25)",
      },
      keyframes: {
        float: {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-8px)" },
        },
        "pulse-glow": {
          "0%, 100%": { boxShadow: "0 0 0px rgba(54, 224, 248, 0.0)" },
          "50%": { boxShadow: "0 0 35px rgba(107, 123, 255, 0.45)" },
        },
        "bounce-soft": {
          "0%, 100%": { transform: "translateY(0)" },
          "50%": { transform: "translateY(-6px)" },
        },
      },
      animation: {
        float: "float 10s ease-in-out infinite",
        "pulse-glow": "pulse-glow 2.5s ease-in-out infinite",
        "bounce-soft": "bounce-soft 2.6s ease-in-out infinite",
      },
      backdropBlur: {
        glass: "18px",
      },
    },
  },
  plugins: [],
};
