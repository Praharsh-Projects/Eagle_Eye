import { defineConfig, devices } from "@playwright/test";

const port = Number(process.env.EAGLE_EYE_E2E_PORT || 18081);
const baseURL = `http://127.0.0.1:${port}`;

export default defineConfig({
  testDir: "./tests/e2e",
  outputDir: "./test-results",
  timeout: 30_000,
  expect: { timeout: 8_000 },
  fullyParallel: false,
  forbidOnly: true,
  retries: 0,
  reporter: [["line"], ["html", { outputFolder: "playwright-report", open: "never" }]],
  use: {
    baseURL,
    colorScheme: "dark",
    reducedMotion: "reduce",
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
  },
  webServer: {
    command: `EAGLE_EYE_SKIP_DOTENV=true OPENAI_API_KEY= EAGLE_EYE_ENABLE_MODEL_RESPONSES=false ADDRESS=127.0.0.1 PORT=${port} ../run_fastapi.sh`,
    url: `${baseURL}/health`,
    reuseExistingServer: false,
    timeout: 30_000,
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
});
