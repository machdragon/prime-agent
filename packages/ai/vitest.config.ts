import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    testTimeout: 30000, // 30 seconds for API calls
    // Live provider suites are opt-in: PI_E2E=1. See test/no-live-providers.ts.
    setupFiles: ['./test/no-live-providers.ts'],
  }
});
