/**
 * Hide provider credentials from the default test run.
 *
 * Nearly every suite here is a live end-to-end call, gated on a credential
 * being present: `describe.skipIf(!process.env.OPENROUTER_API_KEY)`. Those
 * gates are correct, but they answer "is a key present", and on a developer
 * machine that is not the same question as "should this run".
 *
 * Two ways it goes wrong, both observed here:
 *
 * - A placeholder counts as a key. `OPENAI_API_KEY=local`, set machine-wide
 *   for a local OpenAI-compatible server, unskipped 38 tests that then failed
 *   with `401 Incorrect API key provided: local`.
 * - A real key does not mean the account can serve the model. A valid
 *   `OPENROUTER_API_KEY` unskipped 19 tests that failed with `404 No allowed
 *   providers are available`, because they name upstream's fixture models
 *   rather than ones this account routes to.
 *
 * Neither says anything about the code. A suite whose failures are always
 * ignored is worse than no suite: 58 standing failures are exactly where a
 * real regression hides.
 *
 * So the live suites are opt-in. `PI_E2E=1 npm test` restores them in full.
 * This clears the variables rather than editing the gates, so upstream's test
 * files stay untouched and merging from upstream stays cheap.
 *
 * CI is exempt: if a CI job supplies real keys it intends the live suites to
 * run, so clearing them there would silently hide the tests instead. The
 * standard `CI` environment variable gates this.
 */

/** Every API-key variable in `getApiKeyEnvVars`, plus the OAuth token forms. */
const CREDENTIAL_ENV_VARS = [
	"AI_GATEWAY_API_KEY",
	"ANTHROPIC_API_KEY",
	"ANTHROPIC_OAUTH_TOKEN",
	"AZURE_OPENAI_API_KEY",
	"CEREBRAS_API_KEY",
	"CLOUDFLARE_API_KEY",
	"COPILOT_GITHUB_TOKEN",
	"DEEPSEEK_API_KEY",
	"FIREWORKS_API_KEY",
	"GEMINI_API_KEY",
	"GH_TOKEN",
	"GITHUB_TOKEN",
	"GOOGLE_CLOUD_API_KEY",
	"GROQ_API_KEY",
	"HF_TOKEN",
	"KIMI_API_KEY",
	"MINIMAX_API_KEY",
	"MINIMAX_CN_API_KEY",
	"MISTRAL_API_KEY",
	"MOONSHOT_API_KEY",
	"OPENAI_API_KEY",
	"OPENCODE_API_KEY",
	"OPENROUTER_API_KEY",
	"PRIME_API_KEY",
	"XAI_API_KEY",
	"XIAOMI_API_KEY",
	"XIAOMI_TOKEN_PLAN_AMS_API_KEY",
	"XIAOMI_TOKEN_PLAN_CN_API_KEY",
	"XIAOMI_TOKEN_PLAN_SGP_API_KEY",
	"ZAI_API_KEY",
];

if (process.env.PI_E2E !== "1" && process.env.CI !== "true") {
	for (const name of CREDENTIAL_ENV_VARS) delete process.env[name];
}
