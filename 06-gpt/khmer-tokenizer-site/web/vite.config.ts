import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// For a GitHub Pages *project* site (https://<user>.github.io/<repo>/),
// `base` must be '/<repo>/'. Update REPO_NAME below to match your actual
// repository name before deploying. For a *user/org* site (a repo literally
// named <user>.github.io), base should just be '/'.
const REPO_NAME = 'khmer-tokenizer-site'

export default defineConfig({
  plugins: [vue()],
  base: process.env.VITE_BASE_PATH ?? `/${REPO_NAME}/`,
})
