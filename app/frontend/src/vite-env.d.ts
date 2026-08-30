/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string
  readonly VITE_ACCEPT_FIXTURE_HEADER?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
