/** False in `pnpm build:static`: the Try it page needs the local relay and is left out. */
declare const __TEENIE_RELAY__: boolean;

interface ImportMetaEnv {
  /** Base URL of the teenie relay (`uv run teenie-serve`); empty when the site is served by the relay itself. */
  readonly PUBLIC_TEENIE_API?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
