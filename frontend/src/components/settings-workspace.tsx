import { FormEvent, useEffect, useState } from 'react'
import './settings-workspace.css'

const API = 'http://127.0.0.1:8765/api'
type Settings = {
  provider: string
  base_url: string
  model: string
  embedding_model?: string
  api_key_configured: boolean
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, init)
  if (!response.ok) throw new Error((await response.text()) || `Request failed (${response.status})`)
  return response.json() as Promise<T>
}

export function SettingsWorkspace({ onNotice }: { onNotice: (message: string) => void }) {
  const [settings, setSettings] = useState<Settings>({
    provider: 'local_cpu',
    base_url: 'http://127.0.0.1:11434/v1',
    model: 'local-cpu-extractive',
    embedding_model: 'local-cpu-simhash',
    api_key_configured: false,
  })
  const [apiKey, setApiKey] = useState('')
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    void request<Settings>('/settings')
      .then(setSettings)
      .catch((error) => onNotice((error as Error).message))
  }, [])

  function applyPreset(preset: 'local_cpu' | 'ollama' | 'openrouter') {
    if (preset === 'local_cpu') {
      setSettings({
        ...settings,
        provider: 'local_cpu',
        base_url: 'http://127.0.0.1:11434/v1',
        model: 'local-cpu-extractive',
        embedding_model: 'local-cpu-simhash',
      })
    } else if (preset === 'ollama') {
      setSettings({
        ...settings,
        provider: 'ollama',
        base_url: 'http://127.0.0.1:11434/v1',
        model: 'llama3.2:3b',
        embedding_model: 'local-cpu-simhash',
      })
    } else {
      setSettings({
        ...settings,
        provider: 'openrouter',
        base_url: 'https://openrouter.ai/api/v1',
        model: 'openai/gpt-4o-mini',
        embedding_model: 'text-embedding-3-small',
      })
    }
  }

  async function save(event: FormEvent) {
    event.preventDefault()
    setSaving(true)
    try {
      const next = await request<Settings>('/settings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...settings, api_key: apiKey || undefined }),
      })
      setSettings(next)
      setApiKey('')
      onNotice('Settings saved locally. Ren is configured and ready.')
    } catch (error) {
      onNotice(`Settings could not be saved: ${(error as Error).message}`)
    } finally {
      setSaving(false)
    }
  }

  return (
    <section className="settings-workspace">
      <header>
        <p className="eyebrow">Ren / Settings & Local CPU Runtime</p>
        <h1>Run 100% locally on your PC CPU.</h1>
        <p>
          Ren runs out-of-the-box on your CPU without a GPU or cloud API key. Choose a runtime preset below or configure a custom endpoint.
        </p>
      </header>

      <div className="settings-presets">
        <button
          type="button"
          className={settings.provider === 'local_cpu' ? 'preset-btn active' : 'preset-btn'}
          onClick={() => applyPreset('local_cpu')}
        >
          <strong>1. Local CPU (Built-in)</strong>
          <span>Zero setup · Offline CPU embeddings + grounded synthesis</span>
        </button>
        <button
          type="button"
          className={settings.provider === 'ollama' ? 'preset-btn active' : 'preset-btn'}
          onClick={() => applyPreset('ollama')}
        >
          <strong>2. Local CPU LLM (Ollama / LM Studio)</strong>
          <span>Offline generative chat via localhost (no API key needed)</span>
        </button>
        <button
          type="button"
          className={settings.provider === 'openrouter' ? 'preset-btn active' : 'preset-btn'}
          onClick={() => applyPreset('openrouter')}
        >
          <strong>3. Cloud API (OpenRouter / OpenAI)</strong>
          <span>Optional external LLM & embedding API</span>
        </button>
      </div>

      <form onSubmit={save} className="settings-card">
        <label>
          Provider mode
          <input
            value={settings.provider}
            onChange={(event) => setSettings({ ...settings, provider: event.target.value })}
            placeholder="local_cpu"
          />
        </label>
        <label>
          API base URL
          <input
            type="url"
            value={settings.base_url}
            onChange={(event) => setSettings({ ...settings, base_url: event.target.value })}
          />
        </label>
        <label>
          Chat model
          <input
            value={settings.model}
            onChange={(event) => setSettings({ ...settings, model: event.target.value })}
            placeholder="local-cpu-extractive"
          />
        </label>
        <label>
          Embedding model <small>(use <code>local-cpu-simhash</code> for fast local CPU embeddings)</small>
          <input
            value={settings.embedding_model || 'local-cpu-simhash'}
            onChange={(event) => setSettings({ ...settings, embedding_model: event.target.value })}
            placeholder="local-cpu-simhash"
          />
        </label>
        <label>
          API key{' '}
          <small>
            {settings.provider === 'local_cpu' || settings.provider === 'ollama'
              ? 'Not needed for Local CPU or Ollama modes.'
              : settings.api_key_configured
                ? 'A key is already saved. Leave blank to keep it.'
                : 'No key saved yet.'}
          </small>
          <input
            type="password"
            value={apiKey}
            onChange={(event) => setApiKey(event.target.value)}
            placeholder="Optional — only needed for external cloud providers"
            autoComplete="off"
          />
        </label>
        <div className="settings-actions">
          <a href="https://openrouter.ai/keys" target="_blank" rel="noreferrer">
            Manage OpenRouter keys ↗
          </a>
          <button className="primary" disabled={saving}>
            {saving ? 'Saving…' : 'Save settings'}
          </button>
        </div>
      </form>

      <aside className="settings-note">
        <strong>Local CPU Guarantee</strong>
        <p>
          In <b>Local CPU (Built-in)</b> mode, Ren computes 256-dimensional signed random projection embeddings, FTS5 BM25 lexical matching, 2-stage proximity reranking, and cited extractive answer synthesis entirely in Python on your CPU—using under 150 MB of RAM.
        </p>
        <strong>Local Data Directory</strong>
        <p>
          Your SQLite database, imported files, and vectors live locally in <code>%APPDATA%\Ren</code>.
        </p>
      </aside>
    </section>
  )
}
