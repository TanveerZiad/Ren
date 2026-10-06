import { FormEvent, useEffect, useState } from 'react'
import './settings-workspace.css'

const API = 'http://127.0.0.1:8765/api'
type Settings = { provider: string; base_url: string; model: string; api_key_configured: boolean }

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, init)
  if (!response.ok) throw new Error((await response.text()) || `Request failed (${response.status})`)
  return response.json() as Promise<T>
}

export function SettingsWorkspace({ onNotice }: { onNotice: (message: string) => void }) {
  const [settings, setSettings] = useState<Settings>({ provider: 'openrouter', base_url: 'https://openrouter.ai/api/v1', model: 'openai/gpt-4o-mini', api_key_configured: false })
  const [apiKey, setApiKey] = useState('')
  const [saving, setSaving] = useState(false)

  useEffect(() => { void request<Settings>('/settings').then(setSettings).catch((error) => onNotice((error as Error).message)) }, [])

  async function save(event: FormEvent) {
    event.preventDefault(); setSaving(true)
    try {
      const next = await request<Settings>('/settings', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...settings, api_key: apiKey || undefined }) })
      setSettings(next); setApiKey(''); onNotice('Provider settings saved locally. Ren uses the operating-system credential store when it is available.')
    } catch (error) { onNotice(`Settings could not be saved: ${(error as Error).message}`) } finally { setSaving(false) }
  }

  return <section className="settings-workspace"><header><p className="eyebrow">Ren / Settings</p><h1>Make the model yours.</h1><p>Ren stays local. Only the text needed for an active chat is sent to the provider you configure.</p></header><form onSubmit={save} className="settings-card"><label>Provider<input value={settings.provider} onChange={(event) => setSettings({ ...settings, provider: event.target.value })} placeholder="openrouter" /></label><label>API base URL<input type="url" value={settings.base_url} onChange={(event) => setSettings({ ...settings, base_url: event.target.value })} /></label><label>Model<input value={settings.model} onChange={(event) => setSettings({ ...settings, model: event.target.value })} placeholder="openai/gpt-4o-mini" /></label><label>API key <small>{settings.api_key_configured ? 'A key is already saved. Leave blank to keep it.' : 'No key saved yet.'}</small><input type="password" value={apiKey} onChange={(event) => setApiKey(event.target.value)} placeholder="Paste a new key only when changing it" autoComplete="off" /></label><div className="settings-actions"><a href="https://openrouter.ai/keys" target="_blank" rel="noreferrer">Manage OpenRouter keys ↗</a><button className="primary" disabled={saving}>{saving ? 'Saving…' : 'Save settings'}</button></div></form><aside className="settings-note"><strong>Available now</strong><p>Source-grounded chat, local threads, adjustable thinking, file-backed citations, local memories, and artifact records.</p><strong>Deliberately next</strong><p>Live web search, MCP/OAuth execution, encrypted locked chats, voice, image generation, and public sharing need their own secure, tested implementation—not a pretend toggle.</p></aside></section>
}
