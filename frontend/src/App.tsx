import { ChangeEvent, DragEvent, FormEvent, useCallback, useEffect, useMemo, useState } from 'react'

const API = 'http://127.0.0.1:8765/api'

type Source = { id: string; filename: string; source_kind: string; status: string; error_message?: string; created_at: string; topics: string[] }
type Topic = { id: string; name: string; source_count: number; note_status?: string }
type SearchResult = { chunk_id: string; source_id: string; filename: string; anchor: string; text: string }
type Citation = { source_id: string; filename: string; anchor: string }
type TopicDetail = { topic: Topic; sources: Source[]; draft: Revision | null; current_note: Revision | null }
type Revision = { id: string; body: string; status: string; diff_summary: string; created_at?: string }
type Settings = { base_url: string; model: string; api_key_configured: boolean }

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, init)
  if (!response.ok) throw new Error((await response.text()) || `Request failed (${response.status})`)
  return response.json() as Promise<T>
}

function App() {
  const [sources, setSources] = useState<Source[]>([])
  const [topics, setTopics] = useState<Topic[]>([])
  const [view, setView] = useState<'inbox' | 'topics' | 'ask' | 'settings'>('inbox')
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const [selectedTopic, setSelectedTopic] = useState<TopicDetail | null>(null)
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResult[]>([])
  const [answer, setAnswer] = useState<{ answer: string; citations: Citation[] } | null>(null)

  const refresh = useCallback(async () => {
    try {
      const [nextSources, nextTopics] = await Promise.all([request<Source[]>('/sources'), request<Topic[]>('/topics')])
      setSources(nextSources); setTopics(nextTopics)
    } catch (error) { setMessage(`Unable to reach Ren's local service: ${(error as Error).message}`) }
  }, [])

  useEffect(() => { void refresh(); const timer = window.setInterval(() => void refresh(), 3500); return () => window.clearInterval(timer) }, [refresh])

  const readyCount = sources.filter((source) => source.status === 'ready').length
  const queuedCount = sources.filter((source) => source.status === 'queued' || source.status === 'processing').length

  async function importFiles(files: FileList | File[]) {
    if (!files.length) return
    setBusy(true); setMessage('Importing into your local Ren library…')
    try {
      const form = new FormData(); Array.from(files).forEach((file) => form.append('files', file))
      const response = await fetch(`${API}/captures/files`, { method: 'POST', body: form })
      if (!response.ok) throw new Error(await response.text())
      const imported = await response.json() as Array<{ status: string }>
      setMessage(`${imported.length} item(s) added. Ren is processing them locally.`)
      await refresh()
    } catch (error) { setMessage(`Import failed: ${(error as Error).message}`) } finally { setBusy(false) }
  }

  async function openTopic(topic: Topic) {
    setBusy(true)
    try { setSelectedTopic(await request<TopicDetail>(`/topics/${topic.id}`)); setView('topics') }
    catch (error) { setMessage((error as Error).message) } finally { setBusy(false) }
  }

  return <main className="shell">
    <aside className="sidebar">
      <div className="brand"><span className="brand-mark">R</span><div><strong>Ren</strong><small>Personal context engine</small></div></div>
      <nav>
        <button className={view === 'inbox' ? 'active' : ''} onClick={() => setView('inbox')}>Inbox <span>{sources.length}</span></button>
        <button className={view === 'topics' ? 'active' : ''} onClick={() => { setSelectedTopic(null); setView('topics') }}>Topics <span>{topics.length}</span></button>
        <button className={view === 'ask' ? 'active' : ''} onClick={() => setView('ask')}>Ask Ren</button>
        <button className={view === 'settings' ? 'active' : ''} onClick={() => setView('settings')}>Settings</button>
      </nav>
      <div className="privacy"><b>Local by default</b><br />Your original files stay untouched. Ren only uses managed copies.</div>
    </aside>
    <section className="content">
      {message && <div className="toast">{message}<button onClick={() => setMessage('')}>×</button></div>}
      {view === 'inbox' && <Inbox sources={sources} busy={busy} onImport={importFiles} onRefresh={refresh} onMessage={setMessage} />}
      {view === 'topics' && <Topics topics={topics} selected={selectedTopic} busy={busy} onOpen={openTopic} onRefresh={refresh} onMessage={setMessage} />}
      {view === 'ask' && <Ask query={query} setQuery={setQuery} results={results} setResults={setResults} answer={answer} setAnswer={setAnswer} onMessage={setMessage} />}
      {view === 'settings' && <SettingsPanel onMessage={setMessage} />}
    </section>
  </main>
}

function Inbox({ sources, busy, onImport, onRefresh, onMessage }: { sources: Source[]; busy: boolean; onImport: (files: FileList | File[]) => Promise<void>; onRefresh: () => Promise<void>; onMessage: (message: string) => void }) {
  const [dragging, setDragging] = useState(false)
  const [mode, setMode] = useState<'text' | 'url'>('text')
  const [title, setTitle] = useState('')
  const [body, setBody] = useState('')

  async function submitCapture(event: FormEvent) {
    event.preventDefault(); if (!title.trim() || !body.trim()) return
    try {
      const payload = mode === 'text' ? { title, content: body } : { title, url: body, context: '' }
      await request(`/captures/${mode}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
      setTitle(''); setBody(''); onMessage('Capture added to Inbox. Ren is processing it.'); await onRefresh()
    } catch (error) { onMessage(`Capture failed: ${(error as Error).message}`) }
  }
  function drop(event: DragEvent<HTMLDivElement>) { event.preventDefault(); setDragging(false); void onImport(event.dataTransfer.files) }
  async function retry(id: string) { await request(`/sources/${id}/retry`, { method: 'POST' }); onMessage('Processing retried.'); await onRefresh() }
  async function remove(id: string) { if (!window.confirm('Remove the managed Ren copy and its derived knowledge? Your original file will not be touched.')) return; await request(`/sources/${id}`, { method: 'DELETE' }); await onRefresh() }

  return <>
    <header><p className="eyebrow">Your capture space</p><h1>Inbox</h1><p>Drop the mess here. Ren creates structure without changing your original files.</p></header>
    <div className={`dropzone ${dragging ? 'dragging' : ''}`} onDragOver={(e) => { e.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={drop}>
      <input id="file-input" type="file" multiple accept=".pdf,.md,.markdown,.txt,.html,.htm,.docx" onChange={(e: ChangeEvent<HTMLInputElement>) => void onImport(e.target.files!)} />
      <label htmlFor="file-input"><strong>{busy ? 'Working…' : 'Drop files here or choose files'}</strong><span>PDF, Markdown, TXT, HTML, and DOCX</span></label>
    </div>
    <section className="capture-card"><div className="capture-tabs"><button className={mode === 'text' ? 'active' : ''} onClick={() => { setMode('text'); setBody('') }}>Quick note</button><button className={mode === 'url' ? 'active' : ''} onClick={() => { setMode('url'); setBody('') }}>Save link</button></div>
      <form onSubmit={submitCapture}><input placeholder={mode === 'text' ? 'Title, e.g. RAG project idea' : 'Link title'} value={title} onChange={(e) => setTitle(e.target.value)} />
      {mode === 'text' ? <textarea placeholder="Capture the thought. No formatting required." value={body} onChange={(e) => setBody(e.target.value)} /> : <input placeholder="https://…" value={body} onChange={(e) => setBody(e.target.value)} />}
      <button className="primary" disabled={!title.trim() || !body.trim()}>Add to Inbox</button></form>
    </section>
    <section className="list-section"><div className="section-title"><h2>Imported material</h2><button className="text-button" onClick={() => void onRefresh()}>Refresh</button></div>
      {sources.length === 0 ? <div className="empty">No captures yet. Start with one messy file or a thought you do not want to lose.</div> : <div className="source-list">{sources.map((source) => <article className="source-row" key={source.id}><div className={`status ${source.status}`} /><div className="source-info"><strong>{source.filename}</strong><small>{source.topics.length ? source.topics.join(' · ') : 'Awaiting suggestions'} · {source.source_kind}</small>{source.error_message && <small className="error">{source.error_message}</small>}</div><span className="status-label">{source.status}</span>{source.status === 'failed' && <button className="text-button" onClick={() => void retry(source.id)}>Retry</button>}<a className="text-button" href={`${API}/sources/${source.id}/open`} target="_blank">Open</a><button className="text-button danger" onClick={() => void remove(source.id)}>Remove</button></article>)}</div>}
    </section>
  </>
}

function Topics({ topics, selected, busy, onOpen, onRefresh, onMessage }: { topics: Topic[]; selected: TopicDetail | null; busy: boolean; onOpen: (topic: Topic) => Promise<void>; onRefresh: () => Promise<void>; onMessage: (message: string) => void }) {
  async function generate() { if (!selected) return; try { await request(`/topics/${selected.topic.id}/note-drafts`, { method: 'POST' }); await onOpen(selected.topic); onMessage('A source-backed note draft is ready for review.') } catch (error) { onMessage((error as Error).message) } }
  async function review(action: 'approve' | 'reject') { if (!selected?.draft) return; try { await request(`/note-revisions/${selected.draft.id}/${action}`, { method: 'POST' }); await onOpen(selected.topic); await onRefresh(); onMessage(action === 'approve' ? 'Note approved and saved locally.' : 'Draft rejected.') } catch (error) { onMessage((error as Error).message) } }
  return <>
    <header><p className="eyebrow">AI-suggested structure</p><h1>Topics</h1><p>Topics are suggestions from your content, not folders Ren imposes on you.</p></header>
    <div className="topic-layout"><section className="topic-list">{topics.length === 0 ? <div className="empty">Import material first; topics will appear after processing.</div> : topics.map((topic) => <button className={`topic-row ${selected?.topic.id === topic.id ? 'selected' : ''}`} key={topic.id} onClick={() => void onOpen(topic)}><span>{topic.name}</span><small>{topic.source_count} sources {topic.note_status ? `· ${topic.note_status}` : ''}</small></button>)}</section>
      <section className="topic-detail">{!selected ? <div className="empty">Select a topic to review its sources and notes.</div> : <><div className="section-title"><div><h2>{selected.topic.name}</h2><p>{selected.sources.length} linked source(s)</p></div><button className="primary" disabled={busy} onClick={() => void generate()}>Generate note draft</button></div>
      <div className="linked-sources">{selected.sources.map((source) => <a key={source.id} href={`${API}/sources/${source.id}/open`} target="_blank">{source.filename}</a>)}</div>
      {selected.draft && <article className="note-card"><div className="section-title"><div><h3>Pending note draft</h3><small>{selected.draft.diff_summary}</small></div><div><button className="text-button danger" onClick={() => void review('reject')}>Reject</button><button className="primary" onClick={() => void review('approve')}>Approve</button></div></div><pre>{selected.draft.body}</pre></article>}
      {selected.current_note && !selected.draft && <article className="note-card"><h3>Approved living note</h3><pre>{selected.current_note.body}</pre></article>}
      {!selected.current_note && !selected.draft && <div className="empty">No note yet. Generate a draft; nothing becomes permanent until you approve it.</div>}</>}</section></div>
  </>
}

function Ask({ query, setQuery, results, setResults, answer, setAnswer, onMessage }: { query: string; setQuery: (value: string) => void; results: SearchResult[]; setResults: (items: SearchResult[]) => void; answer: { answer: string; citations: Citation[] } | null; setAnswer: (value: { answer: string; citations: Citation[] } | null) => void; onMessage: (message: string) => void }) {
  async function search(event: FormEvent) { event.preventDefault(); if (!query.trim()) return; try { setResults(await request<SearchResult[]>(`/search?q=${encodeURIComponent(query)}`)); setAnswer(null) } catch (error) { onMessage((error as Error).message) } }
  async function ask() { if (!query.trim()) return; try { setAnswer(await request<{ answer: string; citations: Citation[] }>('/questions', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: query }) })) } catch (error) { onMessage((error as Error).message) } }
  return <><header><p className="eyebrow">Grounded in your library</p><h1>Ask Ren</h1><p>Ren searches only what you imported and always shows the supporting sources.</p></header><form className="searchbar" onSubmit={search}><input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="What do I know about…" /><button className="primary">Search</button><button type="button" onClick={() => void ask()}>Answer from sources</button></form>
  {answer && <section className="answer-card"><h2>Ren’s answer</h2><p className="answer-text">{answer.answer}</p><h3>Sources</h3>{answer.citations.length ? <ul>{answer.citations.map((citation, index) => <li key={`${citation.source_id}-${index}`}><a href={`${API}/sources/${citation.source_id}/open`} target="_blank">{citation.filename}</a> · {citation.anchor}</li>)}</ul> : <p>No supporting source was found.</p>}</section>}
  {results.length > 0 && <section className="results"><h2>Matching material</h2>{results.map((result) => <article key={result.chunk_id}><div><a href={`${API}/sources/${result.source_id}/open`} target="_blank">{result.filename}</a><small>{result.anchor}</small></div><p>{result.text}</p></article>)}</section>}
  </>
}

function SettingsPanel({ onMessage }: { onMessage: (message: string) => void }) {
  const [settings, setSettings] = useState<Settings | null>(null); const [key, setKey] = useState('')
  useEffect(() => { void request<Settings>('/settings').then(setSettings).catch((error) => onMessage((error as Error).message)) }, [onMessage])
  async function save(event: FormEvent) { event.preventDefault(); if (!settings) return; try { const reply = await request<{ vault_saved: boolean | null; api_key_configured: boolean }>('/settings', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ base_url: settings.base_url, model: settings.model, api_key: key || undefined }) }); setKey(''); setSettings({ ...settings, api_key_configured: reply.api_key_configured }); onMessage(reply.vault_saved === false ? 'Settings saved. This environment has no credential vault, so the API key lasts only for this session.' : 'Settings saved securely.') } catch (error) { onMessage((error as Error).message) } }
  if (!settings) return <div className="empty">Loading settings…</div>
  return <><header><p className="eyebrow">Your machine, your key</p><h1>Settings</h1><p>AI is optional. Ren’s local import, search, and note workflow remains available without a provider.</p></header><form className="settings-card" onSubmit={save}><label>OpenAI-compatible base URL<input value={settings.base_url} onChange={(e) => setSettings({ ...settings, base_url: e.target.value })} /></label><label>Model<input value={settings.model} onChange={(e) => setSettings({ ...settings, model: e.target.value })} /></label><label>API key <small>{settings.api_key_configured ? 'A key is configured.' : 'No key configured.'}</small><input type="password" value={key} onChange={(e) => setKey(e.target.value)} placeholder="Leave blank to keep the current key" /></label><button className="primary">Save settings</button></form></>
}

export default App

