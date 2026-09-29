import { ChangeEvent, DragEvent, useCallback, useEffect, useState } from 'react'

const API = 'http://127.0.0.1:8765/api'

type Source = { id: string; filename: string; source_kind: string; status: string; error_message?: string; topics: string[]; card_count: number }
type Card = {
  id: string; title: string; body: string; topic_name?: string; review_count: number
  last_seen_at?: string; next_review_at: string; source: { id: string; filename: string }
}

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, init)
  if (!response.ok) throw new Error((await response.text()) || `Request failed (${response.status})`)
  return response.json() as Promise<T>
}

export default function App() {
  const [view, setView] = useState<'inbox' | 'knowledge' | 'remember'>('inbox')
  const [sources, setSources] = useState<Source[]>([])
  const [cards, setCards] = useState<Card[]>([])
  const [remember, setRemember] = useState<Card[]>([])
  const [selected, setSelected] = useState<Card | null>(null)
  const [notice, setNotice] = useState('')
  const [busy, setBusy] = useState(false)

  const refresh = useCallback(async () => {
    try {
      const [nextSources, nextCards, nextRemember] = await Promise.all([
        api<Source[]>('/sources'), api<Card[]>('/cards'), api<Card[]>('/remember?limit=5'),
      ])
      setSources(nextSources); setCards(nextCards); setRemember(nextRemember)
    } catch (error) { setNotice(`Ren's local service is unavailable: ${(error as Error).message}`) }
  }, [])

  useEffect(() => { void refresh(); const timer = window.setInterval(() => void refresh(), 4000); return () => window.clearInterval(timer) }, [refresh])

  async function importFiles(files: FileList | File[]) {
    if (!files.length) return
    setBusy(true)
    try {
      const form = new FormData(); Array.from(files).forEach((file) => form.append('files', file))
      const response = await fetch(`${API}/captures/files`, { method: 'POST', body: form })
      if (!response.ok) throw new Error(await response.text())
      const result = await response.json() as Array<{ status: string; filename?: string }>
      const unsupported = result.filter((item) => item.status === 'unsupported')
      setNotice(unsupported.length ? `${unsupported.map((item) => item.filename).join(', ')} skipped. Ren currently accepts PDF, TXT, and Markdown.` : 'Added to Inbox. Ren is extracting knowledge cards.')
      await refresh()
    } catch (error) { setNotice(`Import failed: ${(error as Error).message}`) } finally { setBusy(false) }
  }

  async function review(card: Card) {
    try {
      const result = await api<{ next_review_at: string }>(`/cards/${card.id}/review`, { method: 'POST' })
      setNotice(`Reviewed. Ren will bring “${card.title}” back on ${new Date(result.next_review_at).toLocaleDateString()}.`)
      setSelected(null); await refresh()
    } catch (error) { setNotice((error as Error).message) }
  }

  async function archive(card: Card) {
    try { await api(`/cards/${card.id}/archive`, { method: 'POST' }); setSelected(null); setNotice(`Archived “${card.title}”.`); await refresh() }
    catch (error) { setNotice((error as Error).message) }
  }

  return <main className="app-shell">
    <aside className="sidebar">
      <div className="brand"><span>R</span><div><strong>Ren</strong><small>Dump → Organize → Remind</small></div></div>
      <nav>
        <button className={view === 'inbox' ? 'active' : ''} onClick={() => setView('inbox')}>📥 Inbox <b>{sources.length}</b></button>
        <button className={view === 'knowledge' ? 'active' : ''} onClick={() => setView('knowledge')}>🧠 My Knowledge <b>{cards.length}</b></button>
        <button className={view === 'remember' ? 'active' : ''} onClick={() => setView('remember')}>🔔 Remember <b>{remember.length}</b></button>
      </nav>
      <p className="rule"><strong>Ren’s rule</strong><br />Never make me organize anything.</p>
    </aside>
    <section className="page">
      {notice && <div className="notice">{notice}<button onClick={() => setNotice('')}>×</button></div>}
      {view === 'inbox' && <Inbox sources={sources} busy={busy} onImport={importFiles} onRefresh={refresh} />}
      {view === 'knowledge' && <Knowledge cards={cards} onSelect={setSelected} />}
      {view === 'remember' && <Remember cards={remember} onSelect={setSelected} />}
    </section>
    {selected && <CardDetail card={selected} onClose={() => setSelected(null)} onReview={review} onArchive={archive} />}
  </main>
}

function Inbox({ sources, busy, onImport, onRefresh }: { sources: Source[]; busy: boolean; onImport: (files: FileList | File[]) => Promise<void>; onRefresh: () => Promise<void> }) {
  const [dragging, setDragging] = useState(false)
  function drop(event: DragEvent<HTMLDivElement>) { event.preventDefault(); setDragging(false); void onImport(event.dataTransfer.files) }
  return <>
    <header><p className="eyebrow">Your knowledge inbox</p><h1>Drop the mess here.</h1><p>Ren accepts PDFs, text files, and Markdown. It finds small pieces worth keeping—without touching your original files.</p></header>
    <div className={`drop-zone ${dragging ? 'dragging' : ''}`} onDragOver={(event) => { event.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={drop}>
      <input id="imports" type="file" multiple accept=".pdf,.txt,.md,.markdown" onChange={(event: ChangeEvent<HTMLInputElement>) => { if (event.target.files?.length) void onImport(event.target.files); event.target.value = '' }} />
      <label htmlFor="imports"><strong>{busy ? 'Adding your files…' : 'Drop files here or choose files'}</strong><span>PDF · TXT · Markdown</span></label>
    </div>
    <section className="source-section"><div className="section-head"><h2>What you dropped</h2><button className="quiet" onClick={() => void onRefresh()}>Refresh</button></div>
      {sources.length === 0 ? <div className="empty">Nothing here yet. Start with a file you saved “for later.”</div> : <div className="source-list">{sources.map((source) => <div className="source-row" key={source.id}><span className={`dot ${source.status}`} /><div><strong>{source.filename}</strong><small>{source.status === 'ready' ? `${source.card_count} knowledge card${source.card_count === 1 ? '' : 's'} created` : source.status}{source.error_message ? ` · ${source.error_message}` : ''}</small></div><a href={`${API}/sources/${source.id}/open`} target="_blank" rel="noopener noreferrer">Open</a></div>)}</div>}
    </section>
  </>
}

function Knowledge({ cards, onSelect }: { cards: Card[]; onSelect: (card: Card) => void }) {
  const [query, setQuery] = useState('')
  const visible = cards.filter((card) => `${card.title} ${card.body} ${card.source.filename}`.toLowerCase().includes(query.toLowerCase()))
  return <>
    <header><p className="eyebrow">Small reusable knowledge</p><h1>My Knowledge</h1><p>These are the useful pieces Ren found inside your files. Search by concept, words, or source.</p></header>
    <input className="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search what you have learned…" />
    <CardGrid cards={visible} onSelect={onSelect} empty="Import a file and Ren will turn it into knowledge cards." />
  </>
}

function Remember({ cards, onSelect }: { cards: Card[]; onSelect: (card: Card) => void }) {
  return <>
    <header><p className="eyebrow">Knowledge resurfacing</p><h1>What should I remember?</h1><p>{cards.length ? `Here ${cards.length === 1 ? 'is' : 'are'} ${cards.length} thing${cards.length === 1 ? '' : 's'} Ren thinks you may want to revisit.` : 'You are caught up. Ren will bring back new knowledge after your next import.'}</p></header>
    <CardGrid cards={cards} onSelect={onSelect} empty="Nothing is due right now." />
  </>
}

function CardGrid({ cards, onSelect, empty }: { cards: Card[]; onSelect: (card: Card) => void; empty: string }) {
  return cards.length === 0 ? <div className="empty">{empty}</div> : <div className="card-grid">{cards.map((card) => <button className="knowledge-card" key={card.id} onClick={() => onSelect(card)}><span className="tag">{card.topic_name || 'Knowledge'}</span><h2>{card.title}</h2><p>{card.body}</p><small>Source: {card.source.filename}</small></button>)}</div>
}

function CardDetail({ card, onClose, onReview, onArchive }: { card: Card; onClose: () => void; onReview: (card: Card) => Promise<void>; onArchive: (card: Card) => Promise<void> }) {
  return <div className="overlay" onMouseDown={onClose}><article className="detail" onMouseDown={(event) => event.stopPropagation()}><button className="close" onClick={onClose}>×</button><span className="tag">{card.topic_name || 'Knowledge'}</span><h1>{card.title}</h1><p>{card.body}</p><a href={`${API}/sources/${card.source.id}/open`} target="_blank" rel="noopener noreferrer">Source: {card.source.filename}</a><small>{card.review_count ? `Reviewed ${card.review_count} time${card.review_count === 1 ? '' : 's'}` : 'New knowledge'}</small><div className="actions"><button className="quiet danger" onClick={() => void onArchive(card)}>Not useful</button><button className="primary" onClick={() => void onReview(card)}>I reviewed this</button></div></article></div>
}
