import { ChangeEvent, DragEvent, useCallback, useEffect, useState } from 'react'
import { ChatWorkspace } from './components/chat-workspace'
import { SettingsWorkspace } from './components/settings-workspace'

const API = 'http://127.0.0.1:8765/api'

type ViewName = 'inbox' | 'knowledge' | 'remember' | 'chat' | 'settings'
type Source = {
  id: string
  filename: string
  source_kind: string
  status: string
  error_message?: string
  topics: string[]
  card_count: number
}
type Card = {
  id: string
  title: string
  body: string
  topic_name?: string
  review_count: number
  last_seen_at?: string
  next_review_at: string
  source: { id: string; filename: string }
}

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, init)
  if (!response.ok) throw new Error((await response.text()) || `Request failed (${response.status})`)
  return response.json() as Promise<T>
}

export default function App() {
  const [view, setView] = useState<ViewName>('inbox')
  const [theme, setTheme] = useState<'dark' | 'light'>(() => {
    try {
      const saved = localStorage.getItem('color-scheme') || localStorage.getItem('ren-theme')
      if (saved === 'light' || saved === 'dark') return saved
      return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
    } catch {
      return 'light'
    }
  })
  const [sources, setSources] = useState<Source[]>([])
  const [cards, setCards] = useState<Card[]>([])
  const [remember, setRemember] = useState<Card[]>([])
  const [selected, setSelected] = useState<Card | null>(null)
  const [notice, setNotice] = useState('')
  const [busy, setBusy] = useState(false)

  const refresh = useCallback(async () => {
    try {
      const [nextSources, nextCards, nextRemember] = await Promise.all([
        api<Source[]>('/sources'),
        api<Card[]>('/cards'),
        api<Card[]>('/remember?limit=5'),
      ])
      setSources(nextSources)
      setCards(nextCards)
      setRemember(nextRemember)
    } catch (error) {
      setNotice(`Ren's local service is unavailable: ${(error as Error).message}`)
    }
  }, [])

  useEffect(() => {
    void refresh()
    const timer = window.setInterval(() => void refresh(), 4000)
    return () => window.clearInterval(timer)
  }, [refresh])

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    const meta = document.querySelector<HTMLMetaElement>('meta[name="color-scheme"]')
    if (meta) meta.content = theme
    try {
      localStorage.setItem('color-scheme', theme)
      localStorage.setItem('ren-theme', theme)
    } catch {
      // Ignore storage errors
    }
  }, [theme])

  async function importFiles(files: FileList | File[]) {
    if (!files.length) return
    setBusy(true)
    try {
      const form = new FormData()
      Array.from(files).forEach((file) => form.append('files', file))
      const response = await fetch(`${API}/captures/files`, { method: 'POST', body: form })
      if (!response.ok) throw new Error(await response.text())
      const result = (await response.json()) as Array<{ status: string; filename?: string }>
      const unsupported = result.filter((item) => item.status === 'unsupported')
      setNotice(
        unsupported.length
          ? `${unsupported.map((item) => item.filename).join(', ')} skipped. Ren accepts PDF, TXT, and Markdown.`
          : 'Imported into Inbox. Ren is extracting knowledge cards.',
      )
      await refresh()
    } catch (error) {
      setNotice(`Import failed: ${(error as Error).message}`)
    } finally {
      setBusy(false)
    }
  }

  async function review(card: Card) {
    try {
      const result = await api<{ next_review_at: string }>(`/cards/${card.id}/review`, { method: 'POST' })
      setNotice(`Reviewed. Ren will bring “${card.title}” back on ${new Date(result.next_review_at).toLocaleDateString()}.`)
      setSelected(null)
      await refresh()
    } catch (error) {
      setNotice((error as Error).message)
    }
  }

  async function archive(card: Card) {
    try {
      await api(`/cards/${card.id}/archive`, { method: 'POST' })
      setSelected(null)
      setNotice(`Archived “${card.title}”.`)
      await refresh()
    } catch (error) {
      setNotice((error as Error).message)
    }
  }

  return (
    <main className={`app-shell ${view === 'inbox' ? 'is-home' : 'is-inner-page'}`}>
      {view !== 'inbox' && <div className="inner-wave-bg" aria-hidden="true" />}

      <aside className="sidebar">
        <div className="brand" onClick={() => setView('inbox')} role="button" tabIndex={0}>
          <img src="/ren-logo.png" alt="Ren logo" className="brand-logo" />
          <div>
            <strong>REN</strong>
            <small>Knowledge Field</small>
          </div>
        </div>

        <nav>
          <button className={view === 'inbox' ? 'active' : ''} onClick={() => setView('inbox')}>
            <NavIcon name="inbox" />
            <span>Inbox</span>
            <b>{sources.length}</b>
          </button>
          <button className={view === 'chat' ? 'active' : ''} onClick={() => setView('chat')}>
            <NavIcon name="chat" />
            <span>Ask Ren</span>
          </button>
          <button className={view === 'knowledge' ? 'active' : ''} onClick={() => setView('knowledge')}>
            <NavIcon name="knowledge" />
            <span>Knowledge</span>
            <b>{cards.length}</b>
          </button>
          <button className={view === 'remember' ? 'active' : ''} onClick={() => setView('remember')}>
            <NavIcon name="remember" />
            <span>Remember</span>
            <b>{remember.length}</b>
          </button>
          <button className={view === 'settings' ? 'active' : ''} onClick={() => setView('settings')}>
            <NavIcon name="settings" />
            <span>Settings</span>
          </button>
        </nav>

        <div className="sidebar-bottom">
          <div className="theme-seg" role="group" aria-label="Color theme">
            <button
              type="button"
              className={theme === 'light' ? 'seg-btn active' : 'seg-btn'}
              onClick={() => setTheme('light')}
              aria-label="Light mode"
            >
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <circle cx="12" cy="12" r="4" />
                <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41" />
              </svg>
              <span>Light</span>
            </button>
            <button
              type="button"
              className={theme === 'dark' ? 'seg-btn active' : 'seg-btn'}
              onClick={() => setTheme('dark')}
              aria-label="Dark mode"
            >
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
              </svg>
              <span>Dark</span>
            </button>
          </div>
          <p className="rule">
            <strong>REN RULE</strong>
            <br />
            Never make me organize anything.
          </p>
        </div>
      </aside>

      <section className="page" key={view}>
        {notice && (
          <div className="notice">
            <span>{notice}</span>
            <button onClick={() => setNotice('')} aria-label="Dismiss notification">
              ×
            </button>
          </div>
        )}
        {view === 'inbox' && (
          <Inbox
            sources={sources}
            cardsCount={cards.length}
            rememberCount={remember.length}
            busy={busy}
            onImport={importFiles}
            onRefresh={refresh}
            onNavigate={setView}
          />
        )}
        {view === 'chat' && <ChatWorkspace onNotice={setNotice} />}
        {view === 'knowledge' && <Knowledge cards={cards} onSelect={setSelected} />}
        {view === 'remember' && <Remember cards={remember} onSelect={setSelected} />}
        {view === 'settings' && <SettingsWorkspace onNotice={setNotice} />}
      </section>

      {selected && (
        <CardDetail
          card={selected}
          onClose={() => setSelected(null)}
          onReview={review}
          onArchive={archive}
        />
      )}
    </main>
  )
}

function NavIcon({ name }: { name: ViewName }) {
  if (name === 'chat')
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M20 11.5a7.5 7.5 0 0 1-8 7.5 8.5 8.5 0 0 1-3.2-.6L4 20l1.4-3.7A7.2 7.2 0 0 1 4 11.5 7.5 7.5 0 0 1 12 4a7.5 7.5 0 0 1 8 7.5Z" />
      </svg>
    )
  if (name === 'inbox')
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M4 5h16v14H4zM4 14h5l1.5 2h3L15 14h5" />
      </svg>
    )
  if (name === 'knowledge')
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M12 3a8 8 0 0 0-4.7 14.5c.7.5.7 1.2.7 2h8c0-.8 0-1.5.7-2A8 8 0 0 0 12 3ZM9 22h6" />
      </svg>
    )
  if (name === 'settings')
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <circle cx="12" cy="12" r="3" />
        <path d="M19.4 15a1.8 1.8 0 0 0 .36 2l.06.06-2.1 2.1-.06-.06a1.8 1.8 0 0 0-2-.36 1.8 1.8 0 0 0-1.1 1.65v.11h-3v-.11A1.8 1.8 0 0 0 10.5 18.7a1.8 1.8 0 0 0-2 .36l-.06.06-2.1-2.1.06-.06a1.8 1.8 0 0 0 .36-2 1.8 1.8 0 0 0-1.65-1.1H5v-3h.11A1.8 1.8 0 0 0 6.8 9.8a1.8 1.8 0 0 0-.36-2l-.06-.06 2.1-2.1.06.06a1.8 1.8 0 0 0 2 .36A1.8 1.8 0 0 0 11.6 4.4v-.11h3v.11a1.8 1.8 0 0 0 1.1 1.65 1.8 1.8 0 0 0 2-.36l.06-.06 2.1 2.1-.06.06a1.8 1.8 0 0 0-.36 2 1.8 1.8 0 0 0 1.65 1.1h.11v3h-.11A1.8 1.8 0 0 0 19.4 15Z" />
      </svg>
    )
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M18 9a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9M10 22h4" />
    </svg>
  )
}

function Inbox({
  sources,
  cardsCount,
  rememberCount,
  busy,
  onImport,
  onRefresh,
  onNavigate,
}: {
  sources: Source[]
  cardsCount: number
  rememberCount: number
  busy: boolean
  onImport: (files: FileList | File[]) => Promise<void>
  onRefresh: () => Promise<void>
  onNavigate: (view: ViewName) => void
}) {
  const [started, setStarted] = useState(false)
  const [dragging, setDragging] = useState(false)

  function drop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault()
    setDragging(false)
    void onImport(event.dataTransfer.files)
  }

  return (
    <div className="hero-stage">
      {/* Top Minimal Status Bar */}
      <div className="hero-topbar">
        <div className="hero-pill">
          <span>Get started by dropping</span>
          <code>PDF · TXT · MD</code>
        </div>
        <div className="hero-byline">
          <span>By</span>
          <span className="hero-mark">▲</span>
          <strong>Tanveer Ziad</strong>
        </div>
      </div>

      {/* Center Stage: Wordmark lifts smoothly when Get Started is clicked, Upload zone blur-reveals */}
      <div className={`hero-center ${started ? 'is-started' : ''}`}>
        <div className="hero-halo" aria-hidden="true" />

        <div className="hero-title-group">
          <h1 className="ren-wordmark" aria-label="REN">
            <span className="ren-letters">REN</span>
          </h1>
          <p className="hero-tagline">
            Drop the mess. Extract atomic cards. Ask with grounded citations.
          </p>

          <div className="hero-cta-wrap">
            <button
              type="button"
              className="get-started-btn"
              onClick={() => setStarted(true)}
            >
              <span>Get Started</span>
              <i aria-hidden="true">→</i>
            </button>
          </div>
        </div>

        {/* Blur-Reveal Upload Panel */}
        <div className="upload-reveal-panel" aria-hidden={!started}>
          <div
            className={`minimal-drop-zone ${dragging ? 'dragging' : ''}`}
            onDragOver={(event) => {
              event.preventDefault()
              setDragging(true)
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={drop}
          >
            <input
              id="imports"
              type="file"
              multiple
              accept=".pdf,.txt,.md,.markdown"
              onChange={(event: ChangeEvent<HTMLInputElement>) => {
                if (event.target.files?.length) void onImport(event.target.files)
                event.target.value = ''
              }}
            />
            <label htmlFor="imports">
              <span className="upload-icon">↑</span>
              <strong>{busy ? 'Indexing files…' : 'Upload Files'}</strong>
              <span className="upload-meta">Drop PDF · TXT · Markdown or click to browse</span>
            </label>
          </div>
        </div>
      </div>

      {/* Bottom 4-Column Feature Grid */}
      <div className="hero-grid">
        <button type="button" className="hero-feature-card" style={{ '--i': 0 } as React.CSSProperties} onClick={() => setStarted(true)}>
          <h2>
            Capture <span>→</span>
          </h2>
          <p>Drop raw PDFs, notes, and text files into your local-first knowledge field.</p>
        </button>

        <button type="button" className="hero-feature-card" style={{ '--i': 1 } as React.CSSProperties} onClick={() => onNavigate('knowledge')}>
          <h2>
            Knowledge <span>→</span>
          </h2>
          <p>Explore {cardsCount} atomic concept cards distilled automatically from your sources.</p>
        </button>

        <button type="button" className="hero-feature-card" style={{ '--i': 2 } as React.CSSProperties} onClick={() => onNavigate('remember')}>
          <h2>
            Remember <span>→</span>
          </h2>
          <p>Revisit {rememberCount} due insights on a 1 → 3 → 7 → 14 → 30 day retention loop.</p>
        </button>

        <button type="button" className="hero-feature-card" style={{ '--i': 3 } as React.CSSProperties} onClick={() => onNavigate('chat')}>
          <h2>
            Grounded Chat <span>→</span>
          </h2>
          <p>Ask Ren with hybrid retrieval, clickable [S1] citations, and confidence gating.</p>
        </button>
      </div>

      {/* Imported Sources List */}
      {sources.length > 0 && (
        <section className="source-section">
          <div className="section-head">
            <h2>Local Library ({sources.length})</h2>
            <button className="quiet" onClick={() => void onRefresh()}>
              Refresh
            </button>
          </div>
          <div className="source-list">
            {sources.map((source, idx) => (
              <div
                className="source-row"
                key={source.id}
                style={{ '--i': idx } as React.CSSProperties}
              >
                <span className={`dot ${source.status}`} />
                <div>
                  <strong>{source.filename}</strong>
                  <small>
                    {source.status === 'ready'
                      ? `${source.card_count} knowledge card${source.card_count === 1 ? '' : 's'} created`
                      : source.status}
                    {source.error_message ? ` · ${source.error_message}` : ''}
                  </small>
                </div>
                <a href={`${API}/sources/${source.id}/open`} target="_blank" rel="noopener noreferrer">
                  Open ↗
                </a>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  )
}

function Knowledge({ cards, onSelect }: { cards: Card[]; onSelect: (card: Card) => void }) {
  const [query, setQuery] = useState('')
  const visible = cards.filter((card) =>
    `${card.title} ${card.body} ${card.source.filename}`.toLowerCase().includes(query.toLowerCase()),
  )
  return (
    <div className="view-fade">
      <header className="minimal-header">
        <p className="eyebrow">02 / Organize</p>
        <h1>Knowledge</h1>
        <p>Atomic, reusable pieces extracted from your files. Search by concept, phrase, or source.</p>
      </header>
      <div className="search-wrap">
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <circle cx="10.8" cy="10.8" r="6.4" />
          <path d="m16 16 4.3 4.3" />
        </svg>
        <input
          className="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Filter knowledge cards…"
        />
      </div>
      <CardGrid cards={visible} onSelect={onSelect} empty="Import a file in Inbox and Ren will distill it into knowledge cards." />
    </div>
  )
}

function Remember({ cards, onSelect }: { cards: Card[]; onSelect: (card: Card) => void }) {
  return (
    <div className="view-fade">
      <header className="minimal-header">
        <p className="eyebrow">03 / Resurface</p>
        <h1>Remember</h1>
        <p>
          {cards.length
            ? `${cards.length} card${cards.length === 1 ? '' : 's'} ready for spaced review today.`
            : 'You are caught up. Ren will resurface cards as their review interval arrives.'}
        </p>
      </header>
      <CardGrid cards={cards} onSelect={onSelect} empty="Nothing is due right now." />
    </div>
  )
}

function CardGrid({
  cards,
  onSelect,
  empty,
}: {
  cards: Card[]
  onSelect: (card: Card) => void
  empty: string
}) {
  return cards.length === 0 ? (
    <div className="empty">{empty}</div>
  ) : (
    <div className="card-grid">
      {cards.map((card, index) => (
        <button
          className="knowledge-card"
          key={card.id}
          style={{ '--i': Math.min(index, 12) } as React.CSSProperties}
          onClick={() => onSelect(card)}
        >
          <div className="card-top">
            <span className="tag">{card.topic_name || 'Knowledge'}</span>
            <span className="card-index">{String(index + 1).padStart(2, '0')}</span>
          </div>
          <h2>{card.title}</h2>
          <p>{card.body}</p>
          <small>
            <span>{card.source.filename}</span>
            <i>→</i>
          </small>
        </button>
      ))}
    </div>
  )
}

function CardDetail({
  card,
  onClose,
  onReview,
  onArchive,
}: {
  card: Card
  onClose: () => void
  onReview: (card: Card) => Promise<void>
  onArchive: (card: Card) => Promise<void>
}) {
  return (
    <div className="overlay" onMouseDown={onClose}>
      <article className="detail" onMouseDown={(event) => event.stopPropagation()}>
        <button className="close" onClick={onClose} aria-label="Close modal">
          ×
        </button>
        <span className="tag">{card.topic_name || 'Knowledge'}</span>
        <h1>{card.title}</h1>
        <p>{card.body}</p>
        <a href={`${API}/sources/${card.source.id}/open`} target="_blank" rel="noopener noreferrer">
          Source: {card.source.filename} ↗
        </a>
        <small>
          {card.review_count
            ? `Reviewed ${card.review_count} time${card.review_count === 1 ? '' : 's'}`
            : 'New knowledge card'}
        </small>
        <div className="actions">
          <button className="quiet danger" onClick={() => void onArchive(card)}>
            Archive
          </button>
          <button className="primary" onClick={() => void onReview(card)}>
            Mark Reviewed →
          </button>
        </div>
      </article>
    </div>
  )
}
