import { FormEvent, useEffect, useState } from 'react'
import './chat-workspace.css'

const API = 'http://127.0.0.1:8765/api'

type Thread = { id: string; title: string; incognito: boolean; updated_at: string }
type Citation = { source_id: string; filename: string; anchor: string }
type Message = { id: string; role: 'user' | 'assistant'; content: string; status: string; citations: Citation[] }

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, init)
  if (!response.ok) throw new Error((await response.text()) || `Request failed (${response.status})`)
  return response.json() as Promise<T>
}

function renderContentWithCitations(content: string, citations?: Citation[]) {
  if (!citations || citations.length === 0) {
    return content
  }
  const parts = content.split(/(\[S\d+\])/g)
  return parts.map((part, index) => {
    const match = part.match(/^\[S(\d+)\]$/)
    if (match) {
      const citationIndex = parseInt(match[1], 10) - 1
      const citation = citations[citationIndex]
      if (citation) {
        const anchorLabel = citation.anchor ? ` — ${citation.anchor}` : ''
        return (
          <a
            key={index}
            className="inline-citation"
            href={`${API}/sources/${citation.source_id}/open`}
            target="_blank"
            rel="noopener noreferrer"
            title={`Open ${citation.filename}${anchorLabel}`}
          >
            {part}
          </a>
        )
      }
    }
    return part
  })
}

export function ChatWorkspace({ onNotice }: { onNotice: (message: string) => void }) {
  const [threads, setThreads] = useState<Thread[]>([])
  const [thread, setThread] = useState<Thread | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [draft, setDraft] = useState('')
  const [thinking, setThinking] = useState<'focused' | 'balanced' | 'creative'>('balanced')
  const [sending, setSending] = useState(false)

  async function loadThreads(selectFirst = false) {
    const next = await request<Thread[]>('/chat/threads')
    setThreads(next)
    if (selectFirst && next[0]) await openThread(next[0])
  }

  async function openThread(next: Thread) {
    const result = await request<{ thread: Thread; messages: Message[] }>(`/chat/threads/${next.id}`)
    setThread(result.thread); setMessages(result.messages)
  }

  useEffect(() => { void loadThreads(true).catch((error) => onNotice((error as Error).message)) }, [])

  async function createThread(incognito = false): Promise<Thread | null> {
    try {
      const next = await request<Thread>('/chat/threads', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ title: incognito ? 'Incognito chat' : 'New chat', incognito }) })
      setThreads((current) => [next, ...current]); await openThread(next)
      return next
    } catch (error) { onNotice((error as Error).message); return null }
  }

  async function send(event: FormEvent) {
    event.preventDefault()
    if (!draft.trim() || sending) return
    let active = thread
    if (!active) {
      active = await createThread()
      if (!active) return
    }
    const text = draft.trim()
    const optimistic: Message = { id: `local-${Date.now()}`, role: 'user', content: text, status: 'complete', citations: [] }
    setMessages((current) => [...current, optimistic]); setDraft(''); setSending(true)
    try {
      const reply = await request<Message & { provider_used: boolean; error?: string | null }>(`/chat/threads/${active.id}/messages`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ content: text, thinking }) })
      setMessages((current) => [...current.filter((message) => message.id !== optimistic.id), optimistic, reply])
      if (reply.error === 'provider_not_configured') onNotice('Add an OpenRouter-compatible key in Settings to activate AI chat.')
      await loadThreads()
    } catch (error) { setMessages((current) => current.filter((message) => message.id !== optimistic.id)); onNotice(`Message failed: ${(error as Error).message}`) } finally { setSending(false) }
  }

  return <div className="chat-workspace">
    <aside className="chat-rail"><div className="chat-rail-head"><h2>Chats</h2><button className="icon-button" aria-label="New chat" onClick={() => void createThread()}>+</button></div><button className="incognito-button" onClick={() => void createThread(true)}>◌ New incognito chat</button><div className="thread-list">{threads.length === 0 ? <p>Start a chat to ask Ren about your knowledge.</p> : threads.map((item) => <button key={item.id} className={item.id === thread?.id ? 'thread active' : 'thread'} onClick={() => void openThread(item)}><span>{item.incognito ? '◌ ' : ''}{item.title}</span><small>{new Date(item.updated_at).toLocaleDateString()}</small></button>)}</div></aside>
    <section className="chat-pane"><header><p className="eyebrow">Ren / Grounded chat</p><h1>{thread?.incognito ? 'Incognito chat' : 'Ask Ren'}</h1><p>{thread?.incognito ? 'This thread stays separate from Ren’s long-term memory.' : 'Chat with the context you imported. Ren cites the source it uses.'}</p></header>      <div className="message-list">
        {messages.length === 0 ? (
          <div className="chat-empty">
            <span>◌</span>
            <strong>What are you trying to understand?</strong>
            <p>Ask about an imported concept, compare ideas, or turn a thought into a next step.</p>
          </div>
        ) : (
          messages.map((message) => (
            <article key={message.id} className={`message ${message.role}`}>
              <div className="message-label">{message.role === 'user' ? 'You' : 'Ren'}</div>
              <p>{renderContentWithCitations(message.content, message.citations)}</p>
              {message.citations && message.citations.length > 0 && (
                <div className="message-sources">
                  {message.citations.map((citation, index) => {
                    const anchorLabel = citation.anchor ? ` — ${citation.anchor}` : ''
                    return (
                      <a
                        key={`${citation.source_id}-${index}`}
                        className="citation-pill"
                        href={`${API}/sources/${citation.source_id}/open`}
                        target="_blank"
                        rel="noopener noreferrer"
                        title={`Open ${citation.filename}${anchorLabel}`}
                      >
                        [S{index + 1}] {citation.filename}{anchorLabel}
                      </a>
                    )
                  })}
                </div>
              )}
            </article>
          ))
        )}
      </div><form className="chat-composer" onSubmit={send}><textarea value={draft} onChange={(event) => setDraft(event.target.value)} placeholder="Ask Ren about your knowledge…" rows={3} /><div><select value={thinking} onChange={(event) => setThinking(event.target.value as typeof thinking)} aria-label="Thinking level"><option value="focused">Focused</option><option value="balanced">Balanced</option><option value="creative">Creative</option></select><button className="primary" disabled={sending || !draft.trim()}>{sending ? 'Thinking…' : 'Send ↗'}</button></div></form></section>
  </div>
}
