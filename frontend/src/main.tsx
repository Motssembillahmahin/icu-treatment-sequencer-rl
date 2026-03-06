import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Link, Route, Routes } from 'react-router-dom'
import { Dashboard } from './pages/Dashboard'
import { Training } from './pages/Training'
import { EpisodeViewer } from './pages/EpisodeViewer'
import './index.css'

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-icu-bg text-slate-100">
        {/* Nav */}
        <nav className="border-b border-icu-border bg-icu-card px-6 py-3">
          <div className="max-w-7xl mx-auto flex items-center gap-6">
            <span className="font-bold text-icu-accent">ICU Sequencer RL</span>
            <Link to="/" className="text-sm text-slate-400 hover:text-slate-200 transition-colors">
              Dashboard
            </Link>
            <Link to="/training" className="text-sm text-slate-400 hover:text-slate-200 transition-colors">
              Training
            </Link>
            <Link to="/episodes" className="text-sm text-slate-400 hover:text-slate-200 transition-colors">
              Episodes
            </Link>
          </div>
        </nav>

        {/* Page content */}
        <main className="max-w-7xl mx-auto px-6 py-6">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/training" element={<Training />} />
            <Route path="/episodes" element={<EpisodeViewer />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
