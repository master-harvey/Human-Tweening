import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <h1 className="heading">Cursor Path Data Collection Mechanism</h1>
    <p className="heading">(Data is recorded when the window is closed)</p>
    <App />
  </React.StrictMode>,
)
