import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'

import { FaGithub } from "react-icons/fa";

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <h1 className="heading">
      Cursor Path Data Collection Mechanism &nbsp;
      <a href="https://github.com/master-harvey/Human-Tweening" target="_blank" className="github">
        <FaGithub />
      </a>
    </h1>
    <App />
  </React.StrictMode>,
)
