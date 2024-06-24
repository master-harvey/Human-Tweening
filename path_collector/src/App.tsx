import { useState, useRef, useEffect } from 'react';
import './App.css'

// This app collects mouse path data
// Randomly places a button on the screen, once clicked the button is replaced randomly somewhere else
// Records the position every ms between clicks, and measures the time between clicks
// Each button click except the first one should result in a record like this: 
//   {"start_timestamp":n,"source":[x,y],"end_timestamp":m,"destination":[x,y],"path":[[x1,y1],[x2,y2],...]}

function App() {
  const [buttonPosition, setButtonPosition] = useState({ x: window.innerWidth / 2 - 9, y: 125 });
  const [lastClickData, setLastClickData] = useState({});

  // Recorded positions of the mouse between clicks. Gets cleared on submit
  const records = useRef([])

  // Recorded positions of the mouse between clicks. Gets cleared after every click
  const path = useRef([])

  window.onmousemove = (event) => {
    //@ts-expect-error // Record mouse position with timestamp
    path.current.push({ x: event.clientX, y: screen.height - event.clientY, t: new Date().valueOf() }) //subtract from screen.height to correct inverted coordinates
  }

  function handleClick(event: any) {
    // Get click timestamp
    const t = new Date().valueOf()

    //@ts-expect-error // Record data if possible
    if (lastClickData.x && lastClickData.y && lastClickData.t && path.current.length > 5) {
      //@ts-expect-error // Create a new record
      records.current.push({
        //@ts-expect-error // Record large vector
        "start_timestamp": lastClickData.t, "source": [lastClickData.x, screen.height - lastClickData.y], //subtract from screen.height to correct inverted coordinates
        "end_timestamp": t, "destination": [event.pageX, screen.height - event.pageY], //subtract from screen.height to correct inverted coordinates
        // Record short vectors
        "path": path.current
      })
      path.current = []
    }

    // New previous mouse position
    setLastClickData({ x: event.pageX, y: event.pageY, t: t })

    // New button position
    const x = Math.floor(Math.random() * (window.innerWidth - 60));
    const y = Math.floor(Math.random() * (window.innerHeight - 60));
    setButtonPosition({ x, y })

    return { x, y };
  }

  // Sends the records as JSON to the endpoint
  async function recordData(data: any) {
    // process is defined during next.js build
    fetch("https://5vv5u662s6rqzczumz7rx3d3ri0qvvsy.lambda-url.us-east-2.on.aws/", {
      method: "PUT",
      body: JSON.stringify(data),
      headers: { "Content-type": "application/json; charset=UTF-8" }
    })
    records.current = [] // Clear records
  }

  useEffect(() => {
    if (records.current.length >= 5) {
      recordData(records.current)
    }
  });

  return (
    <div className="App">
      <div
        style={{
          position: 'absolute',
          left: buttonPosition.x,
          top: buttonPosition.y,
          padding: 0.25,
          backgroundColor: "#0F0",
          width: 18,
          height: 18,
          borderRadius: 6
        }}
        onClick={handleClick}
      >
      </div>
    </div>
  );
}

export default App;