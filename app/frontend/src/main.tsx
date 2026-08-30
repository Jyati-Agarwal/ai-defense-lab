import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import { MotionConfig } from "motion/react"
import App from "./App"
import "./index.css"

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <MotionConfig reducedMotion="user" transition={{ duration: 0.32, ease: [0.22, 1, 0.36, 1] }}>
      <App />
    </MotionConfig>
  </StrictMode>,
)
