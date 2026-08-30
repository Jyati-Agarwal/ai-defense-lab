import { rm } from "node:fs/promises"
import { resolve } from "node:path"

export default async function globalSetup() {
  await rm(resolve("jury-screens", ".evidence-staging"), { recursive: true, force: true })
}
