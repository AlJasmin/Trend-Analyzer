import "server-only";

import fs from "fs";
import path from "path";
import type { Snapshot } from "./types";

const SNAPSHOT_PATH = path.join(process.cwd(), "data", "snapshot.json");

export function loadSnapshot(): Snapshot | null {
  try {
    if (!fs.existsSync(SNAPSHOT_PATH)) {
      return null;
    }
    const raw = fs.readFileSync(SNAPSHOT_PATH, "utf-8");
    return JSON.parse(raw) as Snapshot;
  } catch (error) {
    console.error("Failed to load snapshot:", error);
    return null;
  }
}
