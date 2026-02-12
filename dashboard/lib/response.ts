import { NextResponse } from "next/server";
import { loadSnapshot } from "./snapshot";
import type { Filters, Snapshot } from "./types";

type SnapshotKey = Exclude<keyof Snapshot, "generated_at">;

export async function respondWithFallback<T>(
  key: SnapshotKey,
  filters: Filters,
  run: () => Promise<T>,
  transform?: (data: T) => T,
) {
  const forceSnapshot = process.env.DASHBOARD_SNAPSHOT_ONLY === "1";
  if (!forceSnapshot) {
    try {
      const raw = await run();
      const data = transform ? transform(raw) : raw;
      return NextResponse.json({
        meta: {
          source: "mongo",
          generated_at: new Date().toISOString(),
          filters,
        },
        data,
      });
    } catch (error) {
      console.error(`Mongo query failed for ${key}:`, error);
    }
  }

  const snapshot = loadSnapshot();
  if (snapshot && snapshot[key]) {
    const raw = snapshot[key] as T;
    const data = transform ? transform(raw) : raw;
    return NextResponse.json({
      meta: {
        source: "snapshot",
        generated_at: snapshot.generated_at || new Date().toISOString(),
        filters,
      },
      data,
    });
  }

  return NextResponse.json(
    { error: "Snapshot unavailable." },
    { status: 503 },
  );
}
