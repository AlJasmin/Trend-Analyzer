import "server-only";

import fs from "fs";
import path from "path";
import yaml from "yaml";

type Config = {
  mongodb: {
    uri: string;
    database: string;
  };
};

let cached: Config | null = null;

export function loadConfig(): Config {
  if (cached) {
    return cached;
  }

  const envUri = process.env.MONGODB_URI;
  const envDb = process.env.MONGODB_DB;
  if (envUri && envDb) {
    cached = { mongodb: { uri: envUri, database: envDb } };
    return cached;
  }

  const configPath = path.join(process.cwd(), "..", "config", "settings.yaml");
  if (!fs.existsSync(configPath)) {
    throw new Error(`Config not found at ${configPath}`);
  }

  const content = fs.readFileSync(configPath, "utf-8");
  const parsed = yaml.parse(content) || {};
  const mongo = parsed.mongodb || {};
  const uri = envUri || mongo.uri;
  const database = envDb || mongo.database;

  if (!uri || !database) {
    throw new Error("MongoDB settings missing (mongodb.uri / mongodb.database).");
  }

  cached = { mongodb: { uri, database } };
  return cached;
}
