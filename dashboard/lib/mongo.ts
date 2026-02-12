import "server-only";

import { MongoClient } from "mongodb";
import { loadConfig } from "./config";

type GlobalMongo = typeof globalThis & {
  _mongoClient?: MongoClient;
};

const globalForMongo = globalThis as GlobalMongo;

export async function getDb() {
  if (!globalForMongo._mongoClient) {
    const { mongodb } = loadConfig();
    globalForMongo._mongoClient = new MongoClient(mongodb.uri);
    await globalForMongo._mongoClient.connect();
  }

  const { mongodb } = loadConfig();
  return globalForMongo._mongoClient.db(mongodb.database);
}
