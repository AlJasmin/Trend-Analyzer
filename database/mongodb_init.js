// MongoDB initialization script (example)
// Run manually with: node mongodb_init.js

const MongoClient = require('mongodb').MongoClient;
const url = 'mongodb://localhost:27017';
const dbName = 'global_trend_analyzer';

(async () => {
  const client = new MongoClient(url, { useUnifiedTopology: true });
  try {
    await client.connect();
    console.log('Connected to MongoDB');
    const db = client.db(dbName);
    // Create collections if they do not exist
    await db.createCollection('reddit');
    await db.createCollection('news');
    await db.createCollection('embeddings');
    console.log('Collections created/verified: reddit, news, embeddings');
  } catch (err) {
    console.error(err);
  } finally {
    await client.close();
  }
})();
