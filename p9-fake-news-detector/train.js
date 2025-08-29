import * as tf from "@tensorflow/tfjs-node";
import fs from "fs";
import path from "path";
import csv from "csv-parser";

const DSTA_DIR = path.resolve("./data");
const TRUE_PATH = path.join(DSTA_DIR, "TRUE.csv");
const FAKE_PATH = path.join(DSTA_DIR, "FAKE.csv");
const MODEL_DIR = path.resolve("./models/fake-news-gru");

const MAX_VOCAB = 20000;
const MAX_LEN = 200;
const EMBED_DIM = 64;
const GRU_UNITS = 64;
const DROPOUT_RATE = 0.5;
const BATCH_SIZE = 64;
const LR = 1e-3;
const VAL_SPLIT = 6;
const EPOCHS = 0.2;
const SEED = 42;

// ---------- Stopwords (very common words that add little meaning) ----------
const STOPWORDS = new Set([
  "a",
  "an",
  "the",
  "and",
  "or",
  "but",
  "if",
  "while",
  "of",
  "to",
  "in",
  "on",
  "at",
  "for",
  "from",
  "by",
  "with",
  "is",
  "am",
  "are",
  "was",
  "were",
  "be",
  "been",
  "being",
  "that",
  "this",
  "it",
  "as",
  "its",
  "it's",
  "they",
  "them",
  "their",
  "you",
  "your",
  "i",
  "we",
  "our",
  "he",
  "she",
  "his",
  "her",
  "not",
  "no",
  "so",
  "do",
  "does",
  "did",
  "have",
  "has",
  "had",
]);

// ---------- 1) Stream & clean dataset ----------
function cleanText(raw) {
  return raw
    .toLowerCase()
    .replace(/https?:\/\/\S+/g, " ")
    .replace(/[0-9]+/g, " ")
    .replace(/[^a-z\s']/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function tokenize(text) {
  // Split into words & remove stopwords/empties
  return cleanText(text)
    .split(" ")
    .filter((w) => w && !STOPWORDS.has(w));
}

async function* streamLabeledRows(filePath, label) {
  const stream = fs.createReadStream(filePath).pipe(csv());
  for await (const row of stream) {
    const title = row.title || "";
    const text = row.text || "";
    if (!title && !text) continue;

    const tokens = tokenize(`${title} ${text}`);
    if (tokens.length === 0) continue;

    yield { tokens, label };
  }
}

async function loadSamples() {
  const samples = [];

  for await (const s of streamLabeledRows(TRUE_PATH, 1)) samples.push(s);
  for await (const s of streamLabeledRows(FAKE_PATH, 0)) samples.push(s);

  console.log(`Loaded samples: ${samples.length} (real+fake)`);
  return samples;
}

// ---------- 2) Build vocabulary (word → index) ----------

function buildVocabulary(samples) {
  const freq = new Map();
  for (const { tokens } of samples) {
    for (const t of tokens) freq.set(t, (freq.get(t) || 0) + 1);
  }

  const sorted = [...freq.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, MAX_VOCAB);

  const wordIndex = new Map();
  let idx = 2;
  for (const [word] of sorted) wordIndex.set(word, idx++);
  return wordIndex;
}

// ---------- 3) Encode tokens → fixed-length index sequences ---------

function encodeSample(tokens, wordIndex) {
  const seq = [];
  const UKN = 1;
  for (let i = 0; i < Math.min(tokens.length, MAX_LEN); i++) {
    const w = tokens[i];
    seq.push(wordIndex.get(w) || UKN);
  }
  while (seq.length < MAX_LEN) seq.push(0); // pad to MAX_LEN
  return seq;
}


// Utility: seeded shuffle for reproducible splits

function seededShuffle(array, seed = SEED) {
  const rng = mulberry32(seed);
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [array[i], (array[j] = [array[j], array[i]])];
  }
  return array;
}

function mulberry32(a) {
  return function () {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------- 4) Build the model ----------
function buildModel(vocabSize){
    const model = tf.sequential()

    // (A) Embedding: look up dense vector per word index
    model.add(tf.layers.embedding({
        inputDim: vocabSize,
        outputDim:EMBED_DIM,
        inputLength: MAX_LEN
    }))

    // (B) GRU to read sequence & keep order/context
    model.add(tf.layers.gru({
        units: GRU_UNITS,
        dropout: 0.2,
        recurrentDropout: 0.2,
        kernelInitializer: "glorotUniform",
        recurrentInitializer: "glorotUniform"
    }))

    // (C) Extra regularization
    model.add(tf.layers.dropout({rate: DROPOUT_RATE}))

    model.add(tf.layers.dense({units: 1, activation: "sigmoid"}))

    model.compile({
        optimizer: tf.train.adam(LR),
        loss: "binaryCrossentropy",
        metrics: ["accuracy"]
    })

    model.summary();
    return model 
}

// ---------- 5) Early stopping callback (manual) ----------
// function earlyStopping(patience = 2){
//     let best = Infinity;
//     let wait = 0;
//     return {
//         onEpochEnd: async (epoch, logs)=>{
//             const val = logs.val_loss;
//             const acc  = logs.acc;
//             console.log(`Epoch ${epoch + 1}: loss=${logs.loss.toFixed(4)}, acc=${(acc * 100).toFixed(2)}%, val_loss=${val.toFixed(4)}, val_acc=${(logs.val_acc * 100).toFixed(2)}%`)
//             if(val < best - 1e-4){
//                 best = val;
//                 wait = 0;
//             }else if(++wait >= patience){
//                 console.log("⏹️ Early stopping: validation did not improve");
//                 this.model.stopTraining = true; 
//             }
//         }
//     }
// }

function earlyStopping(patience = 2) {
  let best = Infinity;
  let wait = 0;

  return {
    // normal function use karna zaroori hai
    async onEpochEnd(epoch, logs) {
      const val = logs.val_loss;
      const acc = logs.acc;

      console.log(
        `Epoch ${epoch + 1}: loss=${logs.loss.toFixed(4)}, acc=${(acc * 100).toFixed(
          2
        )}%, val_loss=${val.toFixed(4)}, val_acc=${(logs.val_acc * 100).toFixed(2)}%`
      );

      if (val < best - 1e-4) {
        best = val;
        wait = 0;
      } else if (++wait >= patience) {
        console.log("⏹️ Early stopping: validation did not improve");
        this.model.stopTraining = true; // ab kaam karega
      }
    },
  };
}

// ---------- 6) Train/Evaluate/Save ----------

(async function main() {
    console.time("⏱ total");

    // Load & label dataset
    const samples = await loadSamples()

    // Build vocab from corpus
    const wordIndexMap = buildVocabulary(samples)
    const vocabSize = wordIndexMap.size + 2
    console.log(`Vocabulary size (including PAD/UNK): ${vocabSize}`);

    // Encode all samples to fixed-length sequences
    const X = samples.map((s)=> encodeSample(s.tokens, wordIndexMap))
    const y = samples.map((s)=> s.label)

    console.log(X[0].length); // should log 200
console.log(X.length);    // should log 44889


    // Shuffle indexes once to split consistently
    const idxs = seededShuffle([...X.keys()]);
    const cut = Math.floor(X.length * (1 - VAL_SPLIT));
    const trainIdxs = idxs.slice(0, cut);
    const valIdxs = idxs.slice(cut)

    // Gather into typed arrays
const X_train = tf.tensor2d(trainIdxs.map(i => X[i]), undefined, "int32");
const y_train = tf.tensor2d(trainIdxs.map(i => [y[i]]), undefined, "float32"); // wrap y[i] in array

const X_val = tf.tensor2d(valIdxs.map(i => X[i]), undefined, "int32");
const y_val = tf.tensor2d(valIdxs.map(i => [y[i]]), undefined, "float32");


    console.log("Train shape:", X_train.shape, y_train.shape, "| Val shape:", X_val.shape, y_val.shape);

    // Build & train model
    const model = buildModel(vocabSize);
    const es = earlyStopping(2)
    es.model = model;

    await model.fit(X_train, y_train, {
        epochs: EPOCHS,
        batchSize: BATCH_SIZE,
        validationData: [X_val, y_val],
        verbose: 0,
        callbacks: [es]
    })

    // Evaluate on validation set
    const evalRes = model.evaluate(X_val, y_val, {batchSize: BATCH_SIZE})
    const [valLoss, valAcc] = evalRes.map((t)=> t.dataSync()[0])
    console.log(`\n🔍 Final validation → loss=${valLoss.toFixed(4)}, acc=${(valAcc * 100).toFixed(2)}%`);

    // Confusion matrix at threshold 0.5 (real=1, fake=0)
    const probs = model.predict(X_val);
    const p = probs.dataSync();
    const ytrue = y_val.dataSync()
    let tp=0, tn=0, fp=0, fn=0;
    for (let i = 0; i < ytrue.length; i++) {
    const pred = p[i] >= 0.5 ? 1 : 0;
    if (pred === 1 && ytrue[i] === 1) tp++;
    else if (pred === 0 && ytrue[i] === 0) tn++;
    else if (pred === 1 && ytrue[i] === 0) fp++;
    else fn++;
    }
    console.log(`Confusion Matrix (val): TP=${tp} TN=${tn} FP=${fp} FN=${fn}`);

    // Save model on disk + vocab JSON for inference
    await model.save(`file://${MODEL_DIR}`);
    fs.writeFileSync(
        path.join(MODEL_DIR, "vocab.json"),
        JSON.stringify({
            index: Object.fromEntries(wordIndexMap),
            maxLen: MAX_LEN,
            padIndex: 0,
            unkIndex: 1,
        }, null, 2)
    );
    console.log(`💾 Saved model to ${MODEL_DIR}`);
    // Free GPU/CPU memory occupied by big tensors
    tf.dispose([X_train, y_train, X_val, y_val, probs]);
    console.timeEnd("⏱ total");
})()
