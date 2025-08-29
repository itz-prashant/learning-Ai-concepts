import * as tf from "@tensorflow/tfjs-node"
import fs from "fs"
import csv from "csv-parser"

// -------------------- STEP 1: Load Dataset --------------------
const DATASET_PATH = "./data/spam.csv"

async function loadDataset() {
    return new Promise((resolve, reject)=>{
        const results = [];
        fs.createReadStream(DATASET_PATH)
        .pipe(csv())
        .on("data", (row)=>{
            results.push({
                label: row.label.trim(),
                text: row.text.trim()
            });
        })
        .on("end", ()=>{
        console.log(`Loaded dataset with ${results.length} rows`);
        resolve(results);
        })
        .on("error", reject)
    })
}

// -------------------- STEP 2: Preprocess Text --------------------
/**
 * Goal: Convert raw SMS text → numeric tensors
 * Steps:
 * 1. Lowercase
 * 2. Remove punctuation
 * 3. Tokenize (split words)
 * 4. Map each word → unique number (vocab index)
 */

const MAX_VOCAB_SIZE = 5000;
const MAX_SEQUENCE_LEN = 20;

let wordIndex = {}

function buildVocabulary(texts){
    const freq = {}

    texts.forEach(text => {
        text.toLowerCase().replace(/[^a-z\s]/g, "").split(/\s+/).forEach((word)=>{
            if(!word) return;
            freq[word] = (freq[word] || 0) + 1
        })  
    });
    // Sort by frequency & keep top words
    const sorted = Object.entries(freq).sort((a,b)=> b[1] - a[1]).slice(0, MAX_VOCAB_SIZE)
    wordIndex = {};
    sorted.forEach(([word], i)=>{
        wordIndex[word] = i + 1
    })
    console.log(`📖 Vocabulary built with ${Object.keys(wordIndex).length} words`);
    // console.log("Wi",wordIndex.slice(0,5))
}

function textToSequence(text){
     const words = text
    .toLowerCase()
    .replace(/[^a-z\s]/g, "")
    .split(/\s+/);

    const seq = words.map((w)=> wordIndex[w] || 1); // unknown words → 1

    // Pad/trim sequence to fixed length
    if(seq.length > MAX_SEQUENCE_LEN){
        return seq.slice(0, MAX_SEQUENCE_LEN)
    }
    while(seq.length < MAX_SEQUENCE_LEN){
        seq.push(0)
    }
    return seq;
}

// -------------------- STEP 3: Prepare Training Data --------------------

function prepareData(dataset){
    const texts = dataset.map(d=> d.text)
    const labels = dataset.map(d=> d.label === 'spam' ? 1 : 0)

    buildVocabulary(texts)

    const sequence = texts.map(textToSequence)

    const xs = tf.tensor2d(sequence, [sequence.length, MAX_SEQUENCE_LEN])
    const ys = tf.tensor2d(labels, [labels.length, 1])

    return {xs, ys}
}

// -------------------- STEP 4: Build Model --------------------

function createModel(){
    const model = tf.sequential()

    model.add(tf.layers.embedding({
        inputDim: MAX_VOCAB_SIZE +1,
        outputDim: 16,
        inputLength: MAX_SEQUENCE_LEN
    }))

    model.add(tf.layers.flatten());

    model.add(tf.layers.dense({
        units:16,
        activation: "relu"
    }))

    model.add(tf.layers.dense({
        units:1,
        activation: "sigmoid"
    }))

    model.compile({
        optimizer: tf.train.adam(0.001),
        loss: "binaryCrossentropy",
        metrics: ["accuracy"]
    })

    model.summary()
    return model
}

// -------------------- STEP 5: Train & Evaluate -------------------

async function run() {
    const dataset = await loadDataset()

    const {xs, ys} = prepareData(dataset)

    const model = createModel()

    await model.fit(xs, ys, {
        epochs: 5,
        batchSize: 32,
        validationData: 0.2,
        callbacks:{
            onEpochEnd: (epochs, logs)=>{
                console.log(`Epoch ${epochs + 1}: loss=${logs.loss.toFixed(4)}, acc=${(logs.acc *100).toFixed(4)}%`)
            }
        }
    })

    // Test on custom SMS
    const testSMS = "Congratulations! You have won a free ticket!";
    const seq = tf.tensor2d([textToSequence(testSMS)], [1, MAX_SEQUENCE_LEN])
    const predection = model.predict(seq);
    predection.print();
}

run()