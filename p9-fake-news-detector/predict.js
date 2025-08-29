import * as tf from "@tensorflow/tfjs-node";
import fs from "fs";
import path from "path";

// Reuse the same cleaning/tokenization to ensure training/inference parity
function cleanText(raw) {
return raw
.toLowerCase()
.replace(/https?:\/\/\S+/g, " ")
.replace(/[0-9]+/g, " ")
.replace(/[^a-z\s']/g, " ")
.replace(/\s+/g, " ")
.trim();
}
const STOPWORDS = new Set(["a","an","the","and","or","but","if","while","of","to","in","on","at","for","from","by","with","is","am","are","was","were","be","been","being","that","this","it","as","its","it's","they","them","their","you","your","i","we","our","he","she","his","her","not","no","so","do","does","did","have","has","had"]);
function tokenize(text) {
return cleanText(text).split(" ").filter((w)=>w && !STOPWORDS.has(w));
}

const MODEL_DIR = path.resolve("./models/fake-news-gru");

function encode(tokens, vocab, maxLen){
    const UKN = 1
    const seq = tokens.slice(0, maxLen).map((w) => vocab[w] || UKN)
    while(seq.length < maxLen) seq.push(0);
    return tf.tensor2d([seq], [1, maxLen], "int32")
}

(async function demo() {
    const model = await tf.loadLayersModel(`file://${MODEL_DIR}/model.json`)

    // Load vocab config
const { index, maxLen } = JSON.parse(fs.readFileSync(path.join(MODEL_DIR, "vocab.json"), "utf8"));


// Example input (try editing this text)
const headline = "Government approves new healthcare reforms to reduce drug prices";
const body = "According to the official press release, the policy will be implemented from next quarter across all states...";
const text = `${headline}. ${body}`;


const tokens = tokenize(text);
const x = encode(tokens, index, maxLen);
const prob = model.predict(x).dataSync()[0];


console.log(`Input: ${text.substring(0, 120)}...`);
console.log(`Prediction (real probability): ${prob.toFixed(4)} → label=${prob >= 0.5 ? "REAL" : "FAKE"}`);
})();