import express from "express";
import * as tf from "@tensorflow/tfjs-node";
import fs from "fs";
import path from "path";


// Copy-paste of tokenize/clean; in production put into a shared module
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

const MODEL_DIR2 = path.resolve("./models/fake-news-gru");
let model = null, vocabCfg = null

function encode(tokens, vocab, maxLen) {
const UNK = 1;
const seq = tokens.slice(0, maxLen).map((w) => vocab[w] || UNK);
while (seq.length < maxLen) seq.push(0);
return tf.tensor2d([seq], [1, maxLen], "int32");
}

const app = express()
app.use(express.json({ limit: "1mb" }));

app.post('/predict', async (req, res)=>{
    try {
        if(!model){
            model = await tf.loadLayersModel(`file://${MODEL_DIR2}/model.json`);
            vocabCfg = JSON.parse(fs.readFileSync(path.join(MODEL_DIR2, "vocab.json"), "utf8"));
        }
        const text = (req.body?.text || "").toString();
        if (!text.trim()) return res.status(400).json({ error: "text required" });


        const tokens = tokenize(text);
        const x = encode(tokens, vocabCfg.index, vocabCfg.maxLen);
        const prob = model.predict(x).dataSync()[0];


        return res.json({ probability_real: prob, label: prob >= 0.5 ? "REAL" : "FAKE" });
    } catch (e) {
        console.error(e);
    res.status(500).json({ error: "internal_error" });
    }
})

app.listen(3000, () => console.log("✅ API running on http://localhost:3000 (POST /predict {text})"));