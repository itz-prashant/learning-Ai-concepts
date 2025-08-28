import * as tf from "@tensorflow/tfjs-node"
import fs from "fs"
import Papa from "papaparse"

// ----------------------------
// 1. Load dataset (IMDB)
// ----------------------------

const file = fs.readFileSync("./data/IMDBDataset.csv", "utf8")
const parsed = Papa.parse(file, { header: true, skipEmptyLines: true });
const rawData = parsed.data; // [{review:"text", sentiment:"positive/negative"}, ...]

// console.log("Total reviews:", rawData.length); // 50k reviews
// console.log("Sample review:", rawData[0]);


// ----------------------------
// 2. Vocabulary & Tokenization
// ----------------------------

const vocab = {}
let index = 2

let X = []
let y = []

for(let item of rawData){
    const words = item.review.toLowerCase().replace(/[^a-zA-Z ]/g,'').split(" ");
    
    words.forEach(w => {
        if(!vocab[w]) vocab[w] = index++
    })

    X.push(words.map(w=> vocab[w] || 1));
    y.push(item.sentiment === "positive" ? 1 : 0)
}

// console.log("Vocabulary size:", index);
// console.log("Example tokenized review:", X[0]);
// console.log("Label:", y[0]);


// ----------------------------
// 3. Padding
// ----------------------------

const maxLen = 100; // max world per review
function padSeq(seq){
    if(seq.length > maxLen) return seq.slice(0, maxLen) // trucate
    while(seq.length < maxLen) seq.push(0) // padding
    return seq;
}

const X_padded = X.map(padSeq)
// console.log("Padded review example:", X_padded[0]);
// console.log("Padded review example:", X_padded[1]);


// ----------------------------
// 4. Tensor
// ----------------------------

const X_tensor = tf.tensor2d(X_padded, [X_padded.length, maxLen])
const y_tensor = tf.tensor2d(y, [y.length, 1])


// console.log("X_tensor shape:", X_tensor.shape); // [50000,100]
// console.log("y_tensor shape:", y_tensor.shape); // [50000,1]

// ----------------------------
// 5. Model Architecture
// ----------------------------

const model = tf.sequential()

// A. Embedding layer
model.add(tf.layers.embedding({
    inputDim: index,
    outputDim: 16,
    inputLength: maxLen
}))

// B. Flatten → 1D array for dense layers
model.add(tf.layers.flatten());

// C. Hidden Dense layer
model.add(tf.layers.dense({units:16, activation:"relu"}))

// D. Output  layer
model.add(tf.layers.dense({ units:1, activation:"sigmoid" }));

// Compile
model.compile({
  optimizer:"adam",
  loss:"binaryCrossentropy",
  metrics:["accuracy"]
});

model.summary();

// ----------------------------
// 6. Train Model
// ----------------------------

(async()=>{
  const history = await model.fit(X_tensor, y_tensor, {
    epochs:5,
    batchSize:64,
    validationSplit:0.2,
    verbose:1,
    callbacks:{
        onEpochEnd: (epochs, logs)=>{
            console.log(`Epoch ${epochs + 1}: loss=${logs.loss.toFixed(4)}, accuracy:${logs.acc.toFixed(4)}`)
        }
    }
  });

  console.log("Training done!");
})();

// ----------------------------
// 7: Test Prediction
// ----------------------------

const testReview = "The movie was amazing and fantastic";
const testWords = testReview.toLowerCase().replace(/[^a-zA-Z ]/g,'').split(" ");
const testSeq = padSeq(testWords.map(w=>vocab[w]||1));
const testTensor = tf.tensor2d([testSeq],[1,maxLen]);

console.log("Final prediction")
const pred = model.predict(testTensor);
pred.print(); // probability 0-1
