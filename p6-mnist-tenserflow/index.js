import * as tf from "@tensorflow/tfjs-node";
import mnist from "mnist";

// ===============================
// 1. DATA LOAD & PREPROCESSING
// ===============================
const { training, test } = mnist.set(8000, 2000);

const trainXs = tf.tensor2d(training.map(item => item.input));
const trainYs = tf.tensor2d(training.map(item => item.output));

const testXs = tf.tensor2d(test.map(item => item.input));
const testYs = tf.tensor2d(test.map(item => item.output));

// ===============================
// 2. MODEL BANANA
// ===============================
const model = tf.sequential();

model.add(tf.layers.dense({
  units: 128,
  inputShape: [784],
  activation: "relu"
}));

model.add(tf.layers.dense({
  units: 64,
  activation: "relu"
}));

model.add(tf.layers.dense({
  units: 10,
  activation: "softmax"
}));

// ===============================
// 3. COMPILE MODEL
// ===============================
model.compile({
  optimizer: "adam",
  loss: "categoricalCrossentropy",
  metrics: ['accuracy']
});

// ===============================
// 4. TRAINING
// ===============================
(async () => {
  console.log("Training start");

  await model.fit(trainXs, trainYs, {
    epochs: 10,                 // ✅ correct key
    batchSize: 512,
    validationData: [testXs, testYs],
    shuffle: true,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(`Epoch ${epoch + 1}: loss=${logs.loss.toFixed(4)}, acc=${(logs.acc*100).toFixed(2)}%`);
      }
    }
  });

  console.log("✅ Training done");

  // ===============================
  // 5. TEST PREDICTION
  // ===============================
  const pred = model.predict(testXs.slice([0,0],[1,784]));
  pred.print();

  const predictedIndex = pred.argMax(-1).dataSync()[0];
  console.log("Predicted digit:", predictedIndex);
})();
