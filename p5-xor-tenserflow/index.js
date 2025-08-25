import tf from "@tensorflow/tfjs-node"

async function runXOR() {
    const xs = tf.tensor2d([
        [0,0],
        [0,1],
        [1,0],
        [1,1],
    ])

    const ys = tf.tensor2d([
        [0],
        [1],
        [1],
        [0],
    ])

    const model = tf.sequential()
    model.add(tf.layers.dense({
        units: 8,
        inputShape: [2],
        activation: "tanh",
        kernelInitializer: "glorotUniform"
    }))

    model.add(tf.layers.dense({
        units: 4,
        activation: "tanh",
        kernelInitializer: "glorotUniform"
    }))

    model.add(tf.layers.dense({
        units:1,
        activation: "sigmoid"
    }))

    model.compile({
        optimizer: tf.train.adam(0.03),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    })

    await model.fit(xs, ys, {
        epochs: 500,
        batchSize: 4,
        callbacks:{
            onEpochEnd: async (epochs, logs)=>{
                if((epochs + 1) % 100 === 0){
                    const acc = (logs.acc ?? logs.accuracy) ?? 0;
                    console.log(`Epoch ${epochs + 1}: loss=${logs.loss.toFixed(5)} acc=${acc.toFixed(4)}`)
                }
            }
        }
    });

    const pred = model.predict(xs);
    const probs = Array.from(await pred.data()) // flat array of probability
    const label = probs.map(p=> p > 0.5 ? 1 : 0)

    console.log(`\nPredected probability`, probs.map(p=> p.toFixed(4)))
    console.log(`Predected label`, label)
    
    // const correct = label.reduce((s,1,i) => s + (1 === [0,1,1,0][i] ? 1 : 0), 0)
    const correct = label.reduce((sum, val, i) => sum + (val === [0,1,1,0][i] ? 1 : 0), 0);

    console.log(`Accuracy mannually ${(correct / label.length * 100).toFixed(2)}%`)
}

runXOR().catch(console.error)