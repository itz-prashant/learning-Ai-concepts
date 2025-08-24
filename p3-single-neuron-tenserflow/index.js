import * as tf from "@tensorflow/tfjs-node"

async function main() {
    // data
    const xs = tf.tensor1d([1,2,3,4,5])
    const ys = tf.tensor1d([2,3,6,8,10])

    // model
    const model = tf.sequential();
    model.add(tf.layers.dense({
        units: 1,
        inputShape: [1]
    }))

    // compile
    model.compile({
        optimizer: tf.train.sgd(0.01),
        loss: 'meanSquaredError'
    });

    // Train
    await model.fit(xs, ys, {
        epochs: 200,
        verbose: 0,
        callbacks:{
            onBatchEnd: async (epoch, logs)=>{
                if((epoch + 1) % 50 === 0){
                    console.log(`Epoch ${epoch + 1}: loss=${logs.loss.toFixed(6)}`)
                }
            }
        }
    })

    // Predict
    const pred = model.predict(tf.tensor2d([7], [1,1]));
    const val = (await pred.data())[0]
    console.log('Prediction for x=7 = ', val.toFixed(4))

    // save model
    await model.save('file://./linear-model')
    console.log('model saved to ./linear-model/')
}

main().catch(console.error)