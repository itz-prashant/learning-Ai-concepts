import tf from "@tensorflow/tfjs-node"

async function main() {
    // data
    const xs = tf.tensor1d([...Array(20).keys()].map(i => i+1))
    const ys = tf.tensor1d(xs.arraySync().map(x=> x*x))

    // model
    const model = tf.sequential()

    // hidden layer (non-linear transformayion)
    model.add(tf.layers.dense({
        units: 62,
        inputShape: [1],
        activation: "tanh"
    }))

    // hidden layer 2
    model.add(tf.layers.dense({
        units: 32,
        activation: "tanh"
    }))

    // hidden layer 3
    // model.add(tf.layers.dense({
    //     units: 16,
    //     activation: "tanh"
    // }))

    // output layer
    model.add(tf.layers.dense({
        units: 1,
    }))

    // compile
    model.compile({
        optimizer: tf.train.adam(0.005),
        loss: "meanSquaredError"
    })

    // train
    await model.fit(xs, ys, {
        epochs: 600,
        callbacks:{
            onEpochEnd: async (epoch, logs)=>{
                if((epoch + 1) % 50 == 0){
                    console.log(`Epoch ${epoch + 1}: loss=${logs.loss.toFixed(4)}`)
                }
            }
        }
    })

    //print
    const textX = tf.tensor2d([6,7,8], [3,1]);
    const preds = model.predict(textX)
    preds.print()
}

main()