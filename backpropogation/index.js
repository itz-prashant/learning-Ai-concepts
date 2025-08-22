function sigmoid(x){
    return 1 / (1 + Math.exp(-x))
}

function sigmoidDerivative(x){
    return x * (1 - x)
}

let x1 = 1;
let x2 = 0;

let w1 = 0;
let w2 = 0;
let b = 0;
let y = 1
let lr = 1.0

for(let epoch = 1; epoch<=10; epoch++){
    // forward pass
    let z = (w1 * x1) + (w2 * x2) + b
    let yPred = sigmoid(z)

    // Loss (MSE)
    let loss = 0.5 * Math.pow((yPred - y), 2)

    // Backpropogation
    let dLoss_dyPred = yPred - y
    let dyPred_dz = sigmoidDerivative(yPred)
    let dLoss_dz = dLoss_dyPred * dyPred_dz;

    // Gradients
    let dLoss_dW1 = dLoss_dz * x1;
    let dLoss_dW2 = dLoss_dz * x2;
    let dLoss_db = dLoss_dz * 1;

    // update
    w1 -= lr * dLoss_dW1
    w2 -= lr * dLoss_dW2
    b -= lr * dLoss_db

    console.log(`Epoch ${epoch}: yPred=${yPred.toFixed(3)}, Loss=${loss.toFixed(4)}, w1=${w1.toFixed(3)}, w2=${w2.toFixed(3)}, b=${b.toFixed(3)}`)
}

