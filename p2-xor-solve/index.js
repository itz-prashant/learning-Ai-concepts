function sigmoid(x){
    return 1 / (1 + Math.exp(-x))
}

function sigmoidDerivative(x){
    return 1 * (1 - x)
}

// Training data for XOR
let X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
]

let Y = [0, 1, 1 , 0] // expected output

// Initialize weihght and bias randomly

let w1 = Math.random();
let w2 = Math.random();
let w3 = Math.random();
let w4 = Math.random();
let b1 = Math.random();
let b2 = Math.random();
let w5 = Math.random();
let w6 = Math.random();
let b3 = Math.random();

let lr = 0.1 // learning rate

// Traning
for (let epoch = 1; epoch<= 25000; epoch++){
    let totalLoss = 0;

    for(let i = 0; i < X.length; i++){
        let [x1, x2] = X[i]
        let y = Y[i]

        // Forward Pass 
        // Hidden layer
        let h1_in = w1 * x1 + w2 * x2 + b1;
        let h1_out = sigmoid(h1_in)

        let h2_in = w3 * x1 + w4 * x2 + b2;
        let h2_out = sigmoid(h2_in)

        // output layer
        let o_in = w5 * h1_out + w6 * h2_out + b3

        let yPred = sigmoid(o_in);

        // Loss MSE
        let loss = 0.5 * Math.pow(y - yPred, 2);

        totalLoss += loss;

        // Backpropogation
        // output layer

        let dLoss_dyPred = yPred - y;
        let dyPred_do = sigmoidDerivative(yPred);
        let dLoss_do = dLoss_dyPred * dyPred_do;

        let dLoss_dw5 = dLoss_do * h1_out;
        let dLoss_dw6 = dLoss_do * h2_out
        let dLoss_db3 = dLoss_do 

        // hidden layer
        let dLoss_dh1 = dLoss_do * w5 * sigmoidDerivative(h1_out)
        let dLoss_dh2 = dLoss_do * w6 * sigmoidDerivative(h2_out)

        let dLoss_w1 = dLoss_dh1 * x1;
        let dLoss_w2 = dLoss_dh1 * x2;
        let dLoss_db1 = dLoss_dh1 ;

        let dLoss_w3 = dLoss_dh2 * x1;
        let dLoss_w4 = dLoss_dh2 * x2;
        let dLoss_db2 = dLoss_dh2 ;

        // update weights
        w1 -= lr * dLoss_w1;
        w2 -= lr * dLoss_w2;
        b1 -= lr * dLoss_db1;

        w3 -= lr * dLoss_w3;
        w4 -= lr * dLoss_w4;
        b2 -= lr * dLoss_db2;
    
        w5 -= lr * dLoss_dw5;
        w6 -= lr * dLoss_dw6;
        b3 -= lr * dLoss_db3;
    }
    if(epoch % 500 === 0){
        console.log(`Epoch ${epoch}, Loss=${totalLoss.toFixed(4)}`)
    }
}

// testing 

console.log("\nTesting XOR:");
for(let i = 0; i< X.length; i++){
    let [x1, x2] = X[i];

    let h1_out = sigmoid(w1 * x1 + w2 * x2 + b1);
    let h2_out = sigmoid(w3 * x1 + w4 * x2 + b2);
    let yPred = sigmoid(w5 * h1_out + w6 * h2_out + b3);

    console.log(`${x1} XOR ${x2}= ${yPred.toFixed(3)}` )
}