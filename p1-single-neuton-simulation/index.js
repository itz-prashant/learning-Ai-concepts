function sigmoid(x){
    return 1 / (1 + Math.exp(-x))
}

function neuron(x1, x2, w1, w2, b){
    let z = (x1 * w1) + (x2 * w2) + b;

    let output = sigmoid(z)
    return output
}

let x1 = 0.5;
let x2 = 1.0;

let w1 = 0.7;
let w2 = -1.2;
let b = 0.3;

let result = neuron(x1, x2, w1, w2, b) 

console.log(result)

let x3 = 8;
let x4 = 4;

let w3 = 0.6;
let w4 = 0.2;
let b2 = -3;

let result2 = neuron(x3, x4, w3, w4, b2) 

console.log(result2)
