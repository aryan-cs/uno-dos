// import '../node_modules/bootstrap/dist/css/bootstrap.css';
// import * as tf from '../node_modules/@tensorflow/tfjs';
import { MnistData } from './data.js';


console.log(tf.sequential())

console.log("main script loaded...");

var model, runs = 0, succ = 0;

function logMessage (message) {

    console.log(message);
    var mess = document.createElement("p");
    mess.innerHTML = " > " + message;
    if (message.includes("!")) { mess.className = "success-log"; }
    else if (message.includes("...")) { mess.className = "loading-log"; }
    document.getElementById("log").appendChild(mess);

}

function create () {

    logMessage("creating model...");
    model = tf.sequential(); // model w layers, output of one layer is the input of the next
    logMessage("model created!");

    logMessage("adding layers...");

    model.add(tf.layers.conv2d({ // first layer

        inputShape: [28, 28, 1], // 28 x 28 pixels, 1 color
        kernelSize: 5, // 5 x 5 kernel
        filters: 8, // 8 filters of size 5
        strides: 1, // 1 pixel stride
        activation: 'relu', // rectified linear unit
        kernelInitializer: 'VarianceScaling', // initializes the kernel weights

    }));

    model.add(tf.layers.maxPooling2d({ // second layer

        poolSize: [2, 2], // 2 x 2 pooling
        strides: [2, 2], // 2 pixel stride

    }));

    model.add(tf.layers.conv2d({ // third layer (same as first layer)

        kernelSize: 5, // 5 x 5 kernel
        filters: 16, // 16 filters of size 5
        strides: 1, // 1 pixel stride
        activation: 'relu', // rectified linear unit
        kernelInitializer: 'VarianceScaling', // initializes the kernel weights

    }));

    model.add(tf.layers.maxPooling2d({ // fourth layer (same as second layer)

        poolSize: [2, 2], // 2 x 2 pooling
        strides: [2, 2], // 2 pixel stride

    }));

    model.add(tf.layers.flatten()); // flatten the output of the previous layer

    model.add(tf.layers.dense({ // fourth layer

        units: 10, // 10 digits, numbers 0-9
        kernelInitializer: 'VarianceScaling', // initializes the kernel weights
        activation: 'softmax', // softmax activation function - creates a probability distribution for output

    }));

    logMessage("layers created!");

    logMessage("compiling model...");
    model.compile({

        optimizer: tf.train.sgd(.15), // stochastic gradient descent - estimate to find best results
        loss: 'categoricalCrossentropy', // categorical crossentropy - quantifies the difference between the predicted and actual output

    });
    logMessage("model compiled!");

    // logMessage("model summary: " + model.summary());

}

let data;

async function load () {

    logMessage("loading mnist data...");
    data = new MnistData();
    await data.load(); // since using await, this function must be asynchronous
    logMessage("mnist data loaded!");

}

 // 150 batches of 64 samples each
const BATCH_SIZE = 64;
const TRAIN_BATCHES = 150;

async function train () {

    logMessage("started training model...");

    for (let b = 0; b < TRAIN_BATCHES; b++) {

        console.log("batch " + b);

        const batch = tf.tidy(() => {

            const batch = data.nextTrainBatch(BATCH_SIZE);
            batch.xs = batch.xs.reshape([BATCH_SIZE, 28, 28, 1]);
            return batch;

        });

        await model.fit( batch.xs, batch.labels, { batchSize: BATCH_SIZE, epochs: 1 } );

        tf.dispose(batch);

        await tf.nextFrame();

    }

    logMessage("model trained!");
    logMessage("press button to generate simulation...");

}

async function main () {

    create();
    await load();
    await train();
    document.getElementById("selectTestDataButton").disabled = false;
    document.getElementById("selectTestDataButton").innerText = "🧠";

}

async function predict (batch) {

    tf.tidy (() => {

        const input_value = Array.from(batch.labels.argMax(1).dataSync()); // actual
        const div = document.createElement("div");
        div.className = "prediction-div";

        const prediction = model.predict(batch.xs.reshape([-1, 28, 28, 1])); // get prob dist for each digit
        const prediction_value = Array.from(prediction.argMax(1).dataSync()); // extract value of highest probability, prediction
        const image = batch.xs.slice([0, 0,], [1, batch.xs.shape[1]]); // extract image

        const canv = document.createElement("canvas");
        canv.className = "prediction-canvas";
        draw(image.flatten(), canv)

        const label = document.createElement("div");
        label.innerHTML = "original value: " + input_value + "<br>predicted value: " + prediction_value;

        console.log("prediction: " + prediction_value | + " actual: " + input_value);

        if (prediction_value - input_value == 0) { label.innerHTML += "<br>correct!"; canv.className += " success"; succ++; }

        else { label.innerHTML += "<br>incorrect..."; canv.className += " fail"; }

        div.appendChild(canv);
        div.appendChild(label);
        document.getElementById("predictionResult").appendChild(div);

    });

}

function draw (image, canvas) {

    const [width, height] = [28, 28];
    canvas.width = width; canvas.height = height;
    const ctx = canvas.getContext('2d');
    const imageData = new ImageData(width, height);
    const data = image.dataSync();

    for (let i = 0; i < height * width; ++i) {

      const j = i * 4;
      imageData.data[j + 0] = data[i] * 255;
      imageData.data[j + 1] = data[i] * 255;
      imageData.data[j + 2] = data[i] * 255;
      imageData.data[j + 3] = 255;

    }

    ctx.putImageData(imageData, 0, 0);

}

document.getElementById("selectTestDataButton").addEventListener('click', async (el, ev) => {

    runs++;

    const batch = data.nextTestBatch(1);
    await predict(batch);
    document.getElementById("predictionResults").innerHTML = "results [" + runs + " samples, " + ((succ / runs) * 100).toFixed(2) + "% success rate]";

});

main();