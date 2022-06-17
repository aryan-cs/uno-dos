import { MnistData } from './data.js';

console.log("main script loaded...");

var model, runs = 0, succ = 0, canvas, rawImage, ctx, data, trained = false;
var identifyButton, clearButton;
var pos = { x: 0, y: 0 };

let BATCH_SIZE = 64;
let TRAIN_BATCHES = 150;

let regexp = /android|iphone|kindle|ipad/i;

function logMessage (message) {

    console.log(message);
    var mess = document.createElement("p");
    mess.innerHTML = " > " + message;
    if (message.includes("!")) { mess.className = "success-log"; }
    else { mess.className = "normal-log"; }
    document.getElementById("log").appendChild(mess);

}

function runCommand (command) {

    if (command == ("help") || (command == ("?"))) {
        
        return "available commands: <br>&nbsp&nbsp[run &ltnumber of simulations&gt] - runs n simulations" +
                                    "<br>&nbsp&nbsp[recreate &ltbatch size&gt, &ltnumber of batches&gt] - recreates & retrains model with set parameters" +
                                    "<br>&nbsp&nbsp[refresh] - clears stats" + 
                                    "<br>&nbsp&nbsp[clear] - clears result cards" +
                                    "<br>&nbsp&nbsp[help] - help" +
                                    "<br> enter a command to continue...";
    
    }

    if (command.includes("run")) {

        const start = Date.now();
        for (var r = 0; r < command.substring(command.indexOf("run") + 4); r++) { simulate(); }
        const duration = Date.now() - start;

        return "ran " + parseInt(command.substring(command.indexOf("run") + 4)) + " simulations... [" + duration + " ms]";

    }

    if (command.includes("stats")) {

        return "runs: " + runs +
               "<br>&nbsp&nbspsuccesses: " + succ +
               "<br>&nbsp&nbspsuccess rate: " + successRate() + "%" +
               "<br>&nbsp&nbsprating: " + rating(successRate());

    }

    if (command.includes("recreate")) {

        BATCH_SIZE = parseInt(command.substring(command.indexOf("recreate") + 8, command.indexOf(",")));
        TRAIN_BATCHES = parseInt(command.substring(command.indexOf(",") + 2));
        main();
        return "recreating model with " + TRAIN_BATCHES + " batches of size " + BATCH_SIZE + "...";

    }

    if (command.includes("refresh")) {
        
        runs = 0; succ = 0;
        document.getElementById("predictionResults").innerHTML = "results";
        return "refreshed stats...";
    
    }

    if (command.includes("clear")) {
        
        document.getElementById("predictionResult").innerHTML = "";
        return "cleared result cards...";
    
    }

    return "unknown command: " + command + "<br> enter 'help' for lost of available commands";

}

function awaitCommand () {

    var symb = document.createElement("p");
    symb.className = "command-log";
    symb.innerHTML = ">";

    var inp = document.createElement("input");
    inp.className = "command-input";
    inp.addEventListener("keydown", function (e) {

        if (e.key === "Enter") {

            logMessage(runCommand(inp.value));
            inp.disabled = true;
            awaitCommand();

        }

    });
    symb.appendChild(inp);
    document.getElementById("log").appendChild(symb);
    inp.focus();

}

function successRate () {

    return !runs > 0 ? "--" : ((succ / runs) * 100).toFixed(2);

}

function rating (rate) {

    switch (true) {
        
        case (rate < 50): return "bad";
        case (rate < 60): return "poor";
        case (rate < 70): return "useless";
        case (rate < 80): return "average";
        case (rate < 90): return "good";
        case (rate < 99): return "great";
        case (rate == 100): return "perfect!";
        default: return "--";

    }

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

async function load () {

    logMessage("loading mnist data...");
    data = new MnistData();
    await data.load(); // since using await, this function must be asynchronous
    logMessage("mnist data loaded!");

}

async function train () {

    logMessage("started training model... [" + TRAIN_BATCHES + " batches of " + BATCH_SIZE + " samples each]");

    const start = Date.now();

    for (let b = 0; b < TRAIN_BATCHES; b++) {

        logMessage("training batch " + (b + 1) + " of " + TRAIN_BATCHES + "...");

        const batch = tf.tidy(() => {

            const batch = data.nextTrainBatch(BATCH_SIZE);
            batch.xs = batch.xs.reshape([BATCH_SIZE, 28, 28, 1]);
            return batch;

        });

        await model.fit( batch.xs, batch.labels, { batchSize: BATCH_SIZE, epochs: 1 } );

        tf.dispose(batch);

        await tf.nextFrame();

    }

    const duration = Date.now() - start;

    logMessage("model trained! [" + duration + " ms]"); trained = true;
    logMessage("press button or run command to generate simulations...");

    awaitCommand();

}

async function main () {

    if (!regexp.test(navigator.userAgent)) {

    init();
    create();
    await load();
    await train();
    document.getElementById("selectTestDataButton").disabled = false;
    clearButton.disabled = false;
    identifyButton.disabled = false;

    }

    else {
        
        logMessage("please use a desktop browser, mobile devices cannopt handle the processes required for this program...");
        logMessage("thank you for your understanding, good bye!");
    
    }

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

async function simulate () {

    runs++;

    const batch = data.nextTestBatch(1);
    await predict(batch);
    document.getElementById("predictionResults").innerHTML = "results [" + runs + " samples, " + successRate() + "% success rate]";


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

document.getElementById("selectTestDataButton").addEventListener('click', async (el, ev) => { simulate(); });

function setPosition (e){

	pos.x = e.clientX - canvas.getBoundingClientRect().left;
	pos.y = e.clientY - canvas.getBoundingClientRect().top;

}
    
function write (e) {

	if (e.buttons != 1) return;

    if (trained) {

        ctx.beginPath();
        ctx.lineWidth = 20;
        ctx.lineCap = "round";
        ctx.strokeStyle = "white";
        ctx.moveTo(pos.x, pos.y);
        setPosition(e);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
        
        rawImage.src = canvas.toDataURL("image/png");

    }

}

function erase() { ctx.fillStyle = "black"; ctx.fillRect(0, 0, 280, 280); }
    
function save() {

    console.log("saving...");

    // get raw image, resize to 28x28, expand to 1D
    var prediction = model.predict(tf.image.resizeBilinear(tf.browser.fromPixels(rawImage, 1), [28,28]).expandDims(0));

    // get prediction value
    var pred = tf.argMax(prediction, 1).dataSync();

    const div = document.createElement("div");
    div.className = "prediction-div custom";

    const img = document.createElement("img");
    img.className = "prediction-canvas";
    img.src = rawImage.src;

    const label = document.createElement("div");
    label.innerHTML = "predicted value: " + pred;
    label.innerHTML += "<br>custom drawing";

    div.appendChild(img);
    div.appendChild(label);
    document.getElementById("predictionResult").appendChild(div);
    
}

function init() {

	canvas = document.getElementById('canvas');
	rawImage = document.getElementById('canvasimg');

	ctx = canvas.getContext("2d");
	ctx.fillStyle = "black";
	ctx.fillRect(0, 0, 280, 280);
	canvas.addEventListener("mousemove", write);
	canvas.addEventListener("mousedown", setPosition);
	canvas.addEventListener("mouseenter", setPosition);

	identifyButton = document.getElementById("identifyButton");
	identifyButton.addEventListener("click", save);
	clearButton = document.getElementById("clearButton");
	clearButton.addEventListener("click", erase);

}

main();