let model;

async function main() {
    model = await tf.loadLayersModel('training/tf_js/decoder/model.json');
    console.log("Model loaded");
    console.log("Model summary:");
    model.summary();
}
let x = 0;
let y = 0;
const colors = [
    [255, 0, 0],    // 'red'
    [0, 0, 255],    // 'blue'
    [0, 128, 0],    // 'green'
    [128, 0, 128],  // 'purple'
    [255, 165, 0],  // 'orange'
    [0, 255, 255],  // 'cyan'
    [255, 0, 255],  // 'magenta'
    [255, 255, 0],  // 'yellow'
    [0, 255, 0],    // 'lime'
    [165, 42, 42],  // 'brown'
];

const labelNames = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
];


const legend = document.createElement("div");
legend.innerHTML = "<h3>Legend:</h3>"
colors.forEach((color, i) => {
    legend.innerHTML += `<span style="width:30px;height:30px;background-color:rgb(${color.join(", ")})">&nbsp&nbsp&nbsp&nbsp&nbsp</span> - ${labelNames[i]} ${i % 2 === 0 ? "&nbsp&nbsp&nbsp&nbsp&nbsp": "<br><br>"}`;
})

let classMapImg;

function setup() {
    createCanvas(28 * 20, 28 * 10);
    classMapImg = loadImage("latentMap.png");
    document.body.appendChild(legend);
}

function draw() {
    if (mouseIsPressed) {
        // Calculate the normalized x-coordinate based on mouse position
        x = constrain(map(mouseX - 28 * 10, 0, 28 * 10, 0, 1), 0, 1);
        
        // Calculate the normalized y-coordinate based on mouse position
        y = constrain(map(mouseY, 0, 28 * 10, 0, 1), 0, 1);
    }
    if (model) {
        noStroke();
        const img = numToImg([x, y]);
        for (let y = 0; y < 28; y++) {
            for (let x = 0; x < 28; x++) {
                fill(img[y * 28 + x]);
                rect(x * 10, y * 10, 10, 10);
            }
        }

    }
    fill(200);
    rect(28 * 10, 0, 28 * 10, 28 * 10);
    image(classMapImg, 280, 0, 280, 280);
    fill(mouseIsPressed ? 100 : 150);
    strokeWeight(5);
    stroke(50);
    circle(map(x, 0, 1, 28 * 10, 28 * 20), map(y, 0, 1, 0, 28 * 10), 25);
}
main();

function numToImg(arr) {
    return model.predict(tf.tensor([arr])).dataSync().map(x => x * 255);
}
