// Load the TFLite model
async function loadModel() {
  const model = await tf.loadGraphModel("./web_model/model.json");
  return model;
}

// Image preprocessing function
function preprocessImage(image) {
  // Resize the image to 224x224 and normalize it
  const tensor = tf.browser
    .fromPixels(image)
    .resizeNearestNeighbor([224, 224]) // Resize
    .toFloat()
    .div(255.0) // Normalize
    .expandDims(); // Add batch dimension [1, 224, 224, 3]
  return tensor;
}

// Event listener for file input
document
  .getElementById("fileInput")
  .addEventListener("change", async (event) => {
    const file = event.target.files[0];
    const img = new Image();
    img.src = URL.createObjectURL(file);

    img.onload = async () => {
      // Draw image on canvas
      const canvas = document.getElementById("canvas");
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0, 224, 224);

      const model = await loadModel();
      const tensor = preprocessImage(canvas);

      // Run prediction
      const prediction = model.predict(tensor);
      prediction.print(); // Log the output to the console

      // Interpret the prediction results
      const result = prediction.argMax(-1).dataSync()[0];
      console.log(`Predicted personality ID: ${result}`);
    };
  });
