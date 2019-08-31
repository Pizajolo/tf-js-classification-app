const classifier = knnClassifier.create();
const webcamElement = document.getElementById('webcam');
let net;
let count_a = 0;
let count_b = 0;
let count_c = 0;
let classes = [];
let context_a = document.getElementById('canvas-a').getContext('2d');
let context_b = document.getElementById('canvas-b').getContext('2d');
let context_c = document.getElementById('canvas-c').getContext('2d');
let checked = false;

// Setting up the webcam.
async function setupWebcam() {
  return new Promise((resolve, reject) => {
    const navigatorAny = navigator;
    navigator.getUserMedia = navigator.getUserMedia ||
        navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia ||
        navigatorAny.msGetUserMedia;
    if (navigator.getUserMedia) {
      navigator.getUserMedia({video: true},
        stream => {
          webcamElement.srcObject = stream;
          webcamElement.addEventListener('loadeddata',  () => resolve(), false);
        },
        error => reject());
    } else {
      reject();
    }
  });
}

// Create an infinite loop which makes predictions through the webcam element.
async function app() {

  console.log('Loading mobilenet..');

  // Load the model.
  net = await mobilenet.load();
  console.log('Sucessfully loaded model');

  await setupWebcam();

  // Reads an image from the webcam and associates it with a specific class
  // index.
  const addExample = classId => {
    // Get the intermediate activation of MobileNet 'conv_preds' and pass that
    // to the KNN classifier.
    const activation = net.infer(webcamElement, 'conv_preds');

    // Count images fed in and if first image fed in ad class to classes
    if(classId === 0){
      if(count_a === 0){
        classes.push('A')
      }
      count_a++;
      document.getElementById('num-a').innerText = `${count_a}`;
      context_a.drawImage(document.getElementById('webcam'), 0, 0, 50, 32);
    }
    if(classId === 1){
      if(count_b === 0){
        classes.push('B')
      }
      count_b++;
      document.getElementById('num-b').innerText = `${count_b}`;
      context_b.drawImage(document.getElementById('webcam'), 0, 0, 50, 32);
    }
    if(classId === 2){
      if(count_c === 0){
        classes.push('C')
      }
      count_c++;
      document.getElementById('num-c').innerText = `${count_c}`;
      context_c.drawImage(document.getElementById('webcam'), 0, 0, 50, 32);
    }

    // Pass the intermediate activation to the classifier.
    classifier.addExample(activation, classId);
  };


  // When clicking a button, add an example for that class.
  document.getElementById('class-a').addEventListener('click', () => addExample(0));
  document.getElementById('class-b').addEventListener('click', () => addExample(1));
  document.getElementById('class-c').addEventListener('click', () => addExample(2));

  document.getElementById('myCheck').addEventListener('click', () => checked = document.getElementById('myCheck').checked);

  while (true) {
    if (classifier.getNumClasses() > 0) {
      // Get the activation from mobilenet from the webcam.
      const activation = net.infer(webcamElement, 'conv_preds');
      // Get the most likely class and confidences from the classifier module.
      if (checked) {

        document.getElementById("custom").style.visibility = "visible";
        document.getElementById("prediction").style.display = "none";
        const result = await classifier.predictClass(activation);
        const length = 4;
        // Print the prediction values to the table
        document.getElementById('out-A').innerText = `${result.confidences[0]}`.substring(0, length);
        document.getElementById('out-B').innerText = `${result.confidences[1]}`.substring(0, length);
        document.getElementById('out-C').innerText = `${result.confidences[2]}`.substring(0, length);

        // Color the background of highest prediction green
        if (classes[result.classIndex] === 'A') {
          document.getElementById("tab-A").style.backgroundColor = "green";
          document.getElementById("tab-B").style.backgroundColor = "transparent";
          document.getElementById("tab-C").style.backgroundColor = "transparent";
        }

        if (classes[result.classIndex] === 'B') {
          document.getElementById("tab-A").style.backgroundColor = "transparent";
          document.getElementById("tab-B").style.backgroundColor = "green";
          document.getElementById("tab-C").style.backgroundColor = "transparent";
        }

        if (classes[result.classIndex] === 'C') {
          document.getElementById("tab-A").style.backgroundColor = "transparent";
          document.getElementById("tab-B").style.backgroundColor = "transparent";
          document.getElementById("tab-C").style.backgroundColor = "green";
        }
      }
      if (checked === false){
        document.getElementById("custom").style.visibility = "hidden";
        document.getElementById("prediction").style.display = "block";
        const result_b = await net.classify(webcamElement);
        document.getElementById('prediction').innerText = `
          prediction: ${result_b[0].className}\n
          probability: ${result_b[0].probability}
        `;
      }

    }

    if (checked === false){
      document.getElementById("custom").style.visibility = "hidden";
      document.getElementById("prediction").style.display = "block";
      const result_b = await net.classify(webcamElement);
      document.getElementById('prediction').innerText = `
          prediction: ${result_b[0].className}\n
          probability: ${result_b[0].probability}
        `;
      }
    if (checked) {
      document.getElementById("custom").style.visibility = "visible";
      document.getElementById("prediction").style.display = "none";
    }

    await tf.nextFrame();
  }
}

app();