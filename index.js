const classifier = knnClassifier.create();
const webcamElement = document.getElementById('webcam');
let net;

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

    // Pass the intermediate activation to the classifier.
    classifier.addExample(activation, classId);
  };

  // When clicking a button, add an example for that class.
  document.getElementById('class-a').addEventListener('click', () => addExample(0));
  document.getElementById('class-b').addEventListener('click', () => addExample(1));
  document.getElementById('class-c').addEventListener('click', () => addExample(2));

  while (true) {
    if (classifier.getNumClasses() > 0) {
      // Get the activation from mobilenet from the webcam.
      const activation = net.infer(webcamElement, 'conv_preds');
      // Get the most likely class and confidences from the classifier module.
      const result = await classifier.predictClass(activation);

      const classes = ['A', 'B', 'C'];
      const length = 4;

      document.getElementById('out-A').innerText = `${result.confidences[0]}`.substring(0, length);
      document.getElementById('out-B').innerText = `${result.confidences[1]}`.substring(0, length);
      document.getElementById('out-C').innerText = `${result.confidences[2]}`.substring(0, length);

      if(classes[result.classIndex] === 'A') {
        document.getElementById("tab-A").style.backgroundColor = "green";
        document.getElementById("tab-B").style.backgroundColor = "transparent";
        document.getElementById("tab-C").style.backgroundColor = "transparent";
      }

      if(classes[result.classIndex] === 'B') {
        document.getElementById("tab-A").style.backgroundColor = "transparent";
        document.getElementById("tab-B").style.backgroundColor = "green";
        document.getElementById("tab-C").style.backgroundColor = "transparent";
      }

      if(classes[result.classIndex] === 'C') {
        document.getElementById("tab-A").style.backgroundColor = "transparent";
        document.getElementById("tab-B").style.backgroundColor = "transparent";
        document.getElementById("tab-C").style.backgroundColor = "green";
      }
    }

    await tf.nextFrame();
  }
}

app();