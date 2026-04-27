import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

let latencies = [];

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;

  return new Promise(resolve => {
    video.onloadedmetadata = () => resolve(video);
  });
}

async function run() {
  await setupCamera();

  await tf.setBackend('webgl');
  await tf.ready();

  console.log('Backend:', tf.getBackend());

  const model = await cocoSsd.load();

  setInterval(async () => {
    const start = performance.now();

    const predictions = await model.detect(video);

    const end = performance.now();
    const latency = end - start;
    latencies.push(latency);

    const avgLatency =
      latencies.reduce((a, b) => a + b, 0) / latencies.length;

    const fps = 1000 / avgLatency;

    console.log(
      `Latency: ${latency.toFixed(2)} ms | Avg: ${avgLatency.toFixed(2)} ms | FPS: ${fps.toFixed(2)}`
    );

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    predictions.forEach(p => {
      ctx.strokeStyle = 'red';
      ctx.lineWidth = 2;
      ctx.strokeRect(...p.bbox);
      ctx.fillStyle = 'red';
      ctx.fillText(`${p.class} ${p.score.toFixed(2)}`, p.bbox[0], p.bbox[1] - 5);
    });
  }, 300);
}

document.getElementById('start').onclick = run;