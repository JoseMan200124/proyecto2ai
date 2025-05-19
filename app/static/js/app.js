// static/js/app.js

// Punto de entrada de tu API FastAPI
const API_URL = "http://127.0.0.1:8000";

const video  = document.getElementById("video");
const canvas = document.getElementById("canvas");
const result = document.getElementById("result");
const ctx    = canvas.getContext("2d");

async function setupCamera() {
  // Pide permiso y enlaza la cámara al <video>
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
  return new Promise(resolve => video.onloadedmetadata = resolve);
}

async function sendFrame() {
  // Redimensiona el canvas a la resolución del video
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  // Copia la imagen del video al canvas
  ctx.drawImage(video, 0, 0);

  // Extrae un blob JPEG del canvas
  canvas.toBlob(async blob => {
    const form = new FormData();
    form.append("file", blob, "frame.jpg");

    try {
      // POST a http://127.0.0.1:8000/predict
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: form
      });

      // Si el servidor devuelve JSON...
      const data = await res.json();

      if (res.ok && data.success) {
        result.textContent = "Gesto: " + data.prediction;
      } else {
        // Puede venir { success: false, message: "..." }
        result.textContent = data.message || "Error en la predicción";
      }

    } catch (e) {
      console.error("Error de conexión:", e);
      result.textContent = "No se pudo conectar al servidor";
    }
  }, "image/jpeg", 0.8);
}

(async () => {
  await setupCamera();
  // Lanza una predicción cada segundo
  setInterval(sendFrame, 1000);
})();
