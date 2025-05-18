const video  = document.getElementById("video");
const canvas = document.getElementById("canvas");
const result = document.getElementById("result");
const ctx    = canvas.getContext("2d");

async function setupCamera(){
  const stream = await navigator.mediaDevices.getUserMedia({video:true});
  video.srcObject = stream;
  return new Promise(resolve => video.onloadedmetadata = resolve);
}

async function sendFrame(){
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video,0,0);
  canvas.toBlob(async blob=>{
    const form = new FormData();
    form.append("file", blob, "frame.jpg");
    try{
      const res = await fetch("/predict",{method:"POST", body:form});
      const data= await res.json();
      if(data.success){
        result.textContent = "Gesto: "+data.prediction;
      }else{
        result.textContent = data.message;
      }
    }catch(e){
      result.textContent = "Error de conexión";
    }
  },"image/jpeg",0.8);
}

(async()=>{
  await setupCamera();
  setInterval(sendFrame, 1000); // 1 FPS para no saturar
})();
