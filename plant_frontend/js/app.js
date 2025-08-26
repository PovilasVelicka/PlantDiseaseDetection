/* ============================================
 * LT: Nustatyk savo FastAPI endpoint žemiau
 * ============================================ */
const API_URL = "http://localhost:8000/predict"; // pvz., https://plant-api.azurewebsites.net/predict

/* ==== DOM ==== */
const video   = document.getElementById('video');
const canvas  = document.getElementById('canvas');
const thumb   = document.getElementById('thumb');

const btnStartCam = document.getElementById('btnStartCam');
const btnShot     = document.getElementById('btnShot');
const btnClear    = document.getElementById('btnClear');
const btnSend     = document.getElementById('btnSend');
const fileInput   = document.getElementById('fileInput');

const loader   = document.getElementById('loader');
const statusEl = document.getElementById('status');
const rawEl    = document.getElementById('raw');
const humanEl  = document.getElementById('human');
const tableEl  = document.getElementById('table');
const tbodyEl  = document.getElementById('tbody');

let stream = null;
let haveImage = false;

/* ==== Utils ==== */
function statusMsg(msg, kind){
  statusEl.className = kind ? kind : '';
  statusEl.textContent = msg || '';
}

function escapeHtml(s){
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function pct(x){ return (x*100).toFixed(1) + '%'; }

function canvasToBlob(cnv, type='image/jpeg', quality=0.92){
  return new Promise(resolve => cnv.toBlob(resolve, type, quality));
}

function readImage(file){
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    const reader = new FileReader();
    reader.onload = () => { img.src = reader.result; };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

/* ============================================
 * Center-crop + resize to 256×256
 * ============================================ */
function drawImageCoverToCanvas(imgOrVideo){
  const ctx = canvas.getContext('2d');
  const target = 256;
  canvas.width = target; canvas.height = target;

  const w = imgOrVideo.videoWidth || imgOrVideo.naturalWidth || imgOrVideo.width;
  const h = imgOrVideo.videoHeight || imgOrVideo.naturalHeight || imgOrVideo.height;
  if(!w || !h){ ctx.clearRect(0,0,target,target); return; }

  const scale = Math.max(target / w, target / h);
  const sw = target / scale;
  const sh = target / scale;
  const sx = (w - sw) / 2;
  const sy = (h - sh) / 2;

  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.clearRect(0,0,target,target);
  ctx.drawImage(imgOrVideo, sx, sy, sw, sh, 0, 0, target, target);
}

function refreshThumb(){
  thumb.src = canvas.toDataURL('image/jpeg', 0.92);
  thumb.style.display = 'block';
  canvas.style.display = 'none';
  video.style.display = 'none';
}

/* ============================================
 * Camera — start/stop
 * ============================================ */
async function startCamera(){
  try{
    stopCamera();
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal:"environment" } }, audio:false
    });
    video.srcObject = stream;
    await video.play();
    video.style.display = 'block';
    btnShot.disabled = false;
    btnClear.disabled = false;
    btnSend.disabled = true;
    haveImage = false;
    thumb.style.display = 'none';
    canvas.style.display = 'none';
    statusMsg('Kamera įjungta', 'ok');
  }catch(err){
    statusMsg('Nepavyko gauti prieigos prie kameros: ' + err.message, 'err');
  }
}

function stopCamera(){
  if(stream){
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  video.srcObject = null;
  video.style.display = 'none';
}

/* ============================================
 * UI Handlers
 * ============================================ */
btnStartCam.addEventListener('click', startCamera);

btnShot.addEventListener('click', () => {
  if(!video.srcObject){ return; }
  drawImageCoverToCanvas(video);
  refreshThumb();
  haveImage = true;
  btnSend.disabled = false;
  statusMsg('Nuotrauka paruošta (256×256)', 'ok');
  stopCamera();
});

fileInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if(!file) return;
  try{
    const img = await readImage(file);
    drawImageCoverToCanvas(img);
    refreshThumb();
    haveImage = true;
    btnSend.disabled = false;
    btnClear.disabled = false;
    statusMsg('Failas įkeltas ir paruoštas (256×256)', 'ok');
  }catch(err){
    statusMsg('Klaida skaitant failą: ' + err.message, 'err');
  }
});

btnClear.addEventListener('click', () => {
  stopCamera();
  thumb.removeAttribute('src');
  thumb.style.display = 'none';
  canvas.getContext('2d').clearRect(0,0,canvas.width,canvas.height);
  haveImage = false;
  btnSend.disabled = true;
  statusMsg('Išvalyta', '');
});

/* ============================================
 * Send to API
 * ============================================ */
btnSend.addEventListener('click', async () => {
  if(!haveImage){ statusMsg('Nėra paveikslėlio', 'warn'); return; }
  try{
    loader.classList.add('on');
    btnSend.disabled = true;

    const blob = await canvasToBlob(canvas, 'image/jpeg', 0.92);
    const form = new FormData();
    form.append('file', blob, 'photo.jpg');

    const res = await fetch(API_URL, { method:'POST', body:form });
    const data = await res.json();

    if(!res.ok){
      throw new Error((data && data.detail) ? data.detail : ('HTTP ' + res.status));
    }

    renderResult(data);
    statusMsg('Atlikta', 'ok');
  }catch(err){
    statusMsg('API klaida: ' + err.message, 'err');
    rawEl.textContent = '';
    humanEl.innerHTML = '';
    tableEl.style.display = 'none';
  }finally{
    loader.classList.remove('on');
    btnSend.disabled = false;
  }
});

/* ============================================
 * Render API response
 * ============================================ */
function renderResult(data){

  humanEl.innerHTML = `
    <div class="row">
      <div class="badge">Rezultatas</div>
      <div><strong>${escapeHtml(data.disease_class)}</strong> · ${pct(data.confidence)}</div>
    </div>
  `;

  tbodyEl.innerHTML = '';
  if(Array.isArray(data.top_predicts) && data.top_predicts.length){
    data.top_predicts.forEach(item => {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${escapeHtml(item.disease_class)}</td><td>${pct(item.confidence)}</td>`;
      tbodyEl.appendChild(tr);
    });
    tableEl.style.display = 'table';
  } else {
    tableEl.style.display = 'none';
  }

  rawEl.textContent = JSON.stringify(data, null, 2);
}
