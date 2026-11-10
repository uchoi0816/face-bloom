/**
 * Face Bloom — 螺旋五官 + 手掌控制
 * 转动手掌 → 整体旋转；手向前推 / 向后缩 → 放大 / 缩小
 */

const LAYERS = 24;
const LEAF = 5;
const ANIM_SPEED = 0.26; // 螺旋 / 噪声时间缩放（越小越慢）
const SPIRAL_PERIOD = 5200; // 原 millis/3500，越大转得越慢
const MIN_DRAW_SIDE = 16; // 跳过过小的碎块
const MIN_SPIRAL_SCALE = 0.9; // 螺旋不会缩到看不见
const FACE_HOLD_MS = 20000; // 脸短暂丢失仍沿用上一帧裁切
const BOX_STROKE = [120, 200, 230];
const PARTICLE_DIV = 1.2; // 越小方框越大
const SPIRAL_RADIUS = 0.72; // 相对屏幕短边的螺旋半径（原 0.5）
const CROP_PAD_MUL = 0.58; // 裁切更紧 → 五官在纹理里更大
const CROP_SQUARE_MUL = 0.9; // 采样方框相对特征区更小
const FEATURE_RES = 140;
const POOL_SLOTS = 24; // 滚动存更多裁切，螺旋更丰富

const FEATURE_SETS = [
  { name: "leftEye", idx: [33, 133, 160, 159, 158, 157, 173, 155, 154, 153, 145, 144, 163, 7], pad: 0.55 },
  { name: "rightEye", idx: [362, 263, 387, 386, 385, 384, 398, 382, 381, 380, 374, 373, 390, 249], pad: 0.55 },
  { name: "leftBrow", idx: [70, 63, 105, 66, 107, 55, 65, 52, 53, 46], pad: 0.5 },
  { name: "rightBrow", idx: [300, 293, 334, 296, 336, 285, 295, 282, 283, 276], pad: 0.5 },
  { name: "nose", idx: [1, 2, 98, 327, 168, 6, 197, 195, 5, 4, 19, 94, 141, 275], pad: 0.45 },
  { name: "noseTip", idx: [1, 4, 5, 19, 94, 2, 98, 327], pad: 0.7 },
  { name: "mouth", idx: [61, 291, 0, 17, 13, 14, 78, 308, 82, 312, 87, 317, 95, 324], pad: 0.48 },
  { name: "upperLip", idx: [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291], pad: 0.55 },
  { name: "lowerLip", idx: [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291], pad: 0.55 },
  { name: "leftCheek", idx: [50, 101, 118, 117, 116, 123, 147, 213, 192, 214, 212], pad: 0.4 },
  { name: "rightCheek", idx: [280, 330, 347, 346, 345, 352, 376, 433, 416, 434, 432], pad: 0.4 },
  { name: "chin", idx: [152, 175, 199, 200, 208, 171, 148, 176, 149, 150, 136], pad: 0.45 },
  { name: "forehead", idx: [10, 67, 69, 104, 108, 151, 337, 299, 333, 298, 9], pad: 0.5 },
];

let layers = LAYERS;
let leaf = LEAF;
let featureCanvases = []; // 实时槽
let featurePool = []; // 滚动历史，给螺旋用
let poolWrite = 0;
let featureReady = false;
let faceLandmarker = null;
let handLandmarker = null;
let videoEl = null;
let camReady = false;
let lastFaceDetect = 0;
let lastHandDetect = 0;
let useLive = true;
let refImg = null;
let refCrops = [];
let statusEl = null;
let camLabelEl = null;
let debugEl = null;
let videoDevices = [];
let currentCamIndex = 0;
let currentStream = null;
let hudHideTimer = null;
let currentCamLabel = "";
let faceSeen = false;
let lastFaceOkAt = 0;

// 手掌驱动（平滑）
let handRot = 0;
let handScale = 1;
let targetRot = 0;
let targetScale = 1;
let basePalmSize = null; // 首次检测到的掌面大小，作缩放基准
let handSeen = false;
let booted = false;
let lastFaceLm = null;
let lastFaceAt = 0;

function clamp(v, a, b) {
  return Math.max(a, Math.min(b, v));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

new window.p5((p) => {
  p.preload = () => {
    refImg = p.loadImage("assets/features-ref.png");
  };

  p.setup = () => {
    p.createCanvas(p.windowWidth, p.windowHeight);
    p.pixelDensity(Math.min(2, window.devicePixelRatio || 1));
    p.imageMode(p.CENTER);
    p.rectMode(p.CENTER);
    p.strokeWeight(1.5);
    p.background(242, 240, 236);

    statusEl = document.getElementById("hud");
    camLabelEl = document.getElementById("cam-label");
    debugEl = document.getElementById("debug");
    videoEl = document.getElementById("cam");

    seedFeatureCanvases(p);
    extractRefCrops(p);

    // 进页直接开摄像头，无需按钮
    boot(p, true);

    // 第一次点击/按键时再进全屏（浏览器手势限制）
    const onceFS = () => {
      enterFullscreen();
      window.removeEventListener("pointerdown", onceFS);
      window.removeEventListener("keydown", onceFS);
    };
    window.addEventListener("pointerdown", onceFS, { once: true });
    window.addEventListener("keydown", onceFS, { once: true });

    window.addEventListener("keydown", (e) => {
      if (e.key === "f" || e.key === "F") {
        e.preventDefault();
        toggleFullscreen();
      }
      if (e.key === "c" || e.key === "C") {
        e.preventDefault();
        cycleCamera();
      }
      if (e.key === "d" || e.key === "D") {
        e.preventDefault();
        document.body.classList.toggle("hide-debug");
      }
    });
  };

  p.windowResized = () => {
    p.resizeCanvas(p.windowWidth, p.windowHeight);
  };

  p.draw = () => {
    // 朦胧白拖影
    p.noStroke();
    p.fill(245, 243, 239, 28);
    p.rect(p.width / 2, p.height / 2, p.width, p.height);

    if (useLive && camReady) {
      if (faceLandmarker) detectAndCrop(p);
      if (handLandmarker) detectHand(p);
    }

    updateDebug();

    // 平滑跟随手掌
    handRot = lerp(handRot, targetRot, 0.18);
    handScale = lerp(handScale, targetScale, 0.16);
    const spiralScale = Math.max(handScale, MIN_SPIRAL_SCALE);

    const pool = featurePool.length ? featurePool : refCrops;
    if (!pool.length) return;

    const size_ = Math.min(p.width, p.height) * SPIRAL_RADIUS * spiralScale;
    const ms = p.millis() * ANIM_SPEED;
    const timer = (ms + 10035) / SPIRAL_PERIOD;

    p.push();
    p.translate(p.width / 2, p.height / 2);
    p.rotate(handRot);

    for (let i = 0; i < layers; i++) {
      const rotation = (Math.PI / leaf) * timer * i + handRot * 0.12;
      const perlin = Math.pow(p.noise(ms / 600.0 + i * 0.2), 3.4);
      const radius = size_ * (1 - i / layers);

      for (let j = 0; j < leaf; j++) {
        const c =
          Math.cos(
            p.TAU *
              ((Math.sin((Math.PI * ms) / 120000) + 2) * j +
                0.1 * p.noise(ms / 100.0)) /
              leaf +
              rotation
          ) / 1.4;
        const d =
          Math.sin(
            p.TAU *
              ((1 + p.noise(ms / 10000.0) / 2) * j) / leaf +
              rotation
          ) / (1.2 * j + 0.1 * p.noise(ms / 100.0));

        const side =
          Math.max(16, (radius * perlin) / PARTICLE_DIV) * Math.sqrt(spiralScale);
        if (side < MIN_DRAW_SIDE) continue;

        const feat = pool[(i * leaf + j) % pool.length];
        drawFeatureBox(p, feat, c * radius, d * radius, side, i * leaf + j, p.millis());
      }
    }
    p.pop();
  };
});

async function boot(p, live) {
  if (booted) return;
  booted = true;
  useLive = live;
  showHud("请求摄像头…");

  if (!live) {
    copyRefIntoFeatures(p);
    featureReady = true;
    showHud("演示模式 · 点一下可全屏");
    return;
  }

  try {
    const { stream, label, index, devices } = await openPreferredCamera();
    await attachStream(stream, label, index, devices);
    camReady = true;
    showHud(`摄像头 · ${label} · 加载识别…`);
    await initLandmarkers();
    if (!featureReady) {
      copyRefIntoFeatures(p);
      featureReady = true;
    }
    showHud(`Webcam · ${label} · 按 C 切换摄像头 · 手掌转动/推拉`, 10000);
  } catch (err) {
    console.warn(err);
    useLive = false;
    copyRefIntoFeatures(p);
    featureReady = true;
    showHud("摄像头被拒绝或不可用 · 参考图模式", 8000);
  }
}

function updateDebug() {
  if (!debugEl) return;
  const vw = videoEl?.videoWidth || 0;
  const vh = videoEl?.videoHeight || 0;
  const faceOk = faceSeen && performance.now() - lastFaceOkAt < 800;
  debugEl.innerHTML =
    `<b>${currentCamLabel || "未开摄像头"}</b><br>` +
    `${vw}×${vh} · 脸 ${faceOk ? "有" : "无"}<br>` +
    `C 换摄像头 · D 隐藏预览`;
}

function showHud(text, hideAfter = 0) {
  if (debugEl && text) {
    // 关键也写进 debug，因为左下角 HUD 已隐藏
    currentCamLabel = currentCamLabel || text;
  }
  if (!statusEl) return;
  statusEl.textContent = text;
  statusEl.classList.remove("dim");
  if (hudHideTimer) clearTimeout(hudHideTimer);
  if (hideAfter > 0) {
    hudHideTimer = setTimeout(() => statusEl?.classList.add("dim"), hideAfter);
  }
}

function stopCurrentStream() {
  if (currentStream) {
    currentStream.getTracks().forEach((t) => t.stop());
    currentStream = null;
  }
}

async function attachStream(stream, label, index, devices) {
  stopCurrentStream();
  currentStream = stream;
  videoDevices = devices || videoDevices;
  currentCamIndex = index ?? currentCamIndex;
  videoEl.srcObject = stream;
  videoEl.muted = true;
  videoEl.playsInline = true;

  await new Promise((resolve) => {
    const done = () => {
      videoEl.removeEventListener("loadeddata", done);
      videoEl.removeEventListener("playing", done);
      resolve();
    };
    if (videoEl.readyState >= 2) resolve();
    else {
      videoEl.addEventListener("loadeddata", done, { once: true });
      videoEl.addEventListener("playing", done, { once: true });
      setTimeout(resolve, 1500);
    }
  });

  try {
    await videoEl.play();
  } catch (_) {}

  // 等真正有宽高
  for (let i = 0; i < 30 && (!videoEl.videoWidth || !videoEl.videoHeight); i++) {
    await new Promise((r) => setTimeout(r, 50));
  }

  if (camLabelEl) {
    camLabelEl.textContent = `${currentCamIndex + 1}/${videoDevices.length} ${label}`;
  }
  try {
    localStorage.setItem("face-bloom-cam", stream.getVideoTracks()[0]?.getSettings?.().deviceId || "");
  } catch (_) {}

  console.log(
    "[face-bloom] stream live:",
    label,
    videoEl.videoWidth,
    "x",
    videoEl.videoHeight,
    "readyState",
    videoEl.readyState
  );
  currentCamLabel = label;
  updateDebug();
}

function isBuiltinCameraLabel(label) {
  const s = (label || "").toLowerCase();
  return (
    s.includes("facetime") ||
    s.includes("built-in") ||
    s.includes("builtin") ||
    s.includes("isight") ||
    s.includes("continuity") ||
    s.includes("iphone") ||
    s.includes("ipad") ||
    s.includes("desk view") ||
    s.includes("macbook") ||
    s.includes("macbook pro") ||
    /^apple\b/.test(s)
  );
}

function cameraScore(label) {
  const s = (label || "").toLowerCase();
  let score = 0;
  if (isBuiltinCameraLabel(s)) score -= 200;
  if (s.includes("logitech") || s.includes("c930") || s.includes("c920") || s.includes("c922") || s.includes("brio")) {
    score += 120;
  }
  if (s.includes("usb")) score += 40;
  if (s.includes("webcam")) score += 50;
  if (s.includes("hd") || s.includes("1080") || s.includes("720")) score += 10;
  if (s.includes("capture") || s.includes("obs")) score += 15;
  if (s.includes("camera") || s.includes("相机")) score += 5;
  return score;
}

async function listVideoDevices() {
  // 权限探测：尽量先摸一下外设，避免只拿到内置
  let probe;
  try {
    probe = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
  } catch (_) {
    probe = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  }
  probe.getTracks().forEach((t) => t.stop());

  const cams = (await navigator.mediaDevices.enumerateDevices()).filter(
    (d) => d.kind === "videoinput"
  );
  if (!cams.length) throw new Error("no videoinput");
  return cams;
}

async function openCameraByDeviceId(deviceId) {
  try {
    return await navigator.mediaDevices.getUserMedia({
      video: {
        deviceId: { exact: deviceId },
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });
  } catch (_) {
    return await navigator.mediaDevices.getUserMedia({
      video: { deviceId: { exact: deviceId } },
      audio: false,
    });
  }
}

async function openPreferredCamera() {
  try {
    localStorage.removeItem("face-bloom-cam");
  } catch (_) {}

  const cams = await listVideoDevices();
  console.log(
    "[face-bloom] all cameras:",
    cams.map((c) => c.label || "(no label)")
  );

  const isWantedExternal = (label) => {
    const s = (label || "").toLowerCase();
    if (isBuiltinCameraLabel(s)) return false;
    return (
      s.includes("logitech") ||
      s.includes("c930") ||
      s.includes("c920") ||
      s.includes("webcam") ||
      s.includes("usb") ||
      s.includes("hd pro") ||
      s.includes("brio")
    );
  };

  // 只开外接；有 Logitech 时绝不回退 MacBook
  const external = cams
    .map((d, i) => ({ d, i, score: cameraScore(d.label) }))
    .filter((r) => !isBuiltinCameraLabel(r.d.label))
    .sort((a, b) => b.score - a.score);

  const wanted = external.filter((r) => isWantedExternal(r.d.label));
  const order = (wanted.length ? wanted : external).map((r) => r.i);

  if (!order.length) {
    // 真的没有外接才用内置
    order.push(...cams.map((_, i) => i));
  }

  let lastErr;
  for (const idx of order) {
    try {
      const stream = await openCameraByDeviceId(cams[idx].deviceId);
      const label = stream.getVideoTracks()[0]?.label || cams[idx].label || `camera ${idx + 1}`;
      // 双保险：外接列表非空时拒绝内置
      if (isBuiltinCameraLabel(label) && external.length) {
        stream.getTracks().forEach((t) => t.stop());
        continue;
      }
      console.log("[face-bloom] using:", label);
      return { stream, label, index: idx, devices: cams };
    } catch (err) {
      lastErr = err;
      console.warn("[face-bloom] failed", cams[idx].label, err);
    }
  }
  throw lastErr || new Error("no camera");
}

async function cycleCamera() {
  if (!videoDevices.length) {
    try {
      videoDevices = await listVideoDevices();
    } catch (err) {
      showHud("无法列出摄像头", 4000);
      return;
    }
  }
  if (videoDevices.length < 2) {
    showHud(`只有 1 个摄像头：${videoDevices[0]?.label || "unknown"}`, 5000);
    return;
  }

  // 优先在外接摄像头之间切换
  const externalIdx = videoDevices
    .map((d, i) => ({ d, i }))
    .filter((x) => !isBuiltinCameraLabel(x.d.label))
    .map((x) => x.i);
  const pool = externalIdx.length >= 1 ? externalIdx : videoDevices.map((_, i) => i);

  const pos = pool.indexOf(currentCamIndex);
  const next = pool[(pos + 1) % pool.length];
  showHud(`切换摄像头 ${next + 1}/${videoDevices.length}…`);
  try {
    const stream = await openCameraByDeviceId(videoDevices[next].deviceId);
    const label =
      stream.getVideoTracks()[0]?.label || videoDevices[next].label || `camera ${next + 1}`;
    await attachStream(stream, label, next, videoDevices);
    basePalmSize = null;
    showHud(`已切换 · ${label} · 再按 C 换下一个`, 8000);
  } catch (err) {
    console.warn(err);
    showHud(`切换失败 · ${videoDevices[next]?.label || "camera"}`, 5000);
  }
}

function enterFullscreen() {
  const root = document.documentElement;
  if (!document.fullscreenElement && root.requestFullscreen) {
    root.requestFullscreen().catch(() => {});
  }
}

function toggleFullscreen() {
  if (document.fullscreenElement) {
    document.exitFullscreen?.();
  } else {
    enterFullscreen();
  }
}

async function initLandmarkers() {
  const { FaceLandmarker, HandLandmarker, FilesetResolver } = await import(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm"
  );
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numFaces: 1,
  });

  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 1,
  });
}

function detectHand(p) {
  const now = performance.now();
  if (now - lastHandDetect < 50) return;
  lastHandDetect = now;
  if (videoEl.readyState < 2) return;

  const result = handLandmarker.detectForVideo(videoEl, now);
  const hands = result?.landmarks || [];

  if (!hands.length) {
    targetScale = lerp(targetScale, 1, 0.06);
    if (handSeen) {
      statusEl.textContent = "把手掌放进画面 · 转动 / 推近拉远";
    }
    return;
  }

  handSeen = true;
  const h = hands[0];

  // —— 旋转：手腕(0) → 中指尖(12) 的朝向 ——
  // 镜像：摄像头左右翻转，x 取反使手感一致
  const ang = Math.atan2(h[12].y - h[0].y, (1 - h[12].x) - (1 - h[0].x));
  // 相对竖直朝上的偏移，让“手自然竖着”≈ 0 旋转
  const upright = ang + Math.PI / 2;
  targetRot = clamp(upright, -Math.PI * 0.85, Math.PI * 0.85);

  // —— 缩放：掌面在画面中的大小 ≈ 远近 ——
  // 用拇指 MCP(2) 到小指 MCP(17) + 手腕到中指 MCP(9) 综合
  const palmW = Math.hypot(h[5].x - h[17].x, h[5].y - h[17].y);
  const palmH = Math.hypot(h[0].x - h[9].x, h[0].y - h[9].y);
  const palmSize = (palmW + palmH) * 0.5;

  if (basePalmSize == null) {
    basePalmSize = palmSize;
  }
  // 缓慢校准：缓慢贴近当前大小，避免换手后锁死
  basePalmSize = lerp(basePalmSize, palmSize, 0.004);

  const ratio = palmSize / Math.max(0.04, basePalmSize);
  // 推近 → 放大，拉远 → 缩小
  targetScale = clamp(Math.pow(ratio, 1.15), MIN_SPIRAL_SCALE, 2.6);

  const deg = Math.round((handRot * 180) / Math.PI);
  statusEl.textContent = `手掌 · 旋转 ${deg}° · 缩放 ${targetScale.toFixed(2)}×`;
}

function seedFeatureCanvases(p) {
  featureCanvases = FEATURE_SETS.map(() => {
    const g = p.createGraphics(FEATURE_RES, FEATURE_RES);
    g.pixelDensity(1);
    return g;
  });
  featurePool = [];
  for (let i = 0; i < POOL_SLOTS; i++) {
    const g = p.createGraphics(FEATURE_RES, FEATURE_RES);
    g.pixelDensity(1);
    featurePool.push(g);
  }
  poolWrite = 0;
}

function pushPoolFromLive(p) {
  // 每帧把若干五官写入滚动池，螺旋里能看到更多切片
  const n = FEATURE_SETS.length;
  const batch = Math.min(6, n);
  const start = poolWrite % n;
  for (let k = 0; k < batch; k++) {
    const src = featureCanvases[(start + k) % n];
    const dst = featurePool[poolWrite % featurePool.length];
    dst.clear();
    dst.image(src, 0, 0, dst.width, dst.height);
    poolWrite++;
  }
  featureReady = true;
}

function copyRefIntoFeatures(p) {
  if (!refCrops.length) return;
  for (let i = 0; i < featureCanvases.length; i++) {
    const src = refCrops[i % refCrops.length];
    const g = featureCanvases[i];
    g.clear();
    g.image(src, 0, 0, g.width, g.height);
  }
  for (let i = 0; i < featurePool.length; i++) {
    const src = refCrops[i % refCrops.length];
    const g = featurePool[i];
    g.clear();
    g.image(src, 0, 0, g.width, g.height);
  }
  featureReady = true;
}

function extractRefCrops(p) {
  if (!refImg || !refImg.width) return;
  refImg.loadPixels();
  const w = refImg.width;
  const h = refImg.height;
  const step = 3;
  const visited = new Uint8Array(w * h);
  const boxes = [];

  const isInk = (x, y) => {
    const i = 4 * (y * w + x);
    const r = refImg.pixels[i];
    const g = refImg.pixels[i + 1];
    const b = refImg.pixels[i + 2];
    return r > 18 || g > 18 || b > 18;
  };

  for (let y = 0; y < h; y += step) {
    for (let x = 0; x < w; x += step) {
      const id = y * w + x;
      if (visited[id] || !isInk(x, y)) continue;

      let minX = x,
        maxX = x,
        minY = y,
        maxY = y,
        count = 0;
      const queue = [x, y];
      visited[id] = 1;
      let qi = 0;
      while (qi < queue.length && count < 40000) {
        const cx = queue[qi++];
        const cy = queue[qi++];
        count++;
        if (cx < minX) minX = cx;
        if (cx > maxX) maxX = cx;
        if (cy < minY) minY = cy;
        if (cy > maxY) maxY = cy;
        for (let dy = -step; dy <= step; dy += step) {
          for (let dx = -step; dx <= step; dx += step) {
            if (!dx && !dy) continue;
            const nx = cx + dx;
            const ny = cy + dy;
            if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
            const nid = ny * w + nx;
            if (visited[nid]) continue;
            visited[nid] = 1;
            if (!isInk(nx, ny)) continue;
            queue.push(nx, ny);
          }
        }
      }

      const bw = maxX - minX + 1;
      const bh = maxY - minY + 1;
      if (bw < 24 || bh < 24 || count < 40) continue;
      if (bw > w * 0.65 || bh > h * 0.65) continue;
      boxes.push({ x: minX, y: minY, w: bw, h: bh, count });
    }
  }

  boxes.sort((a, b) => b.count - a.count);
  const top = boxes.slice(0, 20);
  refCrops = top.map((b) => {
    const g = p.createGraphics(Math.max(48, b.w), Math.max(48, b.h));
    g.pixelDensity(1);
    g.image(refImg, 0, 0, g.width, g.height, b.x, b.y, b.w, b.h);
    return g;
  });

  if (refCrops.length) copyRefIntoFeatures(p);
}

function getMirroredVideoFrame(vw, vh) {
  if (!detectAndCrop._buf) {
    detectAndCrop._buf = document.createElement("canvas");
    detectAndCrop._ctx = detectAndCrop._buf.getContext("2d", { willReadFrequently: true });
  }
  const buf = detectAndCrop._buf;
  const ctx = detectAndCrop._ctx;
  if (buf.width !== vw || buf.height !== vh) {
    buf.width = vw;
    buf.height = vh;
  }
  ctx.save();
  ctx.scale(-1, 1);
  ctx.drawImage(videoEl, -vw, 0, vw, vh);
  ctx.restore();
  return buf;
}

function detectAndCrop(p) {
  const now = performance.now();
  if (now - lastFaceDetect < 90) return;
  lastFaceDetect = now;
  if (videoEl.readyState < 2) return;

  const result = faceLandmarker.detectForVideo(videoEl, now);
  if (result?.faceLandmarks?.length) {
    lastFaceLm = result.faceLandmarks[0];
    lastFaceAt = now;
    faceSeen = true;
    lastFaceOkAt = now;
  } else {
    faceSeen = false;
  }

  const lm = lastFaceLm;
  if (!lm || now - lastFaceAt > FACE_HOLD_MS) return;
  const vw = videoEl.videoWidth;
  const vh = videoEl.videoHeight;
  const buf = getMirroredVideoFrame(vw, vh);

  const batch = 4;
  if (detectAndCrop._fi == null) detectAndCrop._fi = 0;
  for (let k = 0; k < batch; k++) {
    const fi = (detectAndCrop._fi + k) % FEATURE_SETS.length;
    const set = FEATURE_SETS[fi];
    let minX = 1,
      minY = 1,
      maxX = 0,
      maxY = 0;
    for (const i of set.idx) {
      const pt = lm[i];
      const x = 1 - pt.x;
      const y = pt.y;
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
    }
    const bw = maxX - minX;
    const bh = maxY - minY;
    const padX = bw * set.pad * CROP_PAD_MUL;
    const padY = bh * set.pad * CROP_PAD_MUL;
    minX = Math.max(0, minX - padX);
    minY = Math.max(0, minY - padY);
    maxX = Math.min(1, maxX + padX);
    maxY = Math.min(1, maxY + padY);

    const sx = Math.floor(minX * vw);
    const sy = Math.floor(minY * vh);
    const sw = Math.max(4, Math.floor((maxX - minX) * vw));
    const sh = Math.max(4, Math.floor((maxY - minY) * vh));

    const g = featureCanvases[fi];
    g.clear();
    const side = Math.floor(Math.max(sw, sh) * CROP_SQUARE_MUL);
    const cx = sx + sw / 2;
    const cy = sy + sh / 2;
    const s0 = Math.max(0, Math.floor(cx - side / 2));
    const s1 = Math.max(0, Math.floor(cy - side / 2));
    const ss = Math.min(side, vw - s0, vh - s1);
    g.drawingContext.drawImage(buf, s0, s1, ss, ss, 0, 0, g.width, g.height);
  }
  detectAndCrop._fi = (detectAndCrop._fi + batch) % FEATURE_SETS.length;
  pushPoolFromLive(p);
}

function boxTelemetrySeed(boxId) {
  return (boxId * 2654435761) % 100000;
}

function drawFeatureBoxNumbers(p, side, boxId, now, cr, cg, cb) {
  if (side < 34) return;

  const half = side / 2;
  const tick = Math.max(4, side * 0.055);
  const gap = Math.max(3, side * 0.06);
  const fs = Math.max(5.5, side * 0.062);
  const seed = boxTelemetrySeed(boxId);
  const tickFrame = Math.floor(now / 140);

  p.textFont("monospace");
  p.textSize(fs);
  p.noStroke();

  const labelAt = (lx, ly, alignX, alignY, n) => {
    p.textAlign(alignX, alignY);
    p.fill(14, 14, 14, 210);
    p.text(n, lx, ly);
  };

  const numFor = (i) => {
    const v = (seed + i * 97 + tickFrame * 3) % 10000;
    return String(v).padStart(4, "0");
  };

  p.stroke(cr, cg, cb, 120);
  p.strokeWeight(Math.max(0.45, side * 0.007));

  const tops = [0.32, 0.68];
  tops.forEach((t, i) => {
    const px = -half + t * side;
    p.line(px, -half, px, -half + tick * 0.45);
    labelAt(px, -half - gap, p.CENTER, p.BOTTOM, numFor(i));
  });
}

function drawFeatureBox(p, feat, x, y, side, boxId, now) {
  p.push();
  p.translate(x, y);

  p.noStroke();
  p.tint(255, 230);
  p.image(feat, 0, 0, side, side);
  p.noTint();

  const [cr, cg, cb] = BOX_STROKE;
  p.noFill();
  p.stroke(cr, cg, cb, 200);
  p.strokeWeight(Math.max(1, side * 0.035));
  p.rect(0, 0, side, side);

  const half = side / 2;
  p.stroke(cr, cg, cb, 160);
  p.strokeWeight(Math.max(0.8, side * 0.02));
  p.line(-half, -half, half, half);
  p.line(half, -half, -half, half);

  drawFeatureBoxNumbers(p, side, boxId, now, cr, cg, cb);

  p.pop();
}
