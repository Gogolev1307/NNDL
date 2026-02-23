/**
 * Neural Network Design: PERFECT CHESS GRADIENT - С ИСПРАВЛЕННЫМ BASELINE
 * Baseline учится копировать вход (MSE), Student создает шахматный градиент
 */

// ==========================================
// 1. Global State & Config
// ==========================================
const CONFIG = {
  inputShapeModel: [16, 16, 1],
  inputShapeData: [1, 16, 16, 1],
  learningRate: 0.01,
  autoTrainSpeed: 40,
  lambdaSmooth: 0.02,
  lambdaDir: 8.0,
  lambdaChess: 3.0,
};

let state = {
  step: 0,
  isAutoTraining: false,
  autoTrainTimeout: null,
  xInput: null,
  baselineModel: null,
  studentModel: null,
  optimizerBaseline: null,  // Отдельный оптимизатор для Baseline
  optimizerStudent: null,    // Отдельный оптимизатор для Student
};

// ==========================================
// 2. Loss Functions
// ==========================================

function mse(yTrue, yPred) {
  return tf.losses.meanSquaredError(yTrue, yPred);
}

function smoothness(yPred) {
  return tf.tidy(() => {
    const diffX = yPred.slice([0, 0, 0, 0], [-1, -1, 15, -1])
      .sub(yPred.slice([0, 0, 1, 0], [-1, -1, 15, -1]));
    return tf.mean(tf.square(diffX)).mul(tf.scalar(0.1));
  });
}

// Плавный градиент
function gradientLoss(yPred) {
  return tf.tidy(() => {
    const height = 16;
    const width = 16;
    
    const target = [];
    for (let i = 0; i < height; i++) {
      for (let j = 0; j < width; j++) {
        target.push(j / (width - 1));
      }
    }
    const targetTensor = tf.tensor(target).reshape([1, height, width, 1]);
    
    return tf.losses.meanSquaredError(targetTensor, yPred).mean();
  });
}

// Шахматный узор через сравнение соседей
function chessNeighborLoss(yPred) {
  return tf.tidy(() => {
    const left = yPred.slice([0, 0, 0, 0], [-1, -1, 15, -1]);
    const right = yPred.slice([0, 0, 1, 0], [-1, -1, 15, -1]);
    
    const horizontalDiff = tf.abs(left.sub(right));
    const horizontalLoss = tf.square(horizontalDiff.sub(tf.scalar(0.3))).mean();
    
    const top = yPred.slice([0, 0, 0, 0], [-1, 15, -1, -1]);
    const bottom = yPred.slice([0, 1, 0, 0], [-1, 15, -1, -1]);
    
    const verticalDiff = tf.abs(top.sub(bottom));
    const verticalLoss = tf.square(verticalDiff.sub(tf.scalar(0.3))).mean();
    
    const diag1 = yPred.slice([0, 0, 0, 0], [-1, 15, 15, -1]);
    const diag2 = yPred.slice([0, 1, 1, 0], [-1, 15, 15, -1]);
    
    const diagonalDiff = tf.abs(diag1.sub(diag2));
    const diagonalLoss = diagonalDiff.square().mean().mul(tf.scalar(2));
    
    return horizontalLoss.add(verticalLoss).add(diagonalLoss).div(tf.scalar(3));
  });
}

// ==========================================
// 3. Model Architecture
// ==========================================
function createBaselineModel() {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: CONFIG.inputShapeModel }));
  model.add(tf.layers.dense({ units: 64, activation: "relu" }));
  model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

function createStudentModel(archType) {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: CONFIG.inputShapeModel }));

  if (archType === "compression") {
    model.add(tf.layers.dense({ units: 32, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  } else if (archType === "transformation") {
    model.add(tf.layers.dense({ units: 128, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  } else {
    model.add(tf.layers.dense({ units: 512, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  }

  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

// ==========================================
// 4. Loss Functions для каждой модели
// ==========================================
function baselineLoss(yTrue, yPred) {
  return tf.tidy(() => {
    return mse(yTrue, yPred).mean();
  });
}

function studentLoss(yTrue, yPred) {
  return tf.tidy(() => {
    const lossGradient = gradientLoss(yPred).mul(CONFIG.lambdaDir);
    const lossChess = chessNeighborLoss(yPred).mul(CONFIG.lambdaChess);
    const lossSmooth = smoothness(yPred).mul(CONFIG.lambdaSmooth);
    
    return lossGradient.add(lossChess).add(lossSmooth);
  });
}

// ==========================================
// 5. Training Loop
// ==========================================
async function trainStep() {
  if (!state.baselineModel || !state.studentModel) {
    log("Models not initialized", true);
    return;
  }
  
  state.step++;
  
  try {
    // Обучаем Baseline (копирование входа) - отдельный оптимизатор
    const baselineGrads = tf.variableGrads(() => {
      const pred = state.baselineModel.predict(state.xInput);
      return baselineLoss(state.xInput, pred);
    }, state.baselineModel.getWeights()).grads;
    
    state.optimizerBaseline.applyGradients(baselineGrads);
    
    // Обучаем Student (шахматный градиент) - отдельный оптимизатор
    const studentGrads = tf.variableGrads(() => {
      const pred = state.studentModel.predict(state.xInput);
      return studentLoss(state.xInput, pred);
    }, state.studentModel.getWeights()).grads;
    
    state.optimizerStudent.applyGradients(studentGrads);
    
    // Считаем потери для отображения
    const [bLoss, sLoss] = tf.tidy(() => {
      const basePred = state.baselineModel.predict(state.xInput);
      const studPred = state.studentModel.predict(state.xInput);
      
      return [
        baselineLoss(state.xInput, basePred),
        studentLoss(state.xInput, studPred)
      ];
    });
    
    const baselineLossVal = bLoss.dataSync()[0];
    const studentLossVal = sLoss.dataSync()[0];
    
    bLoss.dispose();
    sLoss.dispose();
    
    await render();
    updateLossDisplay(baselineLossVal, studentLossVal);
    
    if (state.step % 10 === 0) {
      log(`Step ${state.step}: Base Loss=${baselineLossVal.toFixed(4)} | Student Loss=${studentLossVal.toFixed(4)}`);
    }
    
  } catch (e) {
    log(`Error: ${e.message}`, true);
    console.error(e);
    stopAutoTrain();
  }
}

// ==========================================
// 6. UI Functions
// ==========================================
function init() {
  if (state.baselineModel) state.baselineModel.dispose();
  if (state.studentModel) state.studentModel.dispose();
  if (state.xInput) state.xInput.dispose();
  
  state.xInput = tf.randomUniform(CONFIG.inputShapeData, 0, 1);
  resetModels();
  
  tf.browser.toPixels(
    state.xInput.squeeze(), 
    document.getElementById("canvas-input")
  ).catch(e => log(`Render error: ${e.message}`, true));
  
  document.getElementById("btn-train").onclick = trainStep;
  document.getElementById("btn-auto").onclick = toggleAutoTrain;
  document.getElementById("btn-reset").onclick = resetModels;
  
  document.querySelectorAll('input[name="arch"]').forEach(radio => {
    radio.onchange = (e) => {
      resetModels(e.target.value);
      document.getElementById("student-arch-label").innerText = 
        e.target.value.charAt(0).toUpperCase() + e.target.value.slice(1);
    };
  });
  
  log("✅ PERFECT CHESS GRADIENT");
  log("   Baseline: копирует вход (MSE)");
  log("   Student: создает шахматный градиент");
}

function resetModels(archType = null) {
  stopAutoTrain();
  
  if (state.baselineModel) state.baselineModel.dispose();
  if (state.studentModel) state.studentModel.dispose();
  if (state.optimizerBaseline) state.optimizerBaseline.dispose();
  if (state.optimizerStudent) state.optimizerStudent.dispose();
  
  const arch = archType || document.querySelector('input[name="arch"]:checked')?.value || "transformation";
  
  state.baselineModel = createBaselineModel();
  state.studentModel = createStudentModel(arch);
  
  // Отдельные оптимизаторы для каждой модели
  state.optimizerBaseline = tf.train.adam(CONFIG.learningRate);
  state.optimizerStudent = tf.train.adam(CONFIG.learningRate);
  
  state.step = 0;
  
  render();
  log(`🔄 Reset: "${arch}" architecture`);
}

async function render() {
  try {
    const basePred = state.baselineModel.predict(state.xInput);
    const studPred = state.studentModel.predict(state.xInput);
    
    await Promise.all([
      tf.browser.toPixels(basePred.squeeze(), document.getElementById("canvas-baseline")),
      tf.browser.toPixels(studPred.squeeze(), document.getElementById("canvas-student"))
    ]);
    
    basePred.dispose();
    studPred.dispose();
  } catch (e) {
    log(`Render error: ${e.message}`, true);
  }
}

function updateLossDisplay(base, stud) {
  document.getElementById("loss-baseline").innerText = `Loss: ${base.toFixed(5)}`;
  document.getElementById("loss-student").innerText = `Loss: ${stud.toFixed(5)}`;
}

function log(msg, isError = false) {
  const el = document.getElementById("log-area");
  const div = document.createElement("div");
  div.innerText = `> ${msg}`;
  if (isError) div.classList.add("error");
  el.prepend(div);
  while (el.children.length > 10) {
    el.removeChild(el.lastChild);
  }
}

function toggleAutoTrain() {
  const btn = document.getElementById("btn-auto");
  if (state.isAutoTraining) {
    stopAutoTrain();
  } else {
    state.isAutoTraining = true;
    btn.innerText = "⏹️ Stop";
    btn.classList.add("btn-stop");
    btn.classList.remove("btn-auto");
    loop();
  }
}

function stopAutoTrain() {
  state.isAutoTraining = false;
  const btn = document.getElementById("btn-auto");
  btn.innerText = "▶️ Auto Train";
  btn.classList.add("btn-auto");
  btn.classList.remove("btn-stop");
  if (state.autoTrainTimeout) {
    clearTimeout(state.autoTrainTimeout);
    state.autoTrainTimeout = null;
  }
}

function loop() {
  if (state.isAutoTraining) {
    trainStep();
    state.autoTrainTimeout = setTimeout(loop, CONFIG.autoTrainSpeed);
  }
}

// Start
init();