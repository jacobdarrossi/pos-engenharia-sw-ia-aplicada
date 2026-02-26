import { warn } from "pixi.js";

importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest');

const MODEL_PATH = `yolov5n_web_model/model.json`;
const LABELS_PATH = `yolov5n_web_model/labels.json`;
const INPUT_MODEL_DIMENSIONS = 640;
const CLASS_THRESHOLD = 0.4; // Limite de confiança para considerar uma previsão válida

let _labels = []
let _model = null
async function loadModelAndLabels() {
    await tf.ready();

    _labels = await fetch(LABELS_PATH).then(res => res.json());
    _model = await tf.loadGraphModel(MODEL_PATH);
    
    //warmup
    const dummyInput = tf.ones(_model.inputs[0].shape)
    console.log(_model.inputs[0].shape)

    await _model.executeAsync(dummyInput);
    dummyInput.dispose();
    
    postMessage({ type: 'modelReady' });
}

/** 
 * Pré processa a imagem para o formato esperado pelo modelo YOLOv5n.
 * - tf.browser.fromPixels converte a imagem para um tensor. [H, W, 3]
 * - tf.image.resizeBilinear redimensiona a imagem para as dimensões de entrada do modelo (640x640).
 * - .div(255) normaliza os valores dos pixels para o intervalo [0, 1].
 * - .expandDims(0) adiciona uma dimensão de lote, resultando em um tensor de forma [1, 640, 640, 3].
 *  Redimensiona a imagem para 640x640, normaliza os pixels para o intervalo [0, 1] e adiciona uma dimensão de lote.
 *  Uso de tf.tidy para garantir que os tensores intermediários sejam descartados e evitar vazamentos de memória.
 */
function preprocessImage(imageBitmap) {
    return tf.tidy(() => {
        const imgTensor = tf.browser.fromPixels(imageBitmap);

        return tf.image
            .resizeBilinear(imgTensor, [INPUT_MODEL_DIMENSIONS, INPUT_MODEL_DIMENSIONS])
            .div(255)
            .expandDims(0);
    });
}


/**
 * Executa a inferência usando o modelo carregado e processa os resultados.
 * - _model.executeAsync(tensor) executa a inferência no tensor de entrada e retorna as saídas do modelo.
 * - tf.dispose(tensor) libera a memória ocupada pelo tensor de entrada após a inferência.
 * - O código extrai as caixas delimitadoras, pontuações e classes das saídas do modelo, assumindo que as 3 primeiras saídas correspondem a essas informações.
 * - As saídas são convertidas para arrays usando .data()
 * - Finalmente, os tensores de saída são descartados para liberar memória, e os resultados processados são retornados como um objeto contendo as caixas, pontuações e classes.
 *  Executa a inferência no tensor de entrada, processa as saídas para extrair caixas, pontuações e classes, e garante que os tensores sejam descartados para evitar vazamentos de memória.
 */
async function runInference(tensor) {
    const output = await _model.executeAsync(tensor);
    tf.dispose(tensor)

    // Assume que as 3 primeiras saídas são caixas, pontuações e classes, respectivamente
    const [boxes, scores, classes] = output.slice(0, 3)
    const [boxesData, scoresData, classesData] = await Promise.all([
        boxes.data(),
        scores.data(),
        classes.data()
    ]);

    output.forEach(t => t.dispose());
    
    return { boxes: boxesData, scores: scoresData, classes: classesData };
}

function* postPrediction({boxes, scores, classes}, width, height) {
    for (let i = 0; i < scores.length; i++) {
        if (scores[i] < CLASS_THRESHOLD) continue // Filtra previsões com baixa confiança
    
        if (_labels[classes[i]] !== 'kite') continue // Filtra previsões que não sejam de patos
    
        let [x1, y1, x2, y2] = boxes.slice(i * 4, (i+1) *4); // Extrai as coordenadas da caixa delimitadora para a previsão atual
        x1 *= width
        x2 *= width
        y1 *= height
        y2 *= height

        const boxWidth = x2 - x1
        const boxHeight = y2 - y1
        const centerX = x1 + boxWidth / 2
        const centerY = y1 + boxHeight / 2
        yield {
            x: centerX,
            y: centerY,
            score: (scores[i]*100).toFixed(2),
        }
    }
}

loadModelAndLabels()

self.onmessage = async ({ data }) => {
    if (data.type !== 'predict') return
    if (!_model) return
    
    const input = preprocessImage(data.image);
    const {width, height} = data.image
    const inferenceResults = await runInference(input);

    for (const prediction of postPrediction(inferenceResults, width, height)) {
        postMessage({
            type: 'prediction',
            x: prediction.x,
            y: prediction.y,
            score: prediction.score
        });
    }


};

console.log('🧠 YOLOv5n Web Worker initialized');
