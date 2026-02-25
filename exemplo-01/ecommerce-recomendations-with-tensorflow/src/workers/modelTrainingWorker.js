import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';
let _globalCtx = {};
let _model = null

const WEIGHTS = {
    category: 0.4,
    color: 0.3,
    price: 0.2,
    age: 0.1
}
// 🔢 Normalize continuous values (price, age) to 0–1 range
// Why? Keeps all features balanced so no one dominates training
// Formula: (val - min) / (max - min)
// Example: price=129.99, minPrice=39.99, maxPrice=199.99 → 0.56
const normalize = (value, min, max) => (value - min) / ((max - min) || 1)

function makeContext(products, users) {
    const ages = users.map(u => u.age)
    const prices = products.map(p => p.price)

    // obter o range de idades e preços para normalizar depois
    const minAge = Math.min(...ages)
    const maxAge = Math.max(...ages)

    //obter o range de preços para normalizar depois
    const minPrice = Math.min(...prices)
    const maxPrice = Math.max(...prices)

    // obter categorias e cores únicas para one-hot encoding (transformar em colunas binárias) posteriormente
    const colors = [...new Set(products.map(p => p.color))]
    const categories = [...new Set(products.map(p => p.category))]

    // criar índices para converter categorias e cores em números (ex: "red" → 0, "blue" → 1)
    const colorsIndex = Object.fromEntries(
        colors.map((color, index) => {
            return [color, index]
        }))
    const categoriesIndex = Object.fromEntries(
        categories.map((category, index) => {
            return [category, index]
        }))

    // Computar a média de idade dos comprados por produto
    // (ajuda a personalizar)
    // Exemplo: se "Tênis de Corrida" tem média de idade 25, e "Cadeira de Escritório" tem média de idade 45,
    // isso indica que o primeiro é mais popular entre os jovens e o segundo entre os mais velhos.
    //  O modelo pode usar essa informação para recomendar produtos mais alinhados com a faixa etária do usuário.
    const midAge = (minAge + maxAge) / 2
    const ageSums = {}
    const ageCounts = {}

    // Para cada produto, calcular a soma das idades dos compradores e o número de compradores
    users.forEach(user => {
        user.purchases.forEach(p => {
            ageSums[p.name] = (ageSums[p.name] || 0) + user.age
            ageCounts[p.name] = (ageCounts[p.name] || 0) + 1
        })
    })

    // Calcular a média de idade para cada produto e normalizar para 0–1
    const productAvgAgeNorm = Object.fromEntries(
        products.map(product => {
            const avg = ageCounts[product.name] ?
                ageSums[product.name] / ageCounts[product.name] :
                midAge

            return [product.name, normalize(avg, minAge, maxAge)]
        })
    )

    // O contexto é um objeto que contém todas as informações necessárias para treinar o modelo de recomendação
    return {
        products,
        users,
        colorsIndex,
        categoriesIndex,
        productAvgAgeNorm,
        minAge,
        maxAge,
        minPrice,
        maxPrice,
        numCategories: categories.length,
        numColors: colors.length,
        // price + age + categories + colors
        dimensions: 2 + categories.length + colors.length
    }
}
// 🔄 One-hot encode categorical features (category, color) with weights
const oneHotWeighted = (index, length, weight) =>
    tf.oneHot(index, length).cast('float32').mul(weight)

function encodeProduct(product, context) {
    // normalizando dados para ficar de 0 a 1 e multiplicando pelo peso de cada característica
    const price = tf.tensor1d([normalize(product.price, context.minPrice, context.maxPrice) * WEIGHTS.price])
    const age = tf.tensor1d([context.productAvgAgeNorm[product.name] ?? 0.5 * WEIGHTS.age])
    const category = oneHotWeighted(context.categoriesIndex[product.category], context.numCategories, WEIGHTS.category)
    const color = oneHotWeighted(context.colorsIndex[product.color], context.numColors, WEIGHTS.color)
    // concatenar todas as características em um único vetor
    return tf.concat1d([price, age, category, color])

}


// Para cada usuário, calcular um vetor de características baseado nos produtos que ele comprou
// Fazendo a média dos vetores dos produtos comprados para obter um perfil de usuário representativo
function encodeUser(user, context) {
    if (user.purchases.length) {
        return tf.stack(
            user.purchases.map(
                product => encodeProduct(product, context).dataSync()
            )
        )
        .mean(0)
        .reshape([1, context.dimensions])
    }
    // Se o usuário não tiver compras, criar um vetor neutro 
    return tf.concat1d(
        [
            tf.zeros([1]), // preço é ignorado,
            tf.tensor1d([
                normalize(user.age, context.minAge, context.maxAge)
                * WEIGHTS.age
            ]),
            tf.zeros([context.numCategories]), // categoria ignorada,
            tf.zeros([context.numColors]), // color ignorada,

        ]
    ).reshape([1, context.dimentions])
}

// Criar dados de treinamento combinando os vetores de usuários e produtos, e gerando rótulos (1 para comprado, 0 para não comprado)
// Para cada usuário, criar um vetor de entrada que é a concatenação do vetor do usuário e do produto, e um rótulo que indica se o usuário comprou o produto ou não
function createTrainingData(context) {
    const inputs = []
    const labels = []
    context.users
    .filter(u => u.purchases.length) // filtrar usuários sem compras para evitar vetores de usuário vazios
    .forEach(user => {
        const userVector = encodeUser(user, context).dataSync()
        context.products.forEach(product => {
            const productVector = encodeProduct(product, context).dataSync()
            const label = user.purchases.some(p => p.name === product.name) ? 1 : 0
            
            inputs.push([...userVector, ...productVector])
            labels.push(label)
        })
    })

    return {
        xs: tf.tensor2d(inputs),
        ys: tf.tensor2d(labels, [labels.length, 1]),
        inpuDimension: context.dimensions * 2
        // tamanho do vetor de entrada é a concatenação do vetor do usuário e do produto
    }
}


async function configureNeuralNetAndTrain(trainingData) {
    const model = tf.sequential();
    // Camada de entrada: recebe o vetor concatenado do usuário e do produto
    model.add(tf.layers.dense({ inputShape: [trainingData.inpuDimension], units: 128, activation: 'relu' }));
    // Camada oculta: processa as interações entre as características do usuário e do produto
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    //quarta camada: reduz para um vetor menor, forçando o modelo a aprender representações mais compactas
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));

    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' })); // Camada de saída: probabilidade de compra (0 a 1)

    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });

    await model.fit(trainingData.xs, trainingData.ys, {
        epochs: 100,
        batchSize: 32,
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                postMessage({
                    type: workerEvents.trainingLog,
                    epoch: epoch, 
                    loss: logs.loss,
                    accuracy: logs.acc
                });
            }
        }
    })
    return model;
}

async function trainModel({ users }) {
    console.log('Training model with users:', users);
    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 1 } });
    const products = await (await fetch('/data/products.json')).json()

    const context = makeContext(products, users)
    context.productVectors = products.map(product => {
        return {
            name: product.name,
            meta: {...product},
            vector: encodeProduct(product, context).dataSync()
        }
    })

    _globalCtx = context
    const trainingData = createTrainingData(context)
    _model = await configureNeuralNetAndTrain(trainingData)

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
    postMessage({ type: workerEvents.trainingComplete });
}
function recommend({ user }) {
    if (!_model) return;
    const context = _globalCtx
    //Converta o usuário para um vetor usando a mesma função de codificação usada durante o treinamento
    const userVector = encodeUser(user, context).dataSync()
    
    // Para cada produto, crie um vetor de entrada concatenando o vetor do usuário e o vetor do produto, e use o modelo para prever a probabilidade de compra
    // Em aplicações reais:
    //  Armazene todos os vetores de produtos em um banco de dados vetorial (como Postgres, Neo4j ou Pinecone)
    //  Consulta: Encontre os 200 produtos mais próximos do vetor do usuário
    //  Execute _model.predict() apenas nesses produtos

    // 2️⃣ Crie pares de entrada: para cada produto, concatene o vetor do usuário
    //    com o vetor codificado do produto.
    //    Por quê? O modelo prevê o "score de compatibilidade" para cada par (usuário, produto).
    const inputs = context.productVectors.map(({ vector }) => {
        return [...userVector, ...vector]
    })

    // Converta a matriz de entrada para um tensor 2D, onde cada linha é um par (usuário, produto)
    const inputTensor = tf.tensor2d(inputs)

    // Rode o modelo para obter as previsões de compra para cada produto.
    // O resultado é um vetor de probabilidades, onde cada valor indica a probabilidade de o usuário comprar aquele produto.
    const predictions = _model.predict(inputTensor)
    
    // Extraia os scores e combine com os produtos para criar uma lista de recomendações
    const scores = predictions.dataSync()
    const recommendations = context.productVectors.map((product, index) => {
        return {
            ...product.meta,
            name: product.name,
            score: scores[index]
        }
    })
    
    const sortedItems = recommendations.sort((a, b) => b.score - a.score)    

    postMessage({
        type: workerEvents.recommend,
        user,
        recommendations: sortedItems
    });

}
const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: recommend,
};

self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
