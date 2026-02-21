import tf from '@tensorflow/tfjs-node';
const w = {
    status: 10,
    nome: 15,
    prev: 20,
    real: 18,
    detalhes: 40
};

const categories = [
    {name: "premium", normalization: [1, 0, 0]},
    {name: "standard", normalization: [0, 1, 0]},
    {name: "blocked", normalization: [0, 0, 1]}
    ]; // Ordem dos labels
const categoryMap = Object.fromEntries(categories.map(c => [c.name, c.normalization]));

function gerarMassaDeDados(quantidade) {
    const dados = [];
    const nomes = ["Alpha", "Beta", "Delta", "Gamma", "Sigma", "Omega", "Zeta"];

    for (let i = 0; i < quantidade; i++) {
        // Alterna entre os perfis para garantir equilíbrio (33% cada)
        const perfil = i % 3; 
        let item = {
            name: `${nomes[i % nomes.length]}-${i}`,
            income: 0,
            score: 0,
            idade: Math.floor(Math.random() * (70 - 18) + 18),
            debt: 0,
            current_job_time: 0,
            monthly_spends: 0,
            category: ""
        };

        if (perfil === 0) {
            // --- PERFIL PREMIUM (Rico e Estável) ---
            item.income = Math.floor(Math.random() * (20000 - 12000) + 12000);
            item.score = Math.floor(Math.random() * (1000 - 800) + 800);
            item.debt = 0; 
            item.current_job_time = Math.floor(Math.random() * (120 - 60) + 60);
            item.monthly_spends = Math.floor(item.income * 0.2); // Gasta pouco da renda
            item.category = "premium";
        } 
        else if (perfil === 1) {
            // --- PERFIL STANDARD (Classe Média) ---
            item.income = Math.floor(Math.random() * (10000 - 5000) + 5000);
            item.score = Math.floor(Math.random() * (799 - 500) + 500);
            item.debt = Math.random() > 0.8 ? 1 : 0; 
            item.current_job_time = Math.floor(Math.random() * (60 - 24) + 24);
            item.monthly_spends = Math.floor(item.income * 0.5);
            item.category = "standard";
        } 
        else {
            // --- PERFIL BLOCKED (Risco Alto) ---
            item.income = Math.floor(Math.random() * (4000 - 1500) + 1500);
            item.score = Math.floor(Math.random() * 450);
            item.debt = Math.random() > 0.4 ? 1 : 0; 
            item.current_job_time = Math.floor(Math.random() * 24);
            item.monthly_spends = Math.floor(item.income * 0.85); // Quase toda a renda comprometida
            item.category = "blocked";
        }

        dados.push(item);
    }

    // O sort aleatório é CRUCIAL para a rede não viciar na ordem dos perfis
    return dados.sort(() => Math.random() - 0.5);
}

async function trainModel(inputXs, outputYs) {
    const model = tf.sequential()

    model.add(tf.layers.dense({ inputShape: [6], units: 100, activation: "relu" }))
    model.add(tf.layers.dense({ units: 3, activation: "softmax" }))
    model.compile({
        loss: "categoricalCrossentropy",
        optimizer: tf.train.adam(0.01),
        metrics: ["accuracy"]
    })
    await model.fit(
        inputXs,
        outputYs,
        {
            verbose: 0,
            epochs: 100,
            shuffle: true,
            callbacks: {
                // onEpochEnd: (epoch, log) => console.log(
                //     `Epoch: ${epoch}: loss = ${log.loss}`
                // )
            }
        }
    )
    return model
}

function normalizeAttributes(person) {
    person.current_job_time = 120 ? person.current_job_time > 120 : person.current_job_time
    return[
        (person.income - 1500)/18500,
        person.score / 1000,
        (person.idade-18) / 62,
        person.debt,
        person.income/person.monthly_spends,
        person.current_job_time
        ]
}

function normalizeTrainData(person) {
    const personNormalized = normalizeAttributes(person)
    let category_norm= categoryMap[person.category] || [0, 0, 1]
    return { input: personNormalized, output: category_norm }
}

async function predict(model, person) {
    const normalizedPerson = normalizeAttributes(person)
    const tfInput = tf.tensor2d([normalizedPerson])

    // Faz a predição (output será um vetor de 3 probabilidades)
    const pred = model.predict(tfInput)
    const predArray = await pred.array()
    return predArray[0].map((prob, index) => ({ prob, index }))
}

function format_predictions(predictions) {
    const orded_predictions = predictions.sort((a, b) => b.prob - a.prob)
    const results = orded_predictions
    .map(p => `${categories[p.index].name} (${(p.prob * 100).toFixed(2)}%)`)
    .join(' - ')
    return {choosed_category: categories[orded_predictions[0].index].name, results}
}

const trainingData = [
    // Premium: Renda alta, Score alto, Dívida baixa
    { name: "Ana", income: 15000, score: 900, idade: 35, debt: 0, monthly_spends: 5000, current_job_time: 80, category: "premium" },
    { name: "Roberto", income: 12000, score: 850, idade: 45, debt: 0, monthly_spends: 4000, current_job_time: 120, category: "premium" },
    { name: "Helena", income: 18000, score: 950, idade: 38, debt: 0, monthly_spends: 6000, current_job_time: 90, category: "premium" },
    { name: "Marcos", income: 10000, score: 780, idade: 32, debt: 0, monthly_spends: 3000, current_job_time: 48, category: "premium" },
    { name: "Julia", income: 20000, score: 990, idade: 50, debt: 0, monthly_spends: 8000, current_job_time: 150, category: "premium" },

    // Standard: Classe média, Score ok, Dívida controlada
    { name: "Carlos", income: 5000, score: 600, idade: 28, debt: 0, monthly_spends: 3500, current_job_time: 60, category: "standard" },
    { name: "Lúcia", income: 4500, score: 550, idade: 30, debt: 1, monthly_spends: 3000, current_job_time: 24, category: "standard" },
    { name: "Ricardo", income: 7000, score: 650, idade: 40, debt: 0, monthly_spends: 5000, current_job_time: 72, category: "standard" },
    { name: "Beatriz", income: 6000, score: 700, idade: 34, debt: 0, monthly_spends: 4000, current_job_time: 36, category: "standard" },
    { name: "Fernando", income: 5500, score: 580, idade: 27, debt: 0, monthly_spends: 3200, current_job_time: 18, category: "standard" },

    // Blocked: Renda baixa OU Score muito baixo OU Gastos maiores que renda
    { name: "Bruno", income: 2000, score: 300, idade: 22, debt: 0, monthly_spends: 1900, current_job_time: 12, category: "blocked" },
    { name: "Maria", income: 5000, score: 100, idade: 25, debt: 1, monthly_spends: 5000, current_job_time: 40, category: "blocked" },
    { name: "Igor", income: 1800, score: 200, idade: 20, debt: 1, monthly_spends: 1800, current_job_time: 6, category: "blocked" },
    { name: "Sonia", income: 3000, score: 400, idade: 60, debt: 1, monthly_spends: 3100, current_job_time: 24, category: "blocked" },
    { name: "Tiago", income: 2500, score: 350, idade: 24, debt: 0, monthly_spends: 2200, current_job_time: 8, category: "blocked" },

    // Casos Mistos (Ricos com problemas ou Pobres esforçados)
    { name: "Wagner", income: 15000, score: 150, idade: 42, debt: 1, monthly_spends: 14000, current_job_time: 12, category: "blocked" }, // Rico endividado
    { name: "Daniela", income: 8000, score: 200, idade: 31, debt: 1, monthly_spends: 7500, current_job_time: 24, category: "blocked" }, // Renda boa, score péssimo
    { name: "Paulo", income: 8500, score: 800, idade: 37, debt: 0, monthly_spends: 3000, current_job_time: 96, category: "premium" }, // Estável
    { name: "Carla", income: 4200, score: 850, idade: 29, debt: 0, monthly_spends: 2000, current_job_time: 48, category: "standard" }, // Score salva renda
    { name: "Renato", income: 11000, score: 500, idade: 39, debt: 1, monthly_spends: 9000, current_job_time: 12, category: "standard" } // Renda alta mas instável
];
    
const trainingData2 = gerarMassaDeDados(500).concat(trainingData);
const normalizedData = trainingData.map(normalizeTrainData)

const trainInputs = normalizedData.map(item => item.input);
const trainOutputs = normalizedData.map(item => item.output);


const inputXs = tf.tensor2d(trainInputs)
const outputYs = tf.tensor2d(trainOutputs)

const model = await trainModel(inputXs, outputYs) 


const validationData = [
    { name: "Zé", income: 4500, score: 550, idade: 30, debt: 0, monthly_spends: 3000, current_job_time: 24, category: "standard" },
    { name: "Gustavo", income: 19000, score: 920, idade: 40, debt: 0, monthly_spends: 7000, current_job_time: 100, category: "premium" },
    { name: "Aline", income: 2100, score: 150, idade: 19, debt: 1, monthly_spends: 2050, current_job_time: 3, category: "blocked" },
    { name: "Claudia", income: 13000, score: 300, idade: 45, debt: 1, monthly_spends: 12000, current_job_time: 12, category: "blocked" },
    { name: "Sandra", income: 7500, score: 720, idade: 33, debt: 0, monthly_spends: 4000, current_job_time: 60, category: "standard" }
];
const validationData2 = gerarMassaDeDados(1000).concat(validationData);

console.log(
    "STATUS".padEnd(w.status) + "  | " +    
    "NOME".padEnd(w.nome) + " | " +
    "CATEGORIA PREVISTA".padEnd(w.prev) + " | " +
    "CATEGORIA REAL".padEnd(w.real) + " | " +
    "DETALHES DA PREDIÇÃO"
);
console.log("-".repeat(w.nome + w.prev + w.real + w.detalhes + 9));

let corrects = 0
for (const person of validationData2) {
    const predictions = await predict(model, person)
    const formatted = format_predictions(predictions)
    let status = "❌ ERRO"

    if (formatted.choosed_category === person.category) {
        corrects ++
        status = "✅ OK"
    }
    console.log(
        status.padEnd(w.status) + " | " +
        person.name.padEnd(w.nome) + " | " +
        formatted.choosed_category.padEnd(w.prev) + " | " +
        person.category.padEnd(w.real) + " | " +
        formatted.results
    );
}
console.log(`Acurácia: ${(corrects / validationData2.length * 100).toFixed(2)}%`)

// const jacob = { name: 'Jacob', income: 12000, score: 700, idade: 35, debt: 0, monthly_spends: 8000, current_job_time: 100 } 
// const ze = { name: 'Zé', income: 10000, score: 500, idade: 20, debt: 1, monthly_spends: 3000, current_job_time: 50 }

// format_response(jacob, await predict(model, jacob))
// format_response(ze, await predict(model, ze))
