import tf from '@tensorflow/tfjs-node';


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

function normalizeData(person) {
    person.current_job_time = 120 ? person.current_job_time > 120 : person.current_job_time
    return [
        (person.income - 1500)/18500,
        person.score / 1000,
        (person.idade-18) / 62,
        person.debt,
        person.income/person.monthly_spends,
        person.current_job_time
        ]
}

async function predict(model, person) {
    const normalized_person = normalizeData(person)
    const tfInput = tf.tensor2d([normalized_person])

    // Faz a predição (output será um vetor de 3 probabilidades)
    const pred = model.predict(tfInput)
    const predArray = await pred.array()
    return predArray[0].map((prob, index) => ({ prob, index }))
}

function format_response(person, predictions, labels) {
    const results = predictions
    .sort((a, b) => b.prob - a.prob)
    .map(p => `${labels[p.index]} (${(p.prob * 100).toFixed(2)}%)`)
    .join('\n')
    console.log(`Result for ${person.name}: ${results}`)
}

const person_array = [
    { name: "Ana", income: 15000, score: 900, idade: 35, debt: 0, monthly_spends: 10000, current_job_time: 80},
    { nome: "Carlos", income: 5000, score: 600, idade: 28, debt: 1, monthly_spends: 4000, current_job_time: 60},
    { nome: "Bruno", income: 2000, score: 300, idade: 22, debt: 0, monthly_spends: 2000, current_job_time: 12},
    { nome: "Maria", income: 5000, score: 100, idade: 25, debt: 1, monthly_spends: 5000, current_job_time: 40},
]
    
const normalized_person_array = person_array.map(normalizeData)

const categories = ["premium", "standard", "blocked"]; // Ordem dos labels
const tensorCategories = [
    [1, 0, 0], // premium - Ana
    [0, 1, 0], // standard - Carlos
    [0, 0, 1],  // blocked - Bruno
    [0, 0, 1]  // blocked - Maria
];

const inputXs = tf.tensor2d(normalized_person_array)
const outputYs = tf.tensor2d(tensorCategories)

const model = await trainModel(inputXs, outputYs)  


const jacob = { name: 'Jacob', income: 12000, score: 700, idade: 35, debt: 0, monthly_spends: 8000, current_job_time: 100 } 
const ze = { name: 'Zé', income: 10000, score: 500, idade: 20, debt: 1, monthly_spends: 3000, current_job_time: 50 }

format_response(jacob, await predict(model, jacob), categories)
format_response(ze, await predict(model, ze), categories)

