const fs = require('fs')
const { Ollama } = require('ollama')

const ollama = new Ollama()

const EMBEDDING_MODEL = 'mxbai-embed-large'
const CHAT_MODEL = 'phi3'
const CSV_PATH = 'projects.csv'
const QUESTION = 'Can you give me escalation rate of Project #230 named Dolmades BMW?'

const getPrompt = (data, question) => {
  return `Using this data: ${data}. Respond to this prompt: ${question}`
}

main()

async function main () {
  const projects = readFromCSV(CSV_PATH)

  const questionEmbeddingResponse = await ollama.embeddings({
    model: EMBEDDING_MODEL, prompt: QUESTION
  })
  const questionEmbedding = questionEmbeddingResponse.embedding
  const projectEmbeddings = await Promise.all(projects.map(project =>
    ollama.embeddings({
      model: EMBEDDING_MODEL, prompt: JSON.stringify(project)
    }).then(response => response.embedding)
  ))

  const similarities = calculateSimilarities(
    questionEmbedding,
    projectEmbeddings,
    projects
  )
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, 5).map(sim => sim.project)
    .map(project => JSON.stringify(project, null, 2)).join('\n\n')

  const answer = await ollama.chat({
    model: CHAT_MODEL,
    messages: [{ role: 'user', content: getPrompt(similarities, QUESTION) }],
    stream: false
  })

  return console.log(answer.message.content)
}

function calculateSimilarities (questionEmbedding, projectEmbeddings, projects) {
  return projectEmbeddings.map((projectEmbedding, index) => ({
    similarity: cosineSimilarity(questionEmbedding, projectEmbedding),
    project: projects[index]
  }))
}

function cosineSimilarity (vecA, vecB) {
  const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0)
  const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0))
  const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0))
  return dotProduct / (magnitudeA * magnitudeB + Number.EPSILON)
}

function readFromCSV (path) {
  const csv = fs.readFileSync(path, 'utf8')
  const projectHeadersFromFirstLine = csv.split('\n')[0].split(',')
  return csv.split('\n').slice(1).map(line => {
    const project = {}
    line.split(',').forEach((value, index) => {
      project[projectHeadersFromFirstLine[index]] = value
    })
    return project
  })
}
