// API URL
const API_URL = "http://127.0.0.1:8000"
const chat = document.getElementById("chat")
const promptEl = document.getElementById("prompt")
const sendBtn = document.getElementById("send")
const newChatBtn = document.getElementById("new-chat")
const exampleBtns = document.querySelectorAll(".example-btn")

const modeToggle = document.getElementById("mode-toggle")
const chatContainer = document.getElementById("chat-container")
const quizContainer = document.getElementById("quiz-container")
const examples = document.getElementById("examples")
const footer = document.querySelector(".footer")
const generateQuizBtn = document.getElementById("generate-quiz")
const quizQuestionsEl = document.getElementById("quiz-questions")
const quizFeedbackEl = document.getElementById("quiz-feedback")
const quizTypeSelect = document.getElementById("quiz-type")
const quizTopicInput = document.getElementById("quiz-topic")
const topicButtonsContainer = document.getElementById("topic-buttons")

let quizMode = false
let currentQuestions = []

const modeSelect = document.getElementById("mode-select")

// Handle mode selection
modeSelect.addEventListener("change", () => {
  const mode = modeSelect.value
  if (mode === "qa") {
    chatContainer.classList.remove("hidden")
    quizContainer.classList.add("hidden")
    examples.classList.remove("hidden")
    footer.classList.remove("hidden")
  } else {
    chatContainer.classList.add("hidden")
    quizContainer.classList.remove("hidden")
    examples.classList.add("hidden")
    footer.classList.add("hidden")
  }
})

const quizCountInput = document.getElementById("quiz-count")
const generationTypeRadios = document.querySelectorAll(
  'input[name="generation-type"]'
)
const topicSection = document.getElementById("topic-section")

// Fetch and populate hardcoded topics as buttons
async function loadTopics() {
  try {
    console.log("DEBUG: Starting to load topics from:", `${API_URL}/quiz/topics`)
    const resp = await fetch(`${API_URL}/quiz/topics`)
    const data = await resp.json()
    console.log("DEBUG: Topics API response:", data)

    // Clear existing buttons
    topicButtonsContainer.innerHTML = ""

    // Create and add the label
    const label = document.createElement("div")
    label.className = "topic-buttons-label"
    label.textContent = "Quick select:"
    topicButtonsContainer.appendChild(label)

    // Create grid container for buttons
    const buttonGrid = document.createElement("div")
    buttonGrid.className = "topic-btn-grid"

    // Create buttons for each topic
    if (data.topics && data.topics.length > 0) {
      console.log("DEBUG: Creating buttons for topics:", data.topics)
      data.topics.forEach((topic, index) => {
        console.log(`DEBUG: Creating button ${index + 1} for topic:`, topic)
        const button = document.createElement("button")
        button.className = "topic-btn"
        button.textContent = topic
        button.type = "button"
        button.addEventListener("click", () => {
          console.log("DEBUG: Topic button clicked:", topic)
          quizTopicInput.value = topic
          quizTopicInput.focus()
        })
        buttonGrid.appendChild(button)
      })
      console.log("DEBUG: All topic buttons created and added")
    } else {
      console.log("DEBUG: No topics found in response, using fallback")
    }

    topicButtonsContainer.appendChild(buttonGrid)
    console.log("DEBUG: Topic buttons container updated in DOM")
  } catch (e) {
    console.error("DEBUG: Error loading topics:", e)
  }
}

// Load topics on page load
loadTopics()

// Handle generation type toggle
generationTypeRadios.forEach((radio) => {
  radio.addEventListener("change", (e) => {
    if (e.target.value === "topic") {
      topicSection.style.display = "flex"
    } else {
      topicSection.style.display = "none"
      quizTopicInput.value = "" // Clear topic when switching to random
    }
  })
})

async function generateQuiz() {
  console.log("DEBUG: Starting quiz generation")
  
  // Show loading state on button
  const btnText = generateQuizBtn.querySelector('.btn-text')
  const btnLoading = generateQuizBtn.querySelector('.btn-loading')
  btnText.classList.add('hidden')
  btnLoading.classList.remove('hidden')
  generateQuizBtn.disabled = true

  const type = quizTypeSelect.value
  const generationType = document.querySelector(
    'input[name="generation-type"]:checked'
  ).value
  const topic = generationType === "topic" ? quizTopicInput.value : null
  const count = parseInt(quizCountInput.value) || 3

  console.log("DEBUG: Quiz parameters:", { type, generationType, topic, count })

  // Validate topic input if topic-specific is chosen
  if (generationType === "topic" && (!topic || !topic.trim())) {
    console.log("DEBUG: Validation failed - empty topic")
    alert("Please enter a topic for topic-specific questions")
    loadingQuiz.classList.add("hidden")
    generateQuizBtn.disabled = false
    return
  }

  const payload = { topic: topic, question_type: type, count: count }
  console.log("DEBUG: Sending payload to server:", payload)

  try {
    const resp = await fetch(`${API_URL}/quiz/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
    const data = await resp.json()
    console.log("DEBUG: Quiz generation response:", data)
    currentQuestions = data.questions
    console.log("DEBUG: Current questions set:", currentQuestions)
    console.log("DEBUG: Number of questions received:", currentQuestions?.length || 0)
    renderQuizQuestions(currentQuestions)
  } catch (e) {
    console.error("DEBUG: Error generating quiz:", e)
    alert("Error generating quiz. Please try again.")
  } finally {
    // Reset button state
    const btnText = generateQuizBtn.querySelector('.btn-text')
    const btnLoading = generateQuizBtn.querySelector('.btn-loading')
    btnText.classList.remove('hidden')
    btnLoading.classList.add('hidden')
    generateQuizBtn.disabled = false
    console.log("DEBUG: Quiz generation completed, UI reset")
  }
}

function renderQuizQuestions(questions) {
  console.log("DEBUG: Starting to render quiz questions")
  quizQuestionsEl.innerHTML = ""
  quizFeedbackEl.innerHTML = "" // Clear global feedback
  quizFeedbackEl.classList.add("hidden")
  
  console.log("DEBUG: Rendering questions:", questions)
  console.log("DEBUG: Number of questions to render:", questions?.length || 0)
  
  if (!questions || questions.length === 0) {
    console.log("DEBUG: No questions to render")
    return
  }
  
  questions.forEach((q, index) => {
    console.log(`DEBUG: Processing question ${index + 1}:`, q)
    console.log(`DEBUG: Question ${index + 1} details:`, {
      id: q.id,
      type: q.type,
      question: q.question,
      hasOptions: !!q.options,
      optionsCount: q.options?.length || 0,
      options: q.options
    })
    
    const qDiv = document.createElement("div")
    qDiv.className = "quiz-question"
    qDiv.setAttribute("data-qid", q.id)
    
    // Create question header
    const questionHeader = document.createElement("h4")
    questionHeader.textContent = `${index + 1}. ${q.question}`
    qDiv.appendChild(questionHeader)
    console.log(`DEBUG: Question header created for Q${index + 1}`)
    
    // Create options container
    if (q.options && q.options.length > 0) {
      console.log(`DEBUG: Creating options for question ${index + 1}, options:`, q.options)
      const optionsDiv = document.createElement("div")
      optionsDiv.className = "quiz-options"
      
      // Add each option as a button
      q.options.forEach((opt, optIndex) => {
        console.log(`DEBUG: Creating option ${optIndex} for question ${index + 1}:`, opt)
        const optionBtn = document.createElement("button")
        optionBtn.className = "quiz-option"
        optionBtn.setAttribute("data-answer", opt)
        optionBtn.setAttribute("data-qid", q.id)
        optionBtn.textContent = opt || `Option ${String.fromCharCode(65 + optIndex)}`
        console.log(`DEBUG: Option button text set to:`, optionBtn.textContent)
        optionsDiv.appendChild(optionBtn)
      })
      
      qDiv.appendChild(optionsDiv)
      console.log(`DEBUG: Options container added to question ${index + 1}`)
    } else {
      console.log(`DEBUG: No options for question ${index + 1}, type: ${q.type}`)
      // For open-ended questions
      const textarea = document.createElement("textarea")
      textarea.className = "quiz-textarea"
      textarea.setAttribute("data-qid", q.id)
      textarea.setAttribute("placeholder", "Your answer...")
      qDiv.appendChild(textarea)
      console.log(`DEBUG: Textarea added for open-ended question ${index + 1}`)
    }
    
    // Add submit button
    const submitBtn = document.createElement("button")
    submitBtn.className = "btn submit-answer"
    submitBtn.setAttribute("data-qid", q.id)
    submitBtn.innerHTML = '<span class="btn-text">Submit Answer</span><div class="btn-loading hidden" style="display: inline-flex; align-items: center; gap: 6px;"><div class="spinner" style="width: 12px; height: 12px; border-width: 2px;"></div><span>Checking...</span></div>'
    qDiv.appendChild(submitBtn)
    
    // Add feedback container
    const feedbackDiv = document.createElement("div")
    feedbackDiv.className = "quiz-feedback-inline hidden"
    feedbackDiv.id = `feedback-${q.id}`
    qDiv.appendChild(feedbackDiv)
    
    quizQuestionsEl.appendChild(qDiv)
    console.log(`DEBUG: Question ${index + 1} added to DOM`)
  })
  
  console.log("DEBUG: All questions rendered successfully")
}

async function submitAnswer(qid, answer, submitBtn = null) {
  console.log("DEBUG: Submitting answer for question:", qid)
  console.log("DEBUG: User answer:", answer)
  try {
    const resp = await fetch(`${API_URL}/quiz/check`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question_id: qid, user_answer: answer }),
    })
    const data = await resp.json()
    console.log("DEBUG: Answer check response:", data)
    renderQuizFeedbackInline(qid, data)
  } catch (e) {
    console.error("DEBUG: Error checking answer:", e)
  } finally {
    // Reset submit button state
    if (submitBtn) {
      const btnText = submitBtn.querySelector('.btn-text')
      const btnLoading = submitBtn.querySelector('.btn-loading')
      btnText.classList.remove('hidden')
      btnLoading.classList.add('hidden')
      submitBtn.disabled = false
    }
  }
}

function renderQuizFeedbackInline(qid, feedback) {
  const feedbackDiv = document.getElementById(`feedback-${qid}`)
  if (!feedbackDiv) return

  feedbackDiv.classList.remove("hidden", "correct", "incorrect")
  feedbackDiv.className = `quiz-feedback-inline ${
    feedback.is_correct ? "correct" : "incorrect"
  }`
  feedbackDiv.innerHTML = `
    <h4>Quiz Feedback</h4>
    <div class="feedback-content">
      <p><strong>Correct Answer:</strong> ${feedback.correct_answer}</p>
      <p><strong>Your Grade:</strong> ${feedback.user_grade} (${(
    feedback.confidence_score * 100
  ).toFixed(1)}% confidence)</p>
      <p><strong>Feedback:</strong> ${feedback.feedback}</p>
      <p><strong>Explanation:</strong> ${feedback.explanation}</p>

      ${
        feedback.citations && feedback.citations.length > 0
          ? `
        <div class="citations-section">
          <h5>Database Citations:</h5>
          ${feedback.citations
            .map(
              (c) =>
                `<div class="citation">
               <span class="citation-source">[${c.source || "unknown"}${
                  c.page ? " · p" + c.page : ""
                }]</span>
             </div>`
            )
            .join("")}
        </div>
      `
          : ""
      }

      ${
        feedback.web_citations && feedback.web_citations.length > 0
          ? `
        <div class="web-citations-section">
          <h5>Additional References:</h5>
          ${feedback.web_citations
            .map(
              (wc) =>
                `<div class="web-citation">
               <h6><a href="${wc.url}" target="_blank" rel="noopener noreferrer">${wc.title}</a></h6>
               <p>${wc.snippet}</p>
               <small class="citation-source">Source: ${wc.source}</small>
             </div>`
            )
            .join("")}
        </div>
      `
          : ""
      }
    </div>
  `
}

generateQuizBtn.addEventListener("click", generateQuiz)

quizQuestionsEl.addEventListener("click", (e) => {
  console.log("DEBUG: Quiz container clicked, target:", e.target)
  
  if (e.target.classList.contains("quiz-option")) {
    console.log("DEBUG: Quiz option clicked")
    // Select the option (highlight it) instead of auto-submitting
    const qid = e.target.getAttribute("data-qid")
    console.log("DEBUG: Question ID from clicked option:", qid)
    const questionDiv = document.querySelector(
      `.quiz-question[data-qid="${qid}"]`
    )

    // Remove selection from all options in this question
    questionDiv.querySelectorAll(".quiz-option").forEach((opt) => {
      opt.classList.remove("selected")
    })

    // Mark this option as selected
    e.target.classList.add("selected")
    console.log("DEBUG: Option selected and highlighted")
  } else if (e.target.classList.contains("submit-answer")) {
    console.log("DEBUG: Submit answer button clicked")
    e.preventDefault()
    const submitBtn = e.target
    const qid = submitBtn.getAttribute("data-qid")
    console.log("DEBUG: Question ID from submit button:", qid)
    
    // Show loading state on submit button
    const btnText = submitBtn.querySelector('.btn-text')
    const btnLoading = submitBtn.querySelector('.btn-loading')
    btnText.classList.add('hidden')
    btnLoading.classList.remove('hidden')
    submitBtn.disabled = true
    
    const questionDiv = document.querySelector(
      `.quiz-question[data-qid="${qid}"]`
    )

    // Check if it's MCQ or open-ended
    const selectedOption = questionDiv.querySelector(".quiz-option.selected")
    const textarea = questionDiv.querySelector(".quiz-textarea")
    
    console.log("DEBUG: Selected option:", selectedOption)
    console.log("DEBUG: Textarea found:", !!textarea)

    let answer = null
    if (selectedOption) {
      answer = selectedOption.getAttribute("data-answer")
      console.log("DEBUG: Answer from selected option:", answer)
    } else if (textarea) {
      answer = textarea.value.trim()
      console.log("DEBUG: Answer from textarea:", answer)
    }

    if (answer) {
      console.log("DEBUG: Submitting answer:", answer)
      submitAnswer(qid, answer, submitBtn)
    } else {
      console.log("DEBUG: No answer provided, showing alert")
      // Reset button state
      btnText.classList.remove('hidden')
      btnLoading.classList.add('hidden')
      submitBtn.disabled = false
      alert("Please select or enter an answer before submitting")
    }
  }
})

exampleBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    const question = btn.getAttribute("data-question")
    promptEl.value = question
    promptEl.focus()
    sendBtn.click() // Auto-submit
  })
})

function parseMarkdown(text) {
  // Escape HTML to prevent XSS
  const escapeHtml = (str) => {
    const div = document.createElement("div")
    div.textContent = str
    return div.innerHTML
  }

  // Split by ** to handle bold text
  let result = ""
  let inBold = false
  const parts = text.split("**")

  parts.forEach((part, index) => {
    if (index === 0) {
      result += escapeHtml(part)
    } else {
      if (inBold) {
        result += "</strong>" + escapeHtml(part)
      } else {
        result += "<strong>" + escapeHtml(part)
      }
      inBold = !inBold
    }
  })

  // Preserve line breaks
  result = result.replace(/\n/g, "<br>")

  return result
}

function appendBubble(role, text) {
  const wrap = document.createElement("div")
  wrap.className = `bubble ${role}`
  const avatar = document.createElement("div")
  avatar.className = "avatar"
  avatar.innerHTML =
    role === "user"
      ? `<svg width="20" height="20" viewBox="0 0 24 24" fill="none"><path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z" fill="currentColor"/></svg>`
      : `<svg width="20" height="20" viewBox="0 0 24 24" fill="none"><path d="M9 12c1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3 1.34 3 3 3zm6 0c1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3 1.34 3 3 3zm-6 6c1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3 1.34 3 3 3zm6 0c1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3 1.34 3 3 3z" fill="currentColor"/></svg>`
  const content = document.createElement("div")
  content.className = "content"

  // Parse markdown for assistant responses
  if (role === "assistant") {
    content.innerHTML = parseMarkdown(text)
  } else {
    content.textContent = text
  }

  wrap.appendChild(avatar)
  wrap.appendChild(content)
  chat.appendChild(wrap)
  chat.scrollTop = chat.scrollHeight
}

async function ask(question) {
  const body = { question, top_k: 4 }
  console.log("Sending Q&A request:", body)

  const res = await fetch(`${API_URL}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })

  console.log("Q&A response status:", res.status)

  if (!res.ok) {
    const txt = await res.text()
    console.error("Q&A error response:", txt)
    throw new Error(txt || `HTTP ${res.status}`)
  }

  const data = await res.json()
  console.log("Q&A parsed response:", data)
  return data
}

function renderAnswer(resp) {
  console.log("Q&A Response:", resp)
  const { snippet, results, citations } = resp

  if (!snippet) {
    console.error("No snippet in response:", resp)
    appendBubble("assistant", "Error: No answer received")
    return
  }

  appendBubble("assistant", snippet)
  const last = chat.lastElementChild.querySelector(".content")
  if (Array.isArray(citations) && citations.length) {
    const mapDiv = document.createElement("div")
    mapDiv.className = "sources"
    mapDiv.innerHTML = citations
      .map(
        (c) =>
          `<span class="badge" data-rank="${c.index}">[${c.index}] ${
            c.source || "unknown"
          }${c.page ? " · p" + c.page : ""}</span>`
      )
      .join(" ")
    last.appendChild(mapDiv)
  } else if (results && results.length) {
    // Fallback to results if citations missing
    const div = document.createElement("div")
    div.className = "sources"
    div.innerHTML = results
      .map(
        (r, i) =>
          `<span class="badge" data-rank="${i + 1}">[${i + 1}] ${
            r.source || "unknown"
          }${r.page ? " · p" + r.page : ""}</span>`
      )
      .join(" ")
    last.appendChild(div)
  }
}

sendBtn.addEventListener("click", async () => {
  const q = promptEl.value.trim()
  if (!q) return
  appendBubble("user", q)
  promptEl.value = ""

  const thinkingWrap = document.createElement("div")
  thinkingWrap.className = "bubble assistant"
  const thinkingAvatar = document.createElement("div")
  thinkingAvatar.className = "avatar"
  thinkingAvatar.innerHTML =
    '<svg width="20" height="20" viewBox="0 0 24 24" fill="none"><path d="M9 12c1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3 1.34 3 3 3zm6 0c1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3 1.34 3 3 3zm-6 6c1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3 1.34 3 3 3zm6 0c1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3 1.34 3 3 3z" fill="currentColor"/></svg>'
  const thinkingContent = document.createElement("div")
  thinkingContent.className = "content thinking"
  thinkingContent.innerHTML =
    '<div class="thinking-indicator"><div class="thinking-spinner"></div><span class="thinking-text">Thinking...</span></div>'
  thinkingWrap.appendChild(thinkingAvatar)
  thinkingWrap.appendChild(thinkingContent)
  chat.appendChild(thinkingWrap)
  chat.scrollTop = chat.scrollHeight

  try {
    const resp = await ask(q)
    chat.removeChild(thinkingWrap)
    renderAnswer(resp)
  } catch (e) {
    chat.removeChild(thinkingWrap)
    appendBubble("assistant", `Error: ${e.message}`)
  }
})

promptEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault()
    sendBtn.click()
  }
})

if (newChatBtn) {
  newChatBtn.addEventListener("click", () => {
    chat.innerHTML = ""
    const meta = document.getElementById("meta")
    if (meta) meta.textContent = ""
    promptEl.value = ""
    promptEl.focus()
  })
}
