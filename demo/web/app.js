const state = {
  data: null,
  activeIndex: 0,
};

const els = {
  buttons: document.getElementById("example-buttons"),
  question: document.getElementById("question"),
  answer: document.getElementById("answer"),
  meanScore: document.getElementById("mean-score"),
  sourceId: document.getElementById("source-id"),
  split: document.getElementById("split"),
  contract: document.getElementById("contract"),
  scoreRange: document.getElementById("score-range"),
  excerpt: document.getElementById("excerpt"),
  disclaimer: document.getElementById("disclaimer-text"),
};

const tooltip = document.createElement("div");
tooltip.className = "tooltip";
document.body.appendChild(tooltip);

function clamp(value, low, high) {
  return Math.max(low, Math.min(high, value));
}

function formatScore(value) {
  return Number(value).toFixed(3);
}

function scoreColor(score) {
  const s = clamp(Number(score), 0, 1);
  const hue = 10 + s * 135;
  const sat = 72 - Math.abs(s - 0.5) * 18;
  const light = 78 - Math.abs(s - 0.5) * 14;
  return `hsl(${hue.toFixed(1)} ${sat.toFixed(1)}% ${light.toFixed(1)}%)`;
}

function textColor(score) {
  const s = clamp(Number(score), 0, 1);
  return s > 0.82 || s < 0.18 ? "#15130f" : "#201b12";
}

function shortQuestion(text) {
  return text.length > 82 ? `${text.slice(0, 79)}…` : text;
}

function renderButtons() {
  els.buttons.innerHTML = "";
  state.data.examples.forEach((example, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `example-button${index === state.activeIndex ? " active" : ""}`;
    button.innerHTML = `
      <span class="qid">${example.maud_source_id} · mean ${formatScore(example.score_summary.mean)}</span>
      <span class="qtext">${shortQuestion(example.question)}</span>
    `;
    button.addEventListener("click", () => {
      state.activeIndex = index;
      render();
    });
    els.buttons.appendChild(button);
  });
}

function sentenceNode(sentence) {
  const span = document.createElement("span");
  span.className = "sentence";
  span.tabIndex = 0;
  span.textContent = sentence.text;
  span.style.background = scoreColor(sentence.calibrated_score);
  span.style.color = textColor(sentence.calibrated_score);
  span.dataset.tooltip = [
    `calibrated: ${formatScore(sentence.calibrated_score)}`,
    `raw: ${formatScore(sentence.raw_score)}`,
    `source: ${sentence.score_source.replaceAll("_", " ")}`,
  ].join("\n");
  span.addEventListener("pointermove", showTooltip);
  span.addEventListener("pointerenter", showTooltip);
  span.addEventListener("pointerleave", hideTooltip);
  span.addEventListener("focus", event => showTooltip(event, true));
  span.addEventListener("blur", hideTooltip);
  return span;
}

function showTooltip(event, fromFocus = false) {
  const rect = event.currentTarget.getBoundingClientRect();
  tooltip.textContent = event.currentTarget.dataset.tooltip;
  const x = fromFocus ? rect.left + rect.width / 2 : event.clientX;
  const y = fromFocus ? rect.top : event.clientY;
  tooltip.style.left = `${x}px`;
  tooltip.style.top = `${y - 12}px`;
  tooltip.classList.add("visible");
}

function hideTooltip() {
  tooltip.classList.remove("visible");
}

function renderAnswer(example) {
  els.answer.innerHTML = "";
  example.sentences.forEach((sentence, index) => {
    if (index > 0) els.answer.appendChild(document.createTextNode(" "));
    els.answer.appendChild(sentenceNode(sentence));
  });
}

function render() {
  const example = state.data.examples[state.activeIndex];
  renderButtons();
  els.question.textContent = example.question;
  els.meanScore.textContent = formatScore(example.score_summary.mean);
  els.sourceId.textContent = example.maud_source_id;
  els.split.textContent = example.split;
  els.contract.textContent = example.contract_id;
  els.scoreRange.textContent = `${formatScore(example.score_summary.min)} → ${formatScore(example.score_summary.max)}`;
  els.excerpt.textContent = `${example.excerpt_preview}…`;
  renderAnswer(example);
}

async function main() {
  try {
    const response = await fetch("./examples.json");
    if (!response.ok) throw new Error(`Failed to load examples.json: ${response.status}`);
    state.data = await response.json();
    if (state.data?.metadata?.disclaimer) {
      els.disclaimer.textContent = state.data.metadata.disclaimer;
    }
    render();
  } catch (error) {
    els.question.textContent = "Could not load examples.";
    els.answer.textContent = String(error);
  }
}

main();
