// ===== 状态 =====
let selectedMode = 'random';
let selectedCount = 30;
let selectedTime = 120;
let currentQuestions = [];
let userAnswers = {};
let timerInterval = null;
let timeRemaining = 0;
let examStartTime = null;

// ===== 按钮组事件 =====
document.querySelectorAll('#modeGroup .btn').forEach(b => {
  b.onclick = () => {
    document.querySelectorAll('#modeGroup .btn').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    selectedMode = b.dataset.val;
    document.getElementById('countGroup').style.display = selectedMode === 'random' ? 'block' : 'none';
  };
});
document.querySelectorAll('#countGroup2 .btn').forEach(b => {
  b.onclick = () => {
    document.querySelectorAll('#countGroup2 .btn').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    selectedCount = parseInt(b.dataset.val);
  };
});
document.querySelectorAll('#timeGroup .btn').forEach(b => {
  b.onclick = () => {
    document.querySelectorAll('#timeGroup .btn').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    selectedTime = parseInt(b.dataset.val);
  };
});

// ===== 随机抽题 =====
function shuffleArray(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function selectQuestions() {
  if (selectedMode === 'all') return shuffleArray(questionBank);
  // 按分类均匀抽样
  const byCat = {};
  categories.forEach(c => { byCat[c] = questionBank.filter(q => q.cat === c); });
  const result = [];
  const perCat = Math.floor(selectedCount / categories.length);
  let rem = selectedCount - perCat * categories.length;
  categories.forEach(c => {
    const pool = shuffleArray(byCat[c]);
    const take = perCat + (rem > 0 ? 1 : 0);
    if (rem > 0) rem--;
    result.push(...pool.slice(0, Math.min(take, pool.length)));
  });
  // 补充不足
  while (result.length < selectedCount) {
    const remaining = questionBank.filter(q => !result.find(r => r.id === q.id));
    if (remaining.length === 0) break;
    result.push(remaining[Math.floor(Math.random() * remaining.length)]);
  }
  return shuffleArray(result.slice(0, selectedCount));
}

// ===== 计时器 =====
function startTimer() {
  updateTimerDisplay();
  timerInterval = setInterval(() => {
    timeRemaining--;
    updateTimerDisplay();
    if (timeRemaining <= 0) {
      clearInterval(timerInterval);
      submitExam();
    }
  }, 1000);
}

function updateTimerDisplay() {
  const m = Math.floor(timeRemaining / 60);
  const s = timeRemaining % 60;
  const el = document.getElementById('timerDisplay');
  el.textContent = `${m}:${s.toString().padStart(2, '0')}`;
  if (timeRemaining < 300) el.classList.add('warning');
}

// ===== 更新进度 =====
function updateProgress() {
  const answered = Object.keys(userAnswers).length;
  document.getElementById('qAnswered').textContent = answered;
  const pct = (answered / currentQuestions.length) * 100;
  document.getElementById('progressFill').style.width = pct + '%';
}

// ===== 渲染题目 =====
function renderQuestions() {
  const container = document.getElementById('questionsContainer');
  container.innerHTML = '';
  let curCat = '';
  currentQuestions.forEach((q, idx) => {
    if (q.cat !== curCat) {
      curCat = q.cat;
      const cnt = currentQuestions.filter(x => x.cat === q.cat).length;
      container.innerHTML += `<div class="section-title"><span>${curCat}</span><span class="score">${cnt} 题</span></div>`;
    }
    const div = document.createElement('div');
    div.className = 'question';
    div.id = `q-${q.id}`;
    const typeNames = { choice: '选择题', fill: '填空题', truefalse: '判断题', short: '简答题' };
    let inputHTML = '';
    if (q.type === 'choice') {
      inputHTML = '<div class="choices">';
      q.opts.forEach(opt => {
        const letter = opt.charAt(0);
        inputHTML += `<div class="choice-opt" data-qid="${q.id}" data-val="${letter}"><input type="radio" name="q${q.id}" value="${letter}"><span>${opt}</span></div>`;
      });
      inputHTML += '</div>';
    } else if (q.type === 'fill') {
      inputHTML = `<input type="text" class="fill-input" data-qid="${q.id}" placeholder="请输入答案...">`;
    } else if (q.type === 'truefalse') {
      inputHTML = `<div class="tf-group"><button class="tf-btn" data-qid="${q.id}" data-val="true">正确 ✓</button><button class="tf-btn" data-qid="${q.id}" data-val="false">错误 ✗</button></div>`;
    } else {
      inputHTML = `<textarea class="short-area" data-qid="${q.id}" placeholder="请输入你的答案..."></textarea>`;
    }
    const correctAns = q.ans === 'true' ? '正确' : q.ans === 'false' ? '错误' : q.ans;
    div.innerHTML = `<div class="q-head"><span>第 ${idx + 1} 题（${typeNames[q.type]}）</span><span class="q-cat">${q.cat}</span></div><div class="q-body"><p>${q.q}</p></div>${inputHTML}<div class="explanation" id="exp-${q.id}"><strong>参考答案：</strong><span class="ref-answer">${correctAns}</span><br>${q.exp}</div>`;
    container.appendChild(div);
  });
  // 绑定事件
  container.querySelectorAll('.choice-opt').forEach(el => {
    el.onclick = () => {
      const qid = parseInt(el.dataset.qid);
      const val = el.dataset.val;
      userAnswers[qid] = val;
      el.parentElement.querySelectorAll('.choice-opt').forEach(o => o.classList.remove('selected'));
      el.classList.add('selected');
      el.querySelector('input').checked = true;
      updateProgress();
    };
  });
  container.querySelectorAll('.fill-input').forEach(el => {
    el.oninput = () => {
      userAnswers[parseInt(el.dataset.qid)] = el.value.trim();
      updateProgress();
    };
  });
  container.querySelectorAll('.tf-btn').forEach(el => {
    el.onclick = () => {
      const qid = parseInt(el.dataset.qid);
      userAnswers[qid] = el.dataset.val;
      el.parentElement.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('selected'));
      el.classList.add('selected');
      updateProgress();
    };
  });
  container.querySelectorAll('.short-area').forEach(el => {
    el.oninput = () => {
      userAnswers[parseInt(el.dataset.qid)] = el.value;
      updateProgress();
    };
  });
  if (window.MathJax && window.MathJax.typeset) MathJax.typeset();
}

// ===== 开始考试 =====
function startExam() {
  currentQuestions = selectQuestions();
  userAnswers = {};
  timeRemaining = selectedTime * 60;
  examStartTime = Date.now();
  document.getElementById('startScreen').classList.add('hidden');
  document.getElementById('examScreen').classList.remove('hidden');
  document.getElementById('qCount').textContent = currentQuestions.length;
  renderQuestions();
  startTimer();
  updateProgress();
}

// ===== 提交试卷 =====
function submitExam() {
  clearInterval(timerInterval);
  let correct = 0, graded = 0;
  const gradable = ['choice', 'truefalse', 'fill'];
  currentQuestions.forEach(q => {
    if (!gradable.includes(q.type)) return;
    graded++;
    const userAns = userAnswers[q.id];
    const isCorrect = (q.type === 'choice' && userAns === q.ans) ||
      (q.type === 'truefalse' && userAns === q.ans) ||
      (q.type === 'fill' && userAns && userAns.toLowerCase() === q.ans.toLowerCase());
    if (isCorrect) correct++;
  });
  const score = graded > 0 ? Math.round((correct / graded) * 100) : 0;
  const wrong = graded - correct;
  const elapsed = Math.floor((Date.now() - examStartTime) / 1000);
  const em = Math.floor(elapsed / 60), es = elapsed % 60;
  const circle = document.getElementById('scoreCircle');
  document.getElementById('scoreText').textContent = score;
  circle.className = 'score-circle ' + (score >= 90 ? 'excellent' : score >= 70 ? 'good' : score >= 60 ? 'pass' : 'fail');
  document.getElementById('correctCount').textContent = correct;
  document.getElementById('wrongCount').textContent = wrong;
  document.getElementById('timeUsed').textContent = `${em}:${es.toString().padStart(2, '0')}`;
  document.getElementById('resultModal').classList.add('show');
}

// ===== 查看解析 =====
function reviewAnswers() {
  document.getElementById('resultModal').classList.remove('show');
  currentQuestions.forEach(q => {
    const expEl = document.getElementById(`exp-${q.id}`);
    if (!expEl) return;
    expEl.classList.add('show');
    const userAns = userAnswers[q.id];
    let isCorrect = false;
    if (q.type === 'choice') {
      isCorrect = userAns === q.ans;
      const qDiv = document.getElementById(`q-${q.id}`);
      qDiv.querySelectorAll('.choice-opt').forEach(el => {
        const val = el.dataset.val;
        if (val === q.ans) el.classList.add('correct');
        else if (val === userAns && !isCorrect) el.classList.add('wrong');
      });
    } else if (q.type === 'truefalse') {
      isCorrect = userAns === q.ans;
      document.getElementById(`q-${q.id}`).querySelectorAll('.tf-btn').forEach(b => {
        if (b.dataset.val === q.ans) b.classList.add('correct');
        else if (b.dataset.val === userAns && !isCorrect) b.classList.add('wrong');
      });
    } else if (q.type === 'fill') {
      isCorrect = userAns && userAns.toLowerCase() === q.ans.toLowerCase();
    } else {
      isCorrect = null; // 简答题不评分
    }
    expEl.classList.add(isCorrect === true ? 'correct-exp' : isCorrect === false ? 'wrong-exp' : '');
    expEl.insertAdjacentHTML('afterbegin', isCorrect === true ? '<p style="color:var(--green);font-weight:700">✓ 回答正确</p>' : isCorrect === false ? `<p style="color:var(--red);font-weight:700">✗ 回答错误（你的答案：${userAns || '未作答'}）</p>` : '<p style="color:var(--orange);font-weight:700">简答题请对照参考答案</p>');
  });
  window.scrollTo({ top: 0, behavior: 'smooth' });
}