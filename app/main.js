const API_BASE = 'http://localhost:8000';

const els = {
  userIdInput: document.getElementById('userIdInput'),
  getRecsBtn: document.getElementById('getRecsBtn'),
  coldStartWarning: document.getElementById('coldStartWarning'),
  historyList: document.getElementById('historyList'),
  historyLoader: document.getElementById('historyLoader'),
  recsList: document.getElementById('recsList'),
  recsLoader: document.getElementById('recsLoader'),
  healthStatus: document.getElementById('healthStatus'),
  metricBaseline: document.getElementById('metricBaseline'),
  metricRetrieval: document.getElementById('metricRetrieval'),
  metricFull: document.getElementById('metricFull'),
};

async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (res.ok) {
      const data = await res.json();
      els.healthStatus.textContent = `API Status: ${data.status.toUpperCase()} (Users: ${data.num_users})`;
      els.healthStatus.style.color = '#86efac';
      els.healthStatus.style.background = 'rgba(34, 197, 94, 0.2)';
      loadMetrics();
    } else {
      setHealthError();
    }
  } catch (err) {
    setHealthError();
  }
}

function setHealthError() {
  els.healthStatus.textContent = 'API Status: OFFLINE';
  els.healthStatus.style.color = '#fca5a5';
  els.healthStatus.style.background = 'rgba(239, 68, 68, 0.2)';
}

async function loadMetrics() {
  try {
    const res = await fetch(`${API_BASE}/metrics`);
    if (res.ok) {
      const data = await res.json();
      const format = (val) => val ? val.toFixed(4) : '-';
      
      els.metricBaseline.textContent = `Recall@10: ${format(data.popularity_baseline['recall@10'])}`;
      els.metricRetrieval.textContent = `Recall@10: ${format(data.retrieval_only['recall@10'])}`;
      els.metricFull.textContent = `Recall@10: ${format(data.full_pipeline['recall@10'])}`;
    }
  } catch (err) {
    console.error("Failed to load metrics", err);
  }
}

function renderMovieCard(movie) {
  // movie: {item_id, title, genres, score/rating}
  const scoreLabel = movie.score !== undefined ? `Score: ${movie.score.toFixed(3)}` : `Rating: ${movie.rating || '-'}`;
  return `
    <div class="movie-card">
      <div class="movie-title">${movie.title}</div>
      <div class="movie-genres">${movie.genres.replace(/\|/g, ' ')}</div>
      <div class="movie-score">
        <span>ID: ${movie.item_id}</span>
        <span class="score-badge">${scoreLabel}</span>
      </div>
    </div>
  `;
}

async function getRecommendations() {
  const userId = els.userIdInput.value;
  if (!userId) return;

  // UI Reset
  els.coldStartWarning.classList.add('hidden');
  els.historyList.innerHTML = '';
  els.recsList.innerHTML = '';
  els.historyLoader.classList.remove('hidden');
  els.recsLoader.classList.remove('hidden');
  els.getRecsBtn.disabled = true;

  try {
    // Fetch History
    const histRes = await fetch(`${API_BASE}/history/${userId}`);
    if (histRes.ok) {
      const histData = await histRes.json();
      els.historyLoader.classList.add('hidden');
      if (histData.history && histData.history.length > 0) {
        els.historyList.innerHTML = histData.history.slice(0, 10).map(renderMovieCard).join('');
      } else {
        els.historyList.innerHTML = '<p style="color: var(--text-muted)">No history found for this user.</p>';
      }
    } else {
      const errorData = await histRes.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP Error ${histRes.status}`);
    }

    // Fetch Recommendations
    const recsRes = await fetch(`${API_BASE}/recommend/${userId}`);
    if (recsRes.ok) {
      const recsData = await recsRes.json();
      els.recsLoader.classList.add('hidden');
      
      if (recsData.is_cold_start) {
        els.coldStartWarning.classList.remove('hidden');
      }

      if (recsData.recommendations && recsData.recommendations.length > 0) {
        els.recsList.innerHTML = recsData.recommendations.map(renderMovieCard).join('');
      } else {
        els.recsList.innerHTML = '<p style="color: var(--text-muted)">No recommendations available.</p>';
      }
    } else {
      const errorData = await recsRes.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP Error ${recsRes.status}`);
    }
  } catch (err) {
    console.error("API Error", err);
    els.historyLoader.classList.add('hidden');
    els.recsLoader.classList.add('hidden');
    els.recsList.innerHTML = `<p style="color: #fca5a5">${err.message || 'Failed to fetch data from API. Is the server running?'}</p>`;
    els.historyList.innerHTML = `<p style="color: #fca5a5">Error loading data.</p>`;
  } finally {
    els.getRecsBtn.disabled = false;
  }
}

// Event Listeners
els.getRecsBtn.addEventListener('click', getRecommendations);
els.userIdInput.addEventListener('keypress', (e) => {
  if (e.key === 'Enter') getRecommendations();
});

// Initialization
checkHealth();
