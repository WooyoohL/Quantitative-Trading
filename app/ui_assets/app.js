const appState = {
  overview: null,
  activeView: "overview",
  tableCache: new Map(),
  actions: [],
  jobsTimer: null,
};

const titles = {
  overview: ["总览", "读取 outputs 下的最新状态文件。"],
  candidates: ["候选池", "查看基础池、热度池、最终候选池和事件筛选输入。"],
  inference: ["推理结果", "查看当前模拟盘引用的模型推理产物。"],
  trading: ["模拟交易", "查看账户、交易流水、待买计划和权益曲线。"],
  config: ["配置与运行", "查看关键配置，并触发白名单运行命令。"],
  logs: ["日志", "查看最近输出日志。"],
};

function qs(selector, root = document) {
  return root.querySelector(selector);
}

function qsa(selector, root = document) {
  return Array.from(root.querySelectorAll(selector));
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await res.json();
  if (!res.ok) {
    throw new Error(payload.error || res.statusText);
  }
  return payload;
}

function fmt(value, digits = 2) {
  if (value === null || value === undefined || value === "") return "-";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return "-";
    return value.toLocaleString("zh-CN", { maximumFractionDigits: digits });
  }
  return String(value);
}

function money(value) {
  if (value === null || value === undefined || value === "") return "-";
  const num = Number(value);
  if (!Number.isFinite(num)) return fmt(value);
  return num.toLocaleString("zh-CN", { maximumFractionDigits: 2 });
}

function htmlEscape(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function setPageTitle(view) {
  const [title, subtitle] = titles[view] || titles.overview;
  qs("#page-title").textContent = title;
  qs("#page-subtitle").textContent = subtitle;
}

function metric(label, value, sub = "") {
  return `
    <div class="metric">
      <div class="metric-label">${htmlEscape(label)}</div>
      <div class="metric-value">${htmlEscape(value)}</div>
      <div class="metric-sub">${htmlEscape(sub)}</div>
    </div>
  `;
}

function statusBadge(value) {
  const text = String(value ?? "-");
  const lower = text.toLowerCase();
  let cls = "status";
  if (["true", "succeeded", "ok", "计划买入", "保留"].some((word) => lower.includes(word.toLowerCase()))) {
    cls += " ok";
  } else if (["failed", "error", "排除", "高"].some((word) => lower.includes(word.toLowerCase()))) {
    cls += " danger";
  } else if (["running", "queued", "观察", "中"].some((word) => lower.includes(word.toLowerCase()))) {
    cls += " warn";
  }
  return `<span class="${cls}">${htmlEscape(text)}</span>`;
}

function renderTablePayload(payload, options = {}) {
  const rows = payload.rows || [];
  const columns = payload.columns || [];
  const title = options.title || payload.title || payload.name;
  const subtitle = payload.source?.path
    ? `${payload.count ?? rows.length} / ${payload.total_count ?? rows.length} 行，${payload.source.path}`
    : `${payload.count ?? rows.length} / ${payload.total_count ?? rows.length} 行`;
  if (!columns.length) {
    const emptyMarkup = `
      <div class="toolbar"><div><h2>${htmlEscape(title)}</h2><p>${htmlEscape(subtitle)}</p></div></div>
      <div class="empty">暂无数据。</div>
    `;
    return options.embedded ? emptyMarkup : `<section class="panel">${emptyMarkup}</section>`;
  }
  const head = columns.map((column) => `<th>${htmlEscape(column)}</th>`).join("");
  const body = rows
    .map((row) => {
      const cells = columns
        .map((column) => {
          const value = row[column];
          const rendered = /(状态|动作|结论|risk|status|action)/i.test(column)
            ? statusBadge(value)
            : htmlEscape(fmt(value, 6));
          return `<td>${rendered}</td>`;
        })
        .join("");
      return `<tr>${cells}</tr>`;
    })
    .join("");
  const tableMarkup = `
    <div class="toolbar">
      <div>
        <h2>${htmlEscape(title)}</h2>
        <p>${htmlEscape(subtitle)}</p>
      </div>
    </div>
    <div class="table-wrap">
      <table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>
    </div>
  `;
  return options.embedded ? tableMarkup : `<section class="panel">${tableMarkup}</section>`;
}

async function loadTable(name, limit = 5000) {
  const key = `${name}:${limit}`;
  if (appState.tableCache.has(key)) return appState.tableCache.get(key);
  const payload = await api(`/api/table?name=${encodeURIComponent(name)}&limit=${limit}`);
  appState.tableCache.set(key, payload);
  return payload;
}

function renderEquityChart(rows) {
  if (!rows || !rows.length) {
    return `<div class="empty">暂无权益曲线。</div>`;
  }
  const width = 760;
  const height = 220;
  const pad = 28;
  const values = rows.map((row) => Number(row.total_equity)).filter((value) => Number.isFinite(value));
  if (!values.length) return `<div class="empty">暂无权益数值。</div>`;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const points = rows
    .map((row, index) => {
      const value = Number(row.total_equity);
      const x = pad + (index * (width - pad * 2)) / Math.max(rows.length - 1, 1);
      const y = height - pad - ((value - min) / span) * (height - pad * 2);
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
  const labels = rows
    .map((row, index) => {
      const value = Number(row.total_equity);
      const x = pad + (index * (width - pad * 2)) / Math.max(rows.length - 1, 1);
      const y = height - pad - ((value - min) / span) * (height - pad * 2);
      return `<circle cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="3"><title>${htmlEscape(
        `${row.date}: ${money(value)}`
      )}</title></circle>`;
    })
    .join("");
  return `
    <svg class="chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="权益曲线">
      <rect x="0" y="0" width="${width}" height="${height}" fill="#fff"></rect>
      <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" stroke="#d9dee6"></line>
      <line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" stroke="#d9dee6"></line>
      <polyline points="${points}" fill="none" stroke="#176b87" stroke-width="3"></polyline>
      <g fill="#1f8a70">${labels}</g>
      <text x="${pad}" y="18" fill="#667085" font-size="12">${money(max)}</text>
      <text x="${pad}" y="${height - 8}" fill="#667085" font-size="12">${money(min)}</text>
    </svg>
  `;
}

function keyValueGrid(object) {
  const entries = Object.entries(object || {});
  if (!entries.length) return `<div class="empty">暂无数据。</div>`;
  return `
    <div class="kv">
      ${entries
        .map(
          ([key, value]) => `
            <div class="kv-key">${htmlEscape(key)}</div>
            <div class="kv-value">${htmlEscape(typeof value === "object" ? JSON.stringify(value) : fmt(value, 6))}</div>
          `
        )
        .join("")}
    </div>
  `;
}

async function renderOverview() {
  const data = appState.overview;
  const k = data.kpis || {};
  const finalPlan = data.final_plan;
  const inferenceRows = data.inference_summaries || [];
  const equityRows = data.state?.equity_curve || [];
  qs("#app").innerHTML = `
    <section class="grid cols-4">
      ${metric("总权益", money(k.total_equity), `现金 ${money(k.cash)}`)}
      ${metric("待买计划", fmt(k.pending_buy_count, 0), `持仓 ${fmt(k.position_count, 0)}`)}
      ${metric("交易记录", fmt(k.trade_count, 0), `定稿 ${fmt(k.last_finalize_date)}`)}
      ${metric("候选池", fmt(data.candidate_universe?.candidate_universe_count, 0), `热度 ${fmt(data.candidate_universe?.heat_candidate_count, 0)}`)}
    </section>
    <section class="grid cols-2">
      <section class="panel">
        <h2>权益曲线</h2>
        ${renderEquityChart(equityRows)}
      </section>
      <section class="panel">
        ${renderTablePayload({
          title: "当前推理摘要",
          columns: Object.keys(inferenceRows[0] || {}),
          rows: inferenceRows,
          count: inferenceRows.length,
          total_count: inferenceRows.length,
          source: { path: "outputs/inference_runs" },
        }, { embedded: true })}
      </section>
    </section>
    ${renderTablePayload(finalPlan)}
  `;
}

async function renderCandidates() {
  const data = appState.overview;
  const universe = data.candidate_universe || {};
  const candidateTable = await loadTable("candidate_universe");
  const filterInput = await loadTable("filter_input");
  qs("#app").innerHTML = `
    <section class="grid cols-4">
      ${metric("基础池", fmt(universe.base_candidate_count, 0), universe.base_candidate_path || "")}
      ${metric("热度新增", fmt(universe.heat_candidate_count, 0), `限制后 ${fmt(universe.heat_candidate_after_limits_count, 0)}`)}
      ${metric("限制前", fmt(universe.candidate_universe_before_limits_count, 0), "基础池 + 热度池")}
      ${metric("最终候选", fmt(universe.candidate_universe_count, 0), `剔除 ${fmt(universe.candidate_universe_removed_by_limits_count, 0)}`)}
    </section>
    ${renderTablePayload(candidateTable)}
    ${renderTablePayload(filterInput)}
  `;
}

async function renderInference() {
  const overview = appState.overview;
  const tables = overview.tables || [];
  const reviewTables = tables.filter((item) => item.name.endsWith("_review"));
  const rankTables = tables.filter((item) => item.name.endsWith("_rank"));
  const reviewPayloads = await Promise.all(reviewTables.map((item) => loadTable(item.name)));
  const rankPayloads = await Promise.all(rankTables.map((item) => loadTable(item.name, 200)));
  const recent = overview.recent_inference_runs || [];
  qs("#app").innerHTML = `
    ${renderTablePayload({
      title: "最近推理目录",
      columns: Object.keys(recent[0] || {}),
      rows: recent,
      count: recent.length,
      total_count: recent.length,
      source: { path: "outputs/inference_runs" },
    })}
    ${reviewPayloads.map((payload) => renderTablePayload(payload)).join("")}
    ${rankPayloads.map((payload) => renderTablePayload(payload, { title: `${payload.title}（前 200）` })).join("")}
  `;
}

async function renderTrading() {
  const pending = await loadTable("pending_buys");
  const positions = await loadTable("positions");
  const equity = await loadTable("equity_curve");
  const account = await loadTable("account_ledger");
  const trades = await loadTable("trade_ledger");
  const tracking = await loadTable("buy_tracking");
  qs("#app").innerHTML = `
    <section class="grid cols-2">
      ${renderTablePayload(pending)}
      ${renderTablePayload(positions)}
    </section>
    ${renderTablePayload(equity)}
    ${renderTablePayload(account)}
    ${renderTablePayload(trades)}
    ${renderTablePayload(tracking)}
  `;
}

async function renderConfig() {
  const overview = appState.overview;
  const actions = await api("/api/actions");
  appState.actions = actions.actions || [];
  qs("#app").innerHTML = `
    <section class="grid cols-2">
      <section class="panel">
        <h2>关键配置</h2>
        ${keyValueGrid(overview.config)}
      </section>
      <section class="panel">
        <h2>运行命令</h2>
        <div class="grid">
          ${appState.actions
            .map(
              (action) => `
                <div class="action-row">
                  <div class="toolbar">
                    <div>
                      <h3>${htmlEscape(action.label)}</h3>
                      <p>${htmlEscape(action.description)}</p>
                      <p>${htmlEscape(action.command)}</p>
                    </div>
                    <button class="btn primary" data-action="${htmlEscape(action.name)}">启动</button>
                  </div>
                </div>
              `
            )
            .join("")}
        </div>
      </section>
    </section>
    <section class="panel">
      <h2>任务状态</h2>
      <div id="jobs">暂无任务。</div>
    </section>
  `;
  qsa("[data-action]").forEach((button) => {
    button.addEventListener("click", async () => {
      button.disabled = true;
      try {
        await api("/api/jobs", {
          method: "POST",
          body: JSON.stringify({ action: button.dataset.action }),
        });
        await renderJobs();
      } finally {
        button.disabled = false;
      }
    });
  });
  await renderJobs();
  if (!appState.jobsTimer) {
    appState.jobsTimer = setInterval(() => {
      if (appState.activeView === "config") renderJobs();
    }, 3000);
  }
}

async function renderJobs() {
  const target = qs("#jobs");
  if (!target) return;
  const payload = await api("/api/jobs");
  const jobs = payload.jobs || [];
  if (!jobs.length) {
    target.innerHTML = `<div class="empty">暂无任务。</div>`;
    return;
  }
  target.innerHTML = renderTablePayload({
    title: "任务状态",
    columns: Object.keys(jobs[0]),
    rows: jobs,
    count: jobs.length,
    total_count: jobs.length,
    source: { path: "outputs/ui_jobs" },
  }, { embedded: true });
}

async function renderLogs() {
  const logs = appState.overview.recent_logs || [];
  qs("#app").innerHTML = `
    <section class="grid cols-2">
      <section class="panel">
        <h2>最近日志</h2>
        <div class="grid">
          ${logs
            .map(
              (log) => `
                <button class="btn" data-log="${htmlEscape(log.path)}">
                  ${htmlEscape(log.updated_at || "")} · ${htmlEscape(log.path)}
                </button>
              `
            )
            .join("") || `<div class="empty">暂无日志。</div>`}
        </div>
      </section>
      <section class="panel">
        <h2>日志内容</h2>
        <pre class="log-box" id="log-content">选择左侧日志。</pre>
      </section>
    </section>
  `;
  qsa("[data-log]").forEach((button) => {
    button.addEventListener("click", async () => {
      const payload = await api(`/api/log?path=${encodeURIComponent(button.dataset.log)}`);
      qs("#log-content").textContent = payload.text || "暂无内容。";
    });
  });
}

async function renderActiveView() {
  setPageTitle(appState.activeView);
  if (appState.activeView === "overview") await renderOverview();
  if (appState.activeView === "candidates") await renderCandidates();
  if (appState.activeView === "inference") await renderInference();
  if (appState.activeView === "trading") await renderTrading();
  if (appState.activeView === "config") await renderConfig();
  if (appState.activeView === "logs") await renderLogs();
}

async function refresh() {
  qs("#app").innerHTML = `<div class="empty">正在读取本地状态。</div>`;
  appState.tableCache.clear();
  appState.overview = await api("/api/overview");
  await renderActiveView();
}

function bindNav() {
  qsa(".nav-item").forEach((button) => {
    button.addEventListener("click", async () => {
      qsa(".nav-item").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      appState.activeView = button.dataset.view;
      await renderActiveView();
    });
  });
  qs("#refresh-btn").addEventListener("click", refresh);
}

async function main() {
  bindNav();
  try {
    await refresh();
  } catch (error) {
    qs("#app").innerHTML = `<div class="empty">加载失败：${htmlEscape(error.message)}</div>`;
  }
}

main();
