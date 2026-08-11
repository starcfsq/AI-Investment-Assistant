async function getJSON(url, options) {
  const resp = await fetch(url, options);
  if (!resp.ok) throw new Error(url + " -> " + resp.status);
  return resp.json();
}

function card(label, value, extra) {
  return `<div class="card"><div class="label">${label}</div>` +
         `<div class="value">${value ?? "—"}</div>${extra ? `<div class="label">${extra}</div>` : ""}</div>`;
}

function table(headers, rows) {
  if (!rows || rows.length === 0) return "<p>暂无数据</p>";
  const h = headers.map(x => `<th>${x}</th>`).join("");
  const body = rows.map(r => `<tr>${r.map(c => `<td>${c ?? ""}</td>`).join("")}</tr>`).join("");
  return `<table><thead><tr>${h}</tr></thead><tbody>${body}</tbody></table>`;
}

async function refreshDashboard() {
  const d = await getJSON("/api/dashboard");
  document.getElementById("data-until").textContent = "数据截至 " + d.data_until;
  const warns = (d.analysis.warnings || []);
  document.querySelectorAll(".warnings").forEach(el => el.remove());
  const warnsHtml = warns.length
    ? `<div class="warnings">⚠️ ${warns.map(w => `<span>${w}</span>`).join("　")}</div>` : "";
  document.querySelector("header").insertAdjacentHTML("afterend", warnsHtml);
  const t = d.analysis.trend || {};
  const a = d.account || {};
  const cards = [
    card("趋势状态", t.state, "综合分 " + (t.composite ?? "—")),
    card("账户净值", "¥" + (a.nav ?? 0).toLocaleString(), "阶段 " + a.period_id),
    card("阶段收益率", (a.return_pct ?? 0) + "%"),
    card("操作胜率", Math.round((a.win_rate ?? 0) * 100) + "%"),
  ];
  document.getElementById("cards").innerHTML = cards.join("");

  document.getElementById("sectors").innerHTML = table(
    ["板块", "RS", "资金流", "动量", "得分"],
    (d.analysis.sectors || []).map(s => [s.name, s.rs, s.flow, s.momentum, s.score]));
  document.getElementById("stocks").innerHTML = table(
    ["代码", "名称", "得分"],
    (d.analysis.stocks || []).map(s => [s.code, s.name, s.score]));
  const p = d.analysis.portfolio || {};
  document.getElementById("portfolio").innerHTML =
    `<p>${p.summary || ""}　${p.rebalance_rule || ""}</p>` + table(
      ["组合", "名称", "权重"],
      (p.core ? [["核心", p.core.name, p.core.weight]] : [])
        .concat((p.satellite || []).map(s => ["卫星", s.name, s.weight])));
  // 投资账户：显示账户统计
  document.getElementById("account").innerHTML = table(
    ["阶段", "净值", "现金", "持仓市值", "胜率", "收益率"],
    [[a.period_id, a.nav, a.cash, a.holdings_value,
      a.win_rate, (a.return_pct ?? 0) + "%"]]);
  // 阶段历史：显示归档阶段
  document.getElementById("periods").innerHTML = table(
    ["阶段", "胜率", "收益率", "基准"],
    (d.periods || []).map(p => [p.period_id, p.win_rate, p.return_pct + "%", p.benchmark_return]));
  // 市场趋势：显示信号明细
  document.getElementById("trend").innerHTML = t.state
    ? table(["状态", "MA偏离", "PE百分位", "股债性价比", "综合分"],
        [[t.state, t.detail?.ma_dev, t.detail?.pe_pct, t.detail?.bond_equity_pct, t.composite]])
    : "<p>暂无趋势数据</p>";
}

async function runAnalyze() {
  const d = await getJSON("/api/analyze", { method: "POST" });
  pushMsg("assistant", "分析完成：" + (d.ai.trend?.state || d.analysis.trend?.state || ""));
  refreshDashboard();
}

async function runBacktest() {
  const d = await getJSON("/api/backtest");
  const r = d.result || {};
  const body = table(["状态", "版本", "胜率"],
    [[r.status, r.version, r.new_win_rate ?? r.win_rate ?? ""]]);
  document.getElementById("backtest").innerHTML =
    body + table(["版本", "运行时间", "胜率", "超额收益"],
      (d.iters || []).map(i => [i.version, i.run_at, i.win_rate, i.excess_return]));
}

async function send() {
  const q = document.getElementById("query").value.trim();
  if (!q) return;
  pushMsg("user", q);
  document.getElementById("query").value = "";
  const d = await getJSON("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query: q, symbol: document.getElementById("symbol").value.trim() || null }),
  });
  const refs = (d.references || []).map(r =>
    `<div class="ref">· ${r.title}（${r.date}，${r.source}）</div>`).join("");
  pushMsg("assistant", d.answer + ` <small>置信度 ${d.confidence}</small>` + refs);
}

function pushMsg(role, text) {
  const box = document.getElementById("messages");
  box.insertAdjacentHTML("beforeend",
    `<div class="msg ${role}">${text.replace(/\n/g, "<br>")}</div>`);
  box.scrollTop = box.scrollHeight;
}

document.getElementById("btn-analyze").addEventListener("click", runAnalyze);
document.getElementById("btn-backtest").addEventListener("click", runBacktest);
document.getElementById("btn-send").addEventListener("click", send);
document.getElementById("query").addEventListener("keydown", e => { if (e.key === "Enter") send(); });

refreshDashboard();

async function loadSimulation() {
  const d = await getJSON("/api/simulation");
  if (d.error) { document.getElementById("sim-trades").innerHTML = "<p>" + d.error + "</p>"; return; }
  const st = d.stats || {};
  document.getElementById("sim-stats").innerHTML = [
    card("总收益", st.total_return != null ? (st.total_return * 100).toFixed(2) + "%" : "—"),
    card("基准收益", st.benchmark_return != null ? (st.benchmark_return * 100).toFixed(2) + "%" : "—"),
    card("交易笔数", st.n_trades != null ? st.n_trades : "—"),
  ].join("");
  drawSimChart(d.curve);
  document.getElementById("sim-trades").innerHTML = table(
    ["时间", "方向", "名称", "价格", "数量", "盈亏"],
    (d.trades || []).map(t => [t.time, t.side, t.name, t.price, t.qty, t.pnl]));
  document.getElementById("sim-rebalances").innerHTML = table(
    ["日期", "权重"],
    (d.rebalances || []).map(r => [r.date, JSON.stringify(r.weights)]));
}

function drawSimChart(curve) {
  const c = document.getElementById("sim-chart");
  if (!c || !curve || curve.length < 2) return;
  const ctx = c.getContext("2d");
  const W = c.width, H = c.height, pad = 30;
  ctx.clearRect(0, 0, W, H);
  // 净值归一化到 1.0 起，与基准同尺度绘制，避免基准线被钉在底部。
  const base = curve[0].nav || 1;
  const navs = curve.map(p => p.nav / base);
  const ys = navs.concat(curve.map(p => p.benchmark));
  const yMax = Math.max(...ys) * 1.05, yMin = Math.min(...ys) * 0.95;
  if (yMax === yMin) return;  // 平坦曲线：无法绘制有意义的刻度
  const px = i => pad + i / (navs.length - 1) * (W - 2 * pad);
  const py = v => H - pad - (v - yMin) / (yMax - yMin) * (H - 2 * pad);
  ctx.strokeStyle = "#1f77b4"; ctx.beginPath();
  curve.forEach((p, i) => i === 0 ? ctx.moveTo(px(i), py(navs[i])) : ctx.lineTo(px(i), py(navs[i])));
  ctx.stroke();
  ctx.strokeStyle = "#999"; ctx.beginPath();
  curve.forEach((p, i) => i === 0 ? ctx.moveTo(px(i), py(p.benchmark)) : ctx.lineTo(px(i), py(p.benchmark)));
  ctx.stroke();
  ctx.fillStyle = "#333";
  ctx.fillText("模拟净值(归一)", pad + 4, pad + 12);
  ctx.fillText("沪深300", pad + 4, pad + 26);
}

document.getElementById("btn-sim").addEventListener("click", loadSimulation);
