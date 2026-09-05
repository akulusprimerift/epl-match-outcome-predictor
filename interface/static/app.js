"use strict";
const $ = (id) => document.getElementById(id);
const state = {metadata: null, busy: false, lastResult: null};
function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}
function format(value, unit = "") {
  if (value === null || value === undefined) return "Not available";
  return Number(value).toLocaleString("en-GB", {maximumFractionDigits: 2}) + (unit === "%" ? "%" : "");
}
function outcome(key, p) { return key === "draw" ? "Draw" : `${key === "home_win" ? p.home_team : p.away_team} win`; }
function sourceLink(label, href) {
  const link = node("a", label);
  if (href && /^https?:\/\//.test(href)) { link.href = href; link.target = "_blank"; link.rel = "noopener noreferrer"; }
  return link;
}
function table(headers, rows) {
  const wrap = node("div", undefined, "table-scroll"), t = node("table"), head = node("thead"), hr = node("tr"), body = node("tbody");
  headers.forEach((label) => { const th = node("th", label); th.scope = "col"; hr.append(th); });
  head.append(hr);
  rows.forEach((cells) => { const tr = node("tr"); cells.forEach((cell) => {const td = node("td"); td.append(cell instanceof Node ? cell : document.createTextNode(String(cell))); tr.append(td);}); body.append(tr); });
  t.append(head, body); wrap.append(t); return wrap;
}
function observedCell(row, role) {
  const element = node("span", format(row[role], row.unit), "number");
  if (row[`${role}_imputed`]) element.append(node("small", `Model uses training median ${format(row[`${role}_model_input`], row.unit)}`, "imputed"));
  return element;
}
function renderMatches(team) {
  const box = node("section"); box.append(node("h3", team.name));
  if (!team.recent_matches.length) box.append(node("p", "No prior EPL matches available. Missing model inputs use training medians."));
  else box.append(table(["Date / source", "Opponent", "H/A", "Score*", "Form"], team.recent_matches.map((m) => [sourceLink(m.date, m.source_url), m.opponent, m.venue === "Home" ? "H" : "A", `${m.goals_for}–${m.goals_against}`, node("span", m.result, `pill ${m.result}`)])));
  const detail = node("details", undefined, "technical"); detail.append(node("summary", `Venue window: last ${team.venue_count} ${team.role === "home" ? "home" : "away"} matches`));
  detail.append(table(["Date / source", "Opponent", "Score*", "Points"], team.venue_matches.map((m) => [sourceLink(m.date, m.source_url), m.opponent, `${m.goals_for}–${m.goals_against}`, m.points])));
  box.append(detail);
  const possession = node("p", `Possession source: ${team.possession_source_season.slice(0,2)}/${team.possession_source_season.slice(2)} EPL season. `, "muted");
  possession.append(team.possession_source_url ? sourceLink(`${team.possession_matches} matches · SofaScore`, team.possession_source_url) : node("span", "No previous-season EPL average for this club."));
  box.append(possession); return box;
}
function render(result) {
  const p = result.prediction, root = $("results"); root.replaceChildren(); root.hidden = false; $("empty").hidden = true;
  const top = node("div", undefined, "result-top"), title = node("div"); title.append(node("p", "MODEL B · FROZEN PREDICTION", "eyebrow"), node("h2", `${p.home_team} vs ${p.away_team}`)); top.append(title, node("span", p.match_date, "date-tag")); root.append(top);
  const probabilities = node("div", undefined, "probabilities");
  ["home_win", "draw", "away_win"].forEach((key) => { const card = node("div", undefined, `probability${p.predicted_outcome === key ? " leading" : ""}`); card.append(node("h3", outcome(key,p)), node("strong", `${(100*p.probabilities[key]).toFixed(1)}%`)); if (p.predicted_outcome === key) card.append(node("span", "LEADING OUTCOME", "chip")); probabilities.append(card); });
  root.append(probabilities, node("p", result.summary, "summary"));
  const notice = node("details", undefined, "notice"); notice.open = true; notice.append(node("summary", `Data as of ${p.feature_as_of} — read before using this prediction`));
  const warnings = node("ul"); p.warnings.forEach((w) => warnings.append(node("li",w))); notice.append(warnings); root.append(notice);
  const influences = node("section", undefined, "panel"); influences.append(node("h2", "Why the model leans this way"), node("p", `Comparing ${outcome(result.attribution.leading_outcome,p)} with ${outcome(result.attribution.comparison_outcome,p)}. These are matchup-specific influences from the saved model, not a generic feature-importance chart.`, "muted"));
  const factors = node("div", undefined, "factors");
  result.attribution.groups.forEach((g) => {const card=node("article",undefined,`factor ${g.direction}`), s=g.statistics; card.append(node("span",g.direction === "supports" ? "Favors the leading outcome" : g.direction === "opposes" ? "Pushes against the leading outcome" : "Neutral influence","direction"),node("h3",g.group),node("p",`${s.label}: ${p.home_team} ${format(s.home,s.unit)}; ${p.away_team} ${format(s.away,s.unit)}.`)); if(s.home_imputed || s.away_imputed) card.append(node("p","Missing or insufficient observations in this group use frozen training medians; observed statistics are shown separately.","muted")); factors.append(card);});
  influences.append(factors, node("p",result.attribution.limitations,"muted"));
  const technical=node("details",undefined,"technical"); technical.append(node("summary","How the model influences were verified"),node("p",result.attribution.method),node("p",result.attribution.units),table(["Group","Score contribution"],result.attribution.groups.map(g=>[g.group,g.contribution.toFixed(4)])),node("p",`Learned baseline difference: ${result.attribution.baseline_gap.toFixed(4)}. Final score difference: ${result.attribution.total_score_gap.toFixed(4)}. The full three-class explanation is checked to reconstruct the displayed probabilities.`)); influences.append(technical);root.append(influences);
  const stats=node("section",undefined,"panel");stats.append(node("h2","The numbers behind the comparison"),node("p",`Recent windows: ${p.home_team} ${result.teams.home.recent_count} matches; ${p.away_team} ${result.teams.away.recent_count} matches. Venue windows are separate. Shots use ${result.teams.home.shots_observations} and ${result.teams.away.shots_observations} available observations respectively.`,"muted"),table(["Observed statistic",p.home_team,p.away_team],result.statistics.map(r=>[r.label,observedCell(r,"home"),observedCell(r,"away")])),node("p",result.evidence_note,"muted"));root.append(stats);
  const records=node("section",undefined,"panel"),matches=node("div",undefined,"matches"); records.append(node("h2","Check the actual matches"),node("p","Most recent first. *Score is shown from the named club’s perspective. Links open the original provider source.","muted"));matches.append(renderMatches(result.teams.home),renderMatches(result.teams.away));records.append(matches);root.append(records);
  const limits=node("section",undefined,"panel");limits.append(node("h2","Keep the limitations in view"),node("p",result.model_caveat),node("p","This is an offline snapshot, not current form, a betting recommendation, or proof that a team will win. No injuries, lineups, or other-league statistics are included.","muted"));root.append(limits);
}
async function predictMatch(input) {
  if (state.busy) throw new Error("A prediction is already running.");
  if (!state.metadata) throw new Error("The frozen snapshot is not ready yet.");
  if (!input || typeof input !== "object" || Array.isArray(input) || Object.keys(input).sort().join(",") !== "away,date,home") throw new Error("Provide only home, away, and date.");
  if (![input.home,input.away,input.date].every(value=>typeof value === "string")) throw new Error("Team names and date must be text.");
  if (!input.home || !input.away || !input.date) throw new Error("Choose both teams and a match date.");
  if (![input.home,input.away].every(team=>state.metadata.teams.includes(team))) throw new Error("Choose a team from the stored EPL clubs.");
  if (input.home === input.away) throw new Error("Choose two different teams.");
  if (!/^\d{4}-\d{2}-\d{2}$/.test(input.date)) throw new Error("Use a date in YYYY-MM-DD format.");
  if (input.date <= state.metadata.snapshot_date) throw new Error(`Choose a date after ${state.metadata.snapshot_date}; historical and same-date requests are not supported.`);
  $("home").value=input.home;$("away").value=input.away;$("date").value=input.date;
  state.busy=true; $("predict").disabled=true; $("status").textContent="Checking the frozen model and calculating the evidence…"; $("error").hidden=true; $("results").hidden=true;
  try {
    const response=await fetch("/api/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(input)}), result=await response.json();
    if(!response.ok)throw new Error(result.error||"The prediction could not be completed.");
    state.lastResult=result;render(result);$("status").textContent="Prediction ready. Review the snapshot warnings and evidence below.";$("results").focus({preventScroll:true});return result;
  } finally {state.busy=false;$("predict").disabled=false;}
}
function showError(error) {$("error").textContent=error.message||String(error);$("error").hidden=false;$("status").textContent="";$("results").hidden=true;state.lastResult=null;}
$("match-form").addEventListener("submit", async e=>{e.preventDefault();try{await predictMatch({home:$("home").value,away:$("away").value,date:$("date").value});}catch(error){showError(error);}});
$("swap").addEventListener("click",()=>{const value=$("home").value;$("home").value=$("away").value;$("away").value=value;});
async function initialize(){try{const response=await fetch("/api/metadata"),data=await response.json();if(!response.ok)throw new Error(data.error);state.metadata=data;["home","away"].forEach(id=>{$(id).replaceChildren(...data.teams.map(team=>{const option=node("option",team);option.value=team;return option;}));$(id).disabled=false;});$("home").value="Arsenal";$("away").value="Chelsea";$("date").min=data.minimum_date;$("date").value=data.default_date;$("date").disabled=false;$("predict").disabled=false;$("swap").disabled=false;$("snapshot").textContent=`Stored matches through ${data.snapshot_date}. Forecasts use historical data only; no live schedule is checked.`;}catch(error){$("snapshot").textContent="Snapshot unavailable.";showError(error);}}
initialize();

// Optional page-scoped agent access uses the exact same action and visible result.
const modelContext = document.modelContext;
if (modelContext?.registerTool) {
  const lifecycle = new AbortController();
  window.addEventListener("pagehide", () => lifecycle.abort(), {once:true});
  try {
    Promise.resolve(modelContext.registerTool({
      name:"predict_epl_match", title:"Predict an EPL match",
      description:"Calculate and display an offline prediction with actual historical evidence. Use canonical teams from the visible selectors and a YYYY-MM-DD date after the snapshot. Does not check scheduling or fetch live data.",
      inputSchema:{type:"object",properties:{home:{type:"string"},away:{type:"string"},date:{type:"string"}},required:["home","away","date"],additionalProperties:false},
      annotations:{readOnlyHint:false,untrustedContentHint:false},
      async execute(input){try{const result=await predictMatch(input);return {prediction:result.prediction,summary:result.summary,statistics:result.statistics};}catch(error){showError(error);throw error;}}
    },{signal:lifecycle.signal})).catch(()=>{/* Normal browser controls remain available. */});
  } catch (_) { /* Unsupported implementations do not prevent ordinary use. */ }
}
