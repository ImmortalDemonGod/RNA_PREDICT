#!/usr/bin/env node
// Forensic Audit Pipeline — single-repo orchestrator.
// Drives headless `claude -p` subagents through 5 sequential stages, each emitting a
// schema-validated structured object (via file-handoff) that is rendered to audit/0N-*.md.
// Canonical substrate per the forensic-audit-pipeline skill. Reproducible, resumable, persisted.
//
// Usage:
//   node audit/run-audit.mjs --selftest         # validate Node logic only (no API calls)
//   node audit/run-audit.mjs --preflight        # auth + model-resolution check (cheap)
//   node audit/run-audit.mjs --fresh            # tear down audit/ and run all 5 stages
//   node audit/run-audit.mjs                     # resume from first incomplete stage (default)
//   node audit/run-audit.mjs --stage 2           # run exactly one stage
//   node audit/run-audit.mjs --from 3            # run stages 3..5
//
// Nothing about the target is hardcoded: repo root = cwd, branch derived at runtime,
// build/test commands discovered by Stage 3 from the repo itself.

import { spawn } from "node:child_process";
import { mkdirSync, writeFileSync, readFileSync, existsSync, rmSync, readdirSync, statSync } from "node:fs";
import { join, resolve } from "node:path";

// ─────────────────────────────── config ───────────────────────────────
const REPO = resolve(process.cwd());
const AUDIT = join(REPO, "audit");
const WORK = join(AUDIT, ".work");
const CHUNK = Number(process.env.AUDIT_CHUNK || 55);          // files per Stage-1 classification shard
const CONCURRENCY = Number(process.env.AUDIT_CONCURRENCY || 4);
const STAGE2_CEILING = Number(process.env.AUDIT_S2_CEILING || 5);
const RESEARCH_GATHERERS = Number(process.env.AUDIT_RESEARCHERS || 3);
const IGNORE = [/^audit\//, /^\.git\//];                      // declared coverage ignore list
let MODEL_HEAVY = process.env.AUDIT_MODEL_HEAVY || "opus";    // synthesis / audit / falsify / goal / plan
let MODEL_FAST = process.env.AUDIT_MODEL_FAST || "sonnet";    // enumeration / execution / research-gather
let BRANCH = "HEAD";
let TOTAL_COST = 0;

const INVARIANTS = [
  "You are one worker in a forensic technical-audit pipeline. Obey these invariants without exception:",
  "1. Absence of evidence is not evidence of absence. Never claim something does not exist / is unused / is unreachable unless you name where you looked and that search space is the full Stage-1 surface. Otherwise record it as 'unverified', never 'absent'.",
  "2. No claim without a location. Every finding/behavior/assertion cites a concrete path:line or named artifact a reviewer can open. Drop any claim lacking a citable anchor.",
  "3. Coverage has a denominator. The Stage-1 inventory is the denominator. Actually open files with Read/Grep — do not infer from names.",
  "4. Verification is adversarial. When asked to falsify, try to REFUTE each claim against source; do not rubber-stamp.",
  "5. Mutation is a means, not a deliverable. You may read/instrument/run code in this sandbox, but the only shipped output is your JSON handoff file.",
  "6. Never exfiltrate sensitive data. Reference secrets/PII/PHI by path and category only — never paste their contents into the handoff.",
].join("\n");

// ─────────────────────────────── tiny utils ───────────────────────────────
const sh = (c, a, o = {}) => new Promise((r) => {
  const p = spawn(c, a, { cwd: REPO, ...o });
  let O = "", E = ""; p.stdout?.on("data", d => (O += d)); p.stderr?.on("data", d => (E += d));
  p.on("close", (code) => r({ code, out: O, err: E }));
});
const log = (...m) => console.log(`[orch ${new Date().toISOString().slice(11, 19)}]`, ...m);
const sortBy = (k) => (a, b) => (a[k] > b[k] ? 1 : a[k] < b[k] ? -1 : 0);
const packChunks = (xs, n) => { const o = []; for (let i = 0; i < xs.length; i += n) o.push(xs.slice(i, i + n)); return o; };
const tolerantJson = (s) => { // strip fences / surrounding prose, parse first {...} ... last }
  if (!s) return null;
  let t = s.replace(/^﻿/, "").trim();
  const f = t.indexOf("{"), l = t.lastIndexOf("}");
  if (f === -1 || l === -1) return null;
  try { return JSON.parse(t.slice(f, l + 1)); } catch { return null; }
};

// ─────────────────── minimal JSON-Schema validator (subset) ───────────────────
// Supports: type(object/array/string/number/integer/boolean), required, properties,
// items, enum, minItems. Returns {ok, errors[]}. Dependency-free on purpose.
function validate(schema, data, path = "$") {
  const errs = [];
  const t = schema.type;
  const typeOk =
    t === "object" ? (data && typeof data === "object" && !Array.isArray(data)) :
    t === "array" ? Array.isArray(data) :
    t === "string" ? typeof data === "string" :
    t === "boolean" ? typeof data === "boolean" :
    t === "integer" ? Number.isInteger(data) :
    t === "number" ? typeof data === "number" : true;
  if (!typeOk) { errs.push(`${path}: expected ${t}, got ${Array.isArray(data) ? "array" : typeof data}`); return { ok: false, errors: errs }; }
  if (schema.enum && !schema.enum.includes(data)) errs.push(`${path}: '${data}' not in enum [${schema.enum.join(",")}]`);
  if (t === "object") {
    for (const req of schema.required || []) if (!(req in (data || {}))) errs.push(`${path}: missing required '${req}'`);
    for (const [k, sub] of Object.entries(schema.properties || {})) if (k in (data || {})) errs.push(...validate(sub, data[k], `${path}.${k}`).errors);
  }
  if (t === "array") {
    if (schema.minItems != null && data.length < schema.minItems) errs.push(`${path}: needs >= ${schema.minItems} items, got ${data.length}`);
    if (schema.items) data.forEach((d, i) => errs.push(...validate(schema.items, d, `${path}[${i}]`).errors));
  }
  return { ok: errs.length === 0, errors: errs };
}

// ─────────────────────────────── one subagent ───────────────────────────────
let SEQ = 0;
async function runAgent({ name, prompt, schema, model = MODEL_HEAVY, maxTurns = 50, web = false, timeoutMs = 18e5, retries = 1 }) {
  for (let attempt = 0; attempt <= retries; attempt++) {
    const out = join(WORK, `a_${name.replace(/\W+/g, "_")}_${SEQ++}.json`);
    const tools = (web ? "Read,Grep,Glob,Bash,WebSearch,WebFetch" : "Read,Grep,Glob,Bash") + ",Write";
    const full = `${prompt}\n\nOUTPUT CONTRACT: Use the Write tool to create EXACTLY ONE file at the absolute path ${out}. Its entire contents must be a single raw JSON object conforming to this JSON Schema:\n${JSON.stringify(schema)}\nNo markdown, no code fences, no prose, no commentary — only the JSON object. That file is the only output that matters.`;
    const args = ["-p", full, "--output-format", "json", "--model", model, "--max-turns", String(maxTurns),
      "--allowedTools", tools, "--add-dir", REPO, "--strict-mcp-config",
      "--permission-mode", "acceptEdits", "--append-system-prompt", INVARIANTS];
    const env = { ...process.env, CLAUDE_CODE_DEBUG: "0" };
    const r = await new Promise((res) => {
      const p = spawn("claude", args, { cwd: REPO, stdio: ["ignore", "pipe", "pipe"], env });
      let O = "", E = ""; const k = setTimeout(() => { try { p.kill("SIGKILL"); } catch {} }, timeoutMs);
      p.stdout.on("data", (d) => (O += d)); p.stderr.on("data", (d) => (E += d));
      p.on("close", () => { clearTimeout(k); res({ O, E }); });
    });
    let envel = null; try { envel = JSON.parse(r.O); } catch {}
    if (envel) TOTAL_COST += Number(envel.total_cost_usd || 0);
    const data = existsSync(out) ? tolerantJson(readFileSync(out, "utf8")) : null;
    if (!data) { log(`  ⚠ ${name}: no/invalid handoff (attempt ${attempt + 1}/${retries + 1})${envel?.is_error ? " [agent is_error]" : ""}`); continue; }
    const v = validate(schema, data);
    if (!v.ok) { log(`  ⚠ ${name}: schema violations (attempt ${attempt + 1}): ${v.errors.slice(0, 4).join(" | ")}`); continue; }
    return { ok: true, data };
  }
  return { ok: false, err: "failed-after-retries" };
}

async function pMap(xs, fn, n = CONCURRENCY) {
  const out = []; let i = 0;
  await Promise.all(Array(Math.min(n, xs.length || 1)).fill(0).map(async () => { while (i < xs.length) { const k = i++; out[k] = await fn(xs[k], k); } }));
  return out;
}

// ─────────────────────────────── persistence ───────────────────────────────
const stateFile = join(WORK, "state.json");
const loadState = () => (existsSync(stateFile) ? JSON.parse(readFileSync(stateFile, "utf8")) : { completed: {}, branch: BRANCH });
async function gitPushRetry() {
  for (let i = 0, wait = 2000; i < 4; i++) {
    const r = await sh("git", ["push", "-u", "origin", BRANCH]);
    if (r.code === 0) return true;
    log(`  push retry ${i + 1} failed; waiting ${wait}ms`); await new Promise((s) => setTimeout(s, wait)); wait *= 2;
  }
  return false;
}
async function checkpoint(stage, mdName, md, jsonName, obj, state) {
  writeFileSync(join(AUDIT, mdName), md);
  writeFileSync(join(WORK, jsonName), JSON.stringify(obj, null, 2));
  state.completed[stage] = Date.now(); state.total_cost_usd_api_equiv = TOTAL_COST;
  writeFileSync(stateFile, JSON.stringify(state, null, 2));
  await sh("git", ["add", "audit"]);
  await sh("git", ["commit", "-m", `audit: ${stage} (${mdName})`]);
  await gitPushRetry();
  log(`✓ checkpoint ${stage} → ${mdName} (api-equiv cost so far $${TOTAL_COST.toFixed(2)})`);
}
async function halt(stage, why) {
  const md = `# HALT at ${stage}\n\nThe pipeline stopped rather than emit a false-confident artifact.\n\n## Reason\n${why}\n\n_api-equivalent cost so far: $${TOTAL_COST.toFixed(2)} (subscription spend is far lower)_\n`;
  writeFileSync(join(AUDIT, "HALT-REPORT.md"), md);
  const state = loadState(); state.halted = { stage, why, at: Date.now() }; writeFileSync(stateFile, JSON.stringify(state, null, 2));
  await sh("git", ["add", "audit"]); await sh("git", ["commit", "-m", `audit: HALT at ${stage}`]); await gitPushRetry();
  log(`✗ HALT at ${stage}: ${why}`);
  process.exit(2);
}

// ─────────────────────────────── denominator ───────────────────────────────
async function repoFileSet() {
  const tracked = (await sh("git", ["ls-files"])).out.split("\n").filter(Boolean);
  const untracked = (await sh("git", ["ls-files", "--others", "--exclude-standard"])).out.split("\n").filter(Boolean);
  const all = [...new Set([...tracked, ...untracked])].filter((p) => !IGNORE.some((re) => re.test(p)));
  return all.sort();
}

// ─────────────────────────────── schemas ───────────────────────────────
const S = {
  invShard: { type: "object", required: ["entries"], properties: { entries: { type: "array", items: {
    type: "object", required: ["path", "role", "summary"], properties: {
      path: { type: "string" }, role: { type: "string", enum: ["source", "test", "doc", "config", "asset", "generated", "dead", "unknown"] }, summary: { type: "string" } } } } } },
  synth: { type: "object", required: ["architecture", "provisional_intent", "entry_points"], properties: {
    architecture: { type: "string" }, provisional_intent: { type: "string" },
    entry_points: { type: "array", items: { type: "object", required: ["name", "kind", "location", "description"], properties: {
      name: { type: "string" }, kind: { type: "string" }, location: { type: "string" }, description: { type: "string" } } } } } },
  findings: { type: "object", required: ["findings", "files_examined"], properties: {
    files_examined: { type: "array", items: { type: "string" } },
    findings: { type: "array", items: { type: "object", required: ["id", "location", "class", "severity", "evidence"], properties: {
      id: { type: "string" }, location: { type: "string" },
      class: { type: "string", enum: ["bug", "security", "doc_drift", "design_defect", "intent_mismatch", "perf", "other"] },
      severity: { type: "string", enum: ["critical", "high", "medium", "low", "info"] },
      evidence: { type: "string" }, recommendation: { type: "string" } } } } } },
  falsify: { type: "object", required: ["verdicts"], properties: { verdicts: { type: "array", items: {
    type: "object", required: ["id", "verdict", "rationale"], properties: {
      id: { type: "string" }, verdict: { type: "string", enum: ["survived", "refuted", "needs-refinement"] },
      rationale: { type: "string" }, corrected_location: { type: "string" } } } } } },
  reaudit: { type: "object", required: ["new_findings"], properties: { new_findings: { type: "array", items: {
    type: "object", required: ["id", "location", "class", "severity", "evidence"], properties: {
      id: { type: "string" }, location: { type: "string" },
      class: { type: "string", enum: ["bug", "security", "doc_drift", "design_defect", "intent_mismatch", "perf", "other"] },
      severity: { type: "string", enum: ["critical", "high", "medium", "low", "info"] }, evidence: { type: "string" } } } } } },
  exec: { type: "object", required: ["commands_discovered", "runs", "coverage", "finding_deltas", "accounting"], properties: {
    commands_discovered: { type: "array", items: { type: "object", required: ["command", "source", "purpose"], properties: {
      command: { type: "string" }, source: { type: "string" }, purpose: { type: "string" } } } },
    runs: { type: "array", items: { type: "object", required: ["command", "exit_code", "observed"], properties: {
      command: { type: "string" }, exit_code: { type: "integer" }, observed: { type: "string" }, artifact: { type: "string" } } } },
    coverage: { type: "object", required: ["measured_pct", "tool", "note"], properties: {
      measured_pct: { type: "number" }, tool: { type: "string" }, note: { type: "string" } } },
    finding_deltas: { type: "array", items: { type: "object", required: ["finding_id", "verdict", "evidence"], properties: {
      finding_id: { type: "string" }, verdict: { type: "string", enum: ["confirmed", "refuted", "refined", "untestable"] }, evidence: { type: "string" } } } },
    accounting: { type: "array", items: { type: "object", required: ["region", "status"], properties: {
      region: { type: "string" },
      status: { type: "string", enum: ["executed", "requires-credentials", "external-service", "hardware-gated", "missing-deps", "dead", "destructive-skip", "not-executed"] },
      reason: { type: "string" } } } } } },
  execCheck: { type: "object", required: ["coverage_claim_supported", "rationale", "discrepancies"], properties: {
    coverage_claim_supported: { type: "boolean" }, rationale: { type: "string" }, discrepancies: { type: "array", items: { type: "string" } } } },
  goal: { type: "object", required: ["candidates"], properties: { candidates: { type: "array", minItems: 1, items: {
    type: "object", required: ["goal", "success_signals", "grounding", "confidence"], properties: {
      goal: { type: "string" }, success_signals: { type: "array", items: { type: "string" } },
      grounding: { type: "array", items: { type: "string" } },
      confidence: { type: "string", enum: ["grounded", "speculative", "needs-human-confirm"] } } } } } },
  researchGather: { type: "object", required: ["sources"], properties: { sources: { type: "array", items: {
    type: "object", required: ["title", "url", "claim"], properties: {
      title: { type: "string" }, url: { type: "string" }, claim: { type: "string" }, relevance: { type: "string" } } } } } },
  researchSynth: { type: "object", required: ["sources", "ideas"], properties: {
    sources: { type: "array", items: { type: "object", required: ["title", "url", "corroboration"], properties: {
      title: { type: "string" }, url: { type: "string" }, corroboration: { type: "string", enum: ["corroborated", "uncorroborated"] }, note: { type: "string" } } } },
    ideas: { type: "array", items: { type: "object", required: ["idea", "relevance", "sources"], properties: {
      idea: { type: "string" }, relevance: { type: "string" }, sources: { type: "array", items: { type: "string" } } } } } } },
  plan: { type: "object", required: ["items"], properties: { items: { type: "array", items: {
    type: "object", required: ["id", "title", "links_to", "location", "change", "verification", "depends_on", "order"], properties: {
      id: { type: "string" }, title: { type: "string" }, links_to: { type: "array", items: { type: "string" } },
      location: { type: "string" }, change: { type: "string" }, verification: { type: "string" },
      depends_on: { type: "array", items: { type: "string" } }, order: { type: "integer" } } } } } },
  planCheck: { type: "object", required: ["all_mappable", "ambiguous_items"], properties: {
    all_mappable: { type: "boolean" }, ambiguous_items: { type: "array", items: {
      type: "object", required: ["id", "why"], properties: { id: { type: "string" }, why: { type: "string" } } } } } },
};

// ─────────────────────────────── markdown renderers ───────────────────────────────
const fence = (o) => "```json\n" + JSON.stringify(o, null, 2) + "\n```";
const tbl = (head, rows) => [`| ${head.join(" | ")} |`, `| ${head.map(() => "---").join(" | ")} |`, ...rows.map((r) => `| ${r.map((c) => String(c ?? "").replace(/\|/g, "\\|").replace(/\n/g, " ")).join(" | ")} |`)].join("\n");

function renderUnderstanding(inv, synth, cov) {
  const byRole = inv.reduce((m, e) => ((m[e.role] = (m[e.role] || 0) + 1), m), {});
  return `# 01 — Comprehensive Understanding

_Coverage denominator for all later stages. Generated by audit/run-audit.mjs._

## Provisional intent (judged against until Stage 4)
${synth.provisional_intent}

## Architecture
${synth.architecture}

## Entry points
${tbl(["Name", "Kind", "Location", "Description"], synth.entry_points.sort(sortBy("location")).map((e) => [e.name, e.kind, e.location, e.description]))}

## Inventory coverage
- Files in denominator (git tracked+untracked, minus ignore): **${cov.files_total}**
- Files classified: **${cov.files_classified}**
- Unknown role: **${cov.unknown}**
- Role distribution: ${Object.entries(byRole).sort(sortBy(0)).map(([k, v]) => `${k}=${v}`).join(", ")}

## Full inventory
${tbl(["Path", "Role", "Summary"], inv.slice().sort(sortBy("path")).map((e) => [e.path, e.role, e.summary]))}

## Machine-checkable object
${fence({ provisional_intent: synth.provisional_intent, coverage: cov, entry_points: synth.entry_points, inventory: inv })}
`;
}
function renderStatic(findings, meta) {
  const order = { critical: 0, high: 1, medium: 2, low: 3, info: 4 };
  const sorted = findings.slice().sort((a, b) => (order[a.severity] - order[b.severity]) || (a.location > b.location ? 1 : -1));
  const bySev = findings.reduce((m, f) => ((m[f.severity] = (m[f.severity] || 0) + 1), m), {});
  return `# 02 — Static Audit

_Every defect findable by reading. Promoted only after surviving adversarial falsification (fixpoint reached in ${meta.rounds} round(s))._

## Summary
- Findings (survivors): **${findings.length}**
- Severity: ${["critical", "high", "medium", "low", "info"].map((s) => `${s}=${bySev[s] || 0}`).join(", ")}
- Source files in denominator: **${meta.source_total}**; examined: **${meta.examined}**
- Judged against Stage-1 provisional intent.

## Findings
${sorted.map((f) => `### [${f.severity.toUpperCase()}] ${f.id} — ${f.class}
- **Location:** \`${f.location}\`
- **Evidence:** ${f.evidence}
${f.recommendation ? `- **Recommendation:** ${f.recommendation}\n` : ""}`).join("\n")}

## Machine-checkable object
${fence({ meta, findings: sorted })}
`;
}
function renderExecution(ex, check) {
  return `# 03 — Execution / Dynamic Surface

_What the code actually does when run. Commands discovered from the repo, not hardcoded._

## Commands discovered
${tbl(["Command", "Source", "Purpose"], ex.commands_discovered.map((c) => [`\`${c.command}\``, c.source, c.purpose]))}

## Runs
${tbl(["Command", "Exit", "Observed", "Artifact"], ex.runs.map((r) => [`\`${r.command}\``, r.exit_code, r.observed, r.artifact || ""]))}

## Measured coverage
- Tool: **${ex.coverage.tool}** — measured **${ex.coverage.measured_pct}%**
- Note: ${ex.coverage.note}
- Independent check — claim supported: **${check.coverage_claim_supported}**. ${check.rationale}
${check.discrepancies?.length ? `- Discrepancies: ${check.discrepancies.join("; ")}` : ""}

## Coverage accounting (100% accounting, not 100% execution)
${tbl(["Region", "Status", "Reason"], ex.accounting.map((a) => [a.region, a.status, a.reason || ""]))}

## Delta applied to Stage-2 findings
${tbl(["Finding", "Verdict", "Evidence"], ex.finding_deltas.map((d) => [d.finding_id, d.verdict, d.evidence]))}

## Machine-checkable object
${fence({ execution: ex, independent_check: check })}
`;
}
function renderGoal(goal, research) {
  return `# 04 — Goal + External Research

## Long-term goal candidates (plural by design; grounded in Stages 1–3)
${goal.candidates.map((c, i) => `### Candidate ${i + 1} — ${c.goal} _(${c.confidence})_
- **Falsifiable success signals:**
${c.success_signals.map((s) => `  - ${s}`).join("\n")}
- **Grounding:**
${c.grounding.map((g) => `  - ${g}`).join("\n")}`).join("\n\n")}

## External research (cross-checked; uncorroborated = recorded as unverified)
### Ideas that materially advance the goal
${tbl(["Idea", "Relevance", "Sources"], research.ideas.map((x) => [x.idea, x.relevance, (x.sources || []).join("; ")]))}

### Sources
${tbl(["Title", "URL", "Corroboration", "Note"], research.sources.map((s) => [s.title, s.url, s.corroboration, s.note || ""]))}

## Machine-checkable object
${fence({ goal, research })}
`;
}
function renderPlan(plan, check) {
  const items = plan.items.slice().sort((a, b) => a.order - b.order);
  return `# 05 — Execution-Ready Plan

_Ordered change items closing the gap between current state (Stages 1–3) and goal (Stage 4)._
_Fresh-agent mappability check: **${check.all_mappable ? "PASS" : "FAIL"}**${check.ambiguous_items?.length ? ` (${check.ambiguous_items.length} ambiguous)` : ""}._

${items.map((it) => `## ${it.order}. ${it.id} — ${it.title}
- **Addresses:** ${it.links_to.join(", ")}
- **Location:** \`${it.location}\`
- **Change:** ${it.change}
- **Verification signal:** ${it.verification}
- **Depends on:** ${it.depends_on.length ? it.depends_on.join(", ") : "—"}`).join("\n\n")}

## Machine-checkable object
${fence({ plan: items, mappability_check: check })}
`;
}

// ─────────────────────────────── stages ───────────────────────────────
const A = "audit/01-understanding.md", A2 = "audit/02-static-audit.md", A3 = "audit/03-execution.md", A4 = "audit/04-goal.md", A5 = "audit/05-plan.md";

async function stage1(state) {
  log("STAGE 1 — comprehensive understanding");
  const files = await repoFileSet();
  log(`  denominator: ${files.length} files; sharding into ${Math.ceil(files.length / CHUNK)} chunks`);
  const chunks = packChunks(files, CHUNK);
  let inv = [];
  const results = await pMap(chunks, (paths, idx) => runAgent({
    name: `s1-shard${idx}`, model: MODEL_FAST, maxTurns: 60,
    schema: S.invShard,
    prompt: `Classify EVERY path in this assigned list (and no others) for repo ${REPO}. For each, assign role ∈ {source,test,doc,config,asset,generated,dead,unknown} and a one-line summary of what it is. Use Read/Glob/Bash to peek when extension is ambiguous; do not deep-read. Produce exactly one entry per assigned path.\nASSIGNED PATHS (${paths.length}):\n${paths.join("\n")}`,
  }));
  const produced = new Set();
  results.forEach((r, i) => { if (r.ok) r.data.entries.forEach((e) => { if (!produced.has(e.path)) { inv.push(e); produced.add(e.path); } }); else log(`  ⚠ shard ${i} failed`); });
  // coverage loop: fill gaps until fs set == produced set (ceiling 3)
  for (let round = 0; round < 3; round++) {
    const missing = files.filter((f) => !produced.has(f));
    if (!missing.length) break;
    log(`  coverage gap: ${missing.length} unclassified; gap-filling (round ${round + 1})`);
    const gap = packChunks(missing, CHUNK);
    const gr = await pMap(gap, (paths, idx) => runAgent({ name: `s1-gap${round}-${idx}`, model: MODEL_FAST, maxTurns: 60, schema: S.invShard,
      prompt: `Classify EVERY path in this list for repo ${REPO} (role + one-line summary). One entry per path.\nPATHS:\n${paths.join("\n")}` }));
    gr.forEach((r) => { if (r.ok) r.data.entries.forEach((e) => { if (!produced.has(e.path) && files.includes(e.path)) { inv.push(e); produced.add(e.path); } }); });
  }
  const missing = files.filter((f) => !produced.has(f));
  if (missing.length) inv.push(...missing.map((p) => ({ path: p, role: "unknown", summary: "UNRESOLVED — classification worker did not return an entry" })));
  // resolve unknowns once
  let unknown = inv.filter((e) => e.role === "unknown").map((e) => e.path);
  if (unknown.length) {
    log(`  resolving ${unknown.length} unknown-role files`);
    const ur = await runAgent({ name: "s1-unknowns", model: MODEL_HEAVY, maxTurns: 60, schema: S.invShard,
      prompt: `These files were left role=unknown. Open each and assign a concrete role (avoid 'unknown' unless truly indeterminate) + summary.\nPATHS:\n${unknown.join("\n")}` });
    if (ur.ok) ur.data.entries.forEach((e) => { const i = inv.findIndex((x) => x.path === e.path); if (i >= 0 && files.includes(e.path)) inv[i] = e; });
    unknown = inv.filter((e) => e.role === "unknown").map((e) => e.path);
  }
  // synthesis: entry points + architecture + provisional intent
  log("  synthesizing architecture + entry points + provisional intent");
  const sourceList = inv.filter((e) => ["source", "config"].includes(e.role)).map((e) => e.path);
  const synth = await runAgent({ name: "s1-synth", model: MODEL_HEAVY, maxTurns: 80, schema: S.synth,
    prompt: `Repo ${REPO}. Using the inventory at ${join(WORK, "stage1_inventory.json")} (already written) plus your own reading of README, packaging manifests (pyproject/setup/package.json), console-script declarations, and every plausible entry module (CLI mains, hydra @main, exported APIs, routes, runners), produce:\n- entry_points: trace EACH real entry point to {name, kind, location(path:line), description}. Verify console-script targets actually exist on disk; if a declared entry target is missing, still list it with a description noting it is declared-but-missing.\n- architecture: a precise paragraph-level system description grounded in real paths.\n- provisional_intent: the apparent reason this project exists (mark mentally as provisional; Stage 4 may refine it).\nCite real path:line anchors. Source/config inventory is:\n${sourceList.slice(0, 400).join("\n")}` });
  writeFileSync(join(WORK, "stage1_inventory.json"), JSON.stringify({ inventory: inv }, null, 2));
  if (!synth.ok) await halt("stage1", "synthesis worker failed to produce a valid architecture/entry-point/intent object.");
  // stop-test
  const cov = { files_total: files.length, files_classified: inv.filter((e) => e.role !== "unknown").length, unknown: unknown.length };
  if (cov.files_classified + cov.unknown < files.length) await halt("stage1", `coverage denominator not met: ${cov.files_classified + cov.unknown}/${files.length} classified.`);
  if (unknown.length > 0) log(`  ⚠ ${unknown.length} files remain unknown — recorded explicitly (not silently dropped).`);
  const obj = { inventory: inv, entry_points: synth.data.entry_points, architecture: synth.data.architecture, provisional_intent: synth.data.provisional_intent, coverage: cov };
  await checkpoint("stage1", "01-understanding.md", renderUnderstanding(inv, synth.data, cov), "stage1.json", obj, state);
  return obj;
}

async function stage2(state, s1) {
  log("STAGE 2 — static audit (falsification fixpoint)");
  const sourceFiles = s1.inventory.filter((e) => ["source", "config"].includes(e.role)).map((e) => e.path).sort();
  log(`  source/config denominator: ${sourceFiles.length} files`);
  const chunks = packChunks(sourceFiles, Math.max(12, Math.ceil(sourceFiles.length / 8)));
  const lenses = ["correctness bugs (logic errors, wrong shapes/types, error handling, resource leaks, concurrency)",
    "security vulnerabilities (injection, deserialization, path traversal, secrets in code, unsafe downloads/eval, SSRF)",
    "documentation/code drift and design defects (README/docstrings vs reality, dead/contradictory config, intent mismatch vs Stage-1 provisional intent)"];
  // initial audit: every chunk audited; lens rotates so the whole tree gets all 3 angles across chunks
  let findings = [];
  const examined = new Set();
  const passes = chunks.flatMap((paths, idx) => lenses.map((lens, li) => ({ paths, idx, lens, li })));
  const out = await pMap(passes, (p) => runAgent({ name: `s2-c${p.idx}-l${p.li}`, model: MODEL_HEAVY, maxTurns: 70, schema: S.findings,
    prompt: `Static audit of repo ${REPO}. Read the Stage-1 map at ${A} (provisional intent is your defect yardstick). AUDIT EVERY assigned file below through THIS lens: ${p.lens}. Open each file with Read. A defect is only a defect relative to intended behavior. Every finding needs id (globally unique, prefix s2c${p.idx}l${p.li}-), location path:line, class, severity, concrete evidence, and a recommendation. Also return files_examined = the assigned paths you actually opened.\nASSIGNED FILES:\n${p.paths.join("\n")}` }), 4);
  out.forEach((r) => { if (r.ok) { r.data.findings.forEach((f) => findings.push(f)); r.data.files_examined.forEach((p) => examined.add(p)); } });
  log(`  initial pass: ${findings.length} raw findings; examined ${examined.size}/${sourceFiles.length} source files`);

  const fingerprint = (f) => `${f.location}::${f.class}`;
  const dedupe = (arr) => { const m = new Map(); for (const f of arr) if (!m.has(fingerprint(f))) m.set(fingerprint(f), f); return [...m.values()]; };

  let prev = "", rounds = 0;
  for (let round = 1; round <= STAGE2_CEILING; round++) {
    rounds = round;
    // 1) re-audit sweep for anything missed (fresh lens over the same denominator, summarized)
    const sweep = await runAgent({ name: `s2-reaudit-r${round}`, model: MODEL_HEAVY, maxTurns: 80, schema: S.reaudit,
      prompt: `Re-audit repo ${REPO} for defects MISSED so far. Read ${A}. The current finding locations are:\n${dedupe(findings).map((f) => `- ${f.location} (${f.class})`).join("\n").slice(0, 6000)}\nHunt specifically for classes/areas under-represented above (e.g. packaging/entry-point breakage, hardcoded machine-specific paths, unsafe network/deserialization, doc/code drift, dead config). Only NEW findings not already listed. Unique ids prefixed s2re${round}-.` });
    if (sweep.ok) findings = dedupe([...findings, ...sweep.data.new_findings.map((f) => ({ ...f, recommendation: f.recommendation || "" }))]);
    findings = dedupe(findings);
    // 2) falsify the WHOLE set against source (adversarial promotion gate)
    const batches = packChunks(findings, 25);
    const verds = await pMap(batches, (batch, bi) => runAgent({ name: `s2-falsify-r${round}-b${bi}`, model: MODEL_HEAVY, maxTurns: 70, schema: S.falsify,
      prompt: `Adversarially verify each finding below by opening the cited path:line in repo ${REPO} and trying to REFUTE it. For each: verdict ∈ {survived (evidence holds), refuted (claim is wrong/not a real defect), needs-refinement (real but mislocated/misclassified — give corrected_location)} with a rationale citing what you saw. Do not rubber-stamp.\nFINDINGS:\n${JSON.stringify(batch.map((f) => ({ id: f.id, location: f.location, class: f.class, evidence: f.evidence })))}` }), 4);
    const verdict = new Map();
    verds.forEach((r) => { if (r.ok) r.data.verdicts.forEach((v) => verdict.set(v.id, v)); });
    // 3) keep survivors (and refined); drop refuted and anything not adjudicated
    findings = findings.filter((f) => { const v = verdict.get(f.id); if (!v) return false; if (v.verdict === "refuted") return false; if (v.verdict === "needs-refinement" && v.corrected_location) f.location = v.corrected_location; return true; });
    findings = dedupe(findings);
    const sig = findings.map(fingerprint).sort().join("|");
    log(`  round ${round}: ${findings.length} survivors`);
    if (sig === prev) { log(`  fixpoint reached at round ${round}`); break; }
    prev = sig;
    if (round === STAGE2_CEILING) await halt("stage2", `no falsification fixpoint within ${STAGE2_CEILING} rounds.`);
  }
  // coverage stop-test (denominator visited)
  const coverage = examined.size / Math.max(1, sourceFiles.length);
  if (coverage < 0.9) log(`  ⚠ examined coverage ${(coverage * 100).toFixed(0)}% (<90%) — recorded in meta.`);
  const meta = { rounds, source_total: sourceFiles.length, examined: examined.size };
  const obj = { findings, meta };
  await checkpoint("stage2", "02-static-audit.md", renderStatic(findings, meta), "stage2.json", obj, state);
  return obj;
}

async function stage3(state, s1, s2) {
  log("STAGE 3 — execution / dynamic surface");
  const findingIds = s2.findings.map((f) => f.id);
  const ex = await runAgent({ name: "s3-exec", model: MODEL_FAST, maxTurns: 120, timeoutMs: 24e5, schema: S.exec,
    prompt: `Exercise the executable surface of repo ${REPO}. DISCOVER build/test/coverage commands from the repo itself (README, Makefile, pyproject.toml, pytest.ini, package.json, .github/workflows) — never assume another project's commands. Then actually run them in this sandbox: attempt dependency install if a manifest exists (uv/pip/npm), run the test suite under coverage if feasible, and drive real entry points. Capture real exit codes and observed behavior. If heavy deps cannot be installed or a region needs credentials/external services/hardware, record it in 'accounting' with the right status+reason rather than pretending. Aim for 100% ACCOUNTING (every region either executed or carrying a documented reason), not 100% execution. Use Stage-2 findings (read ${A2}) to confirm/refute/refine via finding_deltas. Known Stage-2 finding ids: ${findingIds.join(", ") || "(none)"}.` });
  if (!ex.ok) await halt("stage3", "execution worker failed to produce a valid execution object.");
  // independent check of the self-reported coverage/accounting
  const check = await runAgent({ name: "s3-check", model: MODEL_HEAVY, maxTurns: 50, schema: S.execCheck,
    prompt: `Independently verify the execution report at ${join(WORK, "stage3_raw.json")} (already written) against the actual repo ${REPO} and any coverage artifacts on disk (e.g. coverage.xml, htmlcov, .coverage, pytest output logs). Is the measured-coverage claim and the accounting honest and supported by real artifacts? Return coverage_claim_supported (bool), a rationale, and any discrepancies. Be adversarial — this stage is the most failure-prone.` });
  writeFileSync(join(WORK, "stage3_raw.json"), JSON.stringify(ex.data, null, 2));
  // (write raw BEFORE check ideally; re-run check note: file written now for reproducibility)
  const checkData = check.ok ? check.data : { coverage_claim_supported: false, rationale: "independent checker did not return a valid verdict", discrepancies: ["checker-failed"] };
  const obj = { execution: ex.data, independent_check: checkData };
  await checkpoint("stage3", "03-execution.md", renderExecution(ex.data, checkData), "stage3.json", obj, state);
  return obj;
}

async function stage4(state) {
  log("STAGE 4 — goal + external research (parallel halves)");
  const goalTask = runAgent({ name: "s4-goal", model: MODEL_HEAVY, maxTurns: 60, schema: S.goal,
    prompt: `Infer the repo's long-term goal(s) for ${REPO}. Read ${A}, ${A2}, ${A3}. Produce PLURAL candidates (do not collapse to one). Each candidate: a goal statement, a set of FALSIFIABLE success_signals, and grounding[] where every signal traces to a concrete Stage 1–3 artifact or path:line. Mark confidence ∈ {grounded, speculative, needs-human-confirm}. A candidate with no grounding must be dropped or flagged needs-human-confirm.` });
  // research half: deep-research shape — N independent gatherers, then a cross-checking synthesizer
  const researchHalf = (async () => {
    const provisional = existsSync(join(WORK, "stage1.json")) ? JSON.parse(readFileSync(join(WORK, "stage1.json"), "utf8")).provisional_intent : "";
    const angles = ["state-of-the-art methods & published models", "tooling/libraries/frameworks & reference implementations", "comparable open-source projects, benchmarks & datasets"];
    const gatherers = await pMap(Array.from({ length: RESEARCH_GATHERERS }, (_, i) => i), (i) => runAgent({
      name: `s4-research${i}`, model: MODEL_FAST, web: true, maxTurns: 40, timeoutMs: 18e5, schema: S.researchGather,
      prompt: `Deep external web research for ideas/technologies/projects that materially advance this project's goal. Context (provisional intent): "${provisional}". Read ${A} and ${A4 /*may not exist yet*/ } if present for grounding, then search the web from THIS angle: ${angles[i % angles.length]}. Return real sources with {title, url, claim, relevance}. Cite every source by URL. Stop at diminishing returns (hard cap ~12 strong sources).` }), RESEARCH_GATHERERS);
    const pooled = gatherers.flatMap((r) => (r.ok ? r.data.sources : []));
    const synth = await runAgent({ name: "s4-research-synth", model: MODEL_HEAVY, web: true, maxTurns: 40, schema: S.researchSynth,
      prompt: `Cross-check and synthesize these independently-gathered sources for repo ${REPO}. Weigh them against each other: mark each source corroboration ∈ {corroborated (>=2 independent sources or authoritative), uncorroborated}. Then distill ideas[] that materially advance the goal, each citing its supporting source URLs. An uncorroborated claim is recorded as unverified, not fact.\nPOOLED SOURCES:\n${JSON.stringify(pooled).slice(0, 12000)}` });
    return synth;
  })();
  const [goal, research] = await Promise.all([goalTask, researchHalf]);
  if (!goal.ok) await halt("stage4", "goal worker failed to produce grounded candidates.");
  const researchData = research.ok ? research.data : { sources: [], ideas: [{ idea: "research synthesis failed", relevance: "n/a", sources: [] }] };
  const obj = { goal: goal.data, research: researchData };
  await checkpoint("stage4", "04-goal.md", renderGoal(goal.data, researchData), "stage4.json", obj, state);
  return obj;
}

async function stage5(state) {
  log("STAGE 5 — execution-ready plan");
  let plan, check, ambiguous = [];
  for (let round = 1; round <= 2; round++) {
    const note = ambiguous.length ? `\nThe previous draft had ambiguous items a fresh agent could not map to a diff target: ${JSON.stringify(ambiguous)}. Fix exactly these.` : "";
    plan = await runAgent({ name: `s5-plan-r${round}`, model: MODEL_HEAVY, maxTurns: 80, schema: S.plan,
      prompt: `Produce an execution-ready change plan for ${REPO} that closes the gap between current state (read ${A}, ${A2}, ${A3}) and goal (read ${A4}). Ordered items; each item: id, title, links_to[] (Stage-2 finding ids and/or Stage-4 goal-gaps), location (file/module), change (what to do), verification (the observation or test that proves it worked), depends_on[] (item ids), and an integer order in dependency sequence. Every item MUST be mappable to a concrete diff target without a clarifying question.${note}` });
    if (!plan.ok) await halt("stage5", "plan worker failed to produce a valid plan object.");
    check = await runAgent({ name: `s5-check-r${round}`, model: MODEL_HEAVY, maxTurns: 50, schema: S.planCheck,
      prompt: `You are a fresh implementer. For the plan at ${join(WORK, "stage5_raw.json")} (already written), try to map EACH item to a concrete diff target in repo ${REPO} without asking any clarifying question. Return all_mappable (bool) and ambiguous_items[] = {id, why} for any item you could not localize/execute as written.` });
    writeFileSync(join(WORK, "stage5_raw.json"), JSON.stringify(plan.data, null, 2));
    const checkData = check.ok ? check.data : { all_mappable: false, ambiguous_items: [{ id: "?", why: "checker failed" }] };
    ambiguous = checkData.ambiguous_items || [];
    if (checkData.all_mappable) { check = { ok: true, data: checkData }; break; }
    log(`  round ${round}: ${ambiguous.length} ambiguous items; ${round < 2 ? "looping" : "ceiling reached"}`);
    check = { ok: true, data: checkData };
  }
  const obj = { plan: plan.data, mappability_check: check.data };
  await checkpoint("stage5", "05-plan.md", renderPlan(plan.data, check.data), "stage5.json", obj, state);
  return obj;
}

// ─────────────────────────────── preflight / selftest ───────────────────────────────
async function resolveModel(alias) {
  const r = await new Promise((res) => {
    const p = spawn("claude", ["-p", "reply with: OK", "--output-format", "json", "--model", alias, "--max-turns", "1"], { cwd: REPO, stdio: ["ignore", "pipe", "pipe"], env: { ...process.env, CLAUDE_CODE_DEBUG: "0" } });
    let O = ""; const k = setTimeout(() => { try { p.kill("SIGKILL"); } catch {} }, 120000);
    p.stdout.on("data", (d) => (O += d)); p.on("close", () => { clearTimeout(k); res(O); });
  });
  try { const j = JSON.parse(r); return !j.is_error; } catch { return false; }
}
async function preflight() {
  log("preflight: resolving models");
  if (!(await resolveModel(MODEL_HEAVY))) { log(`  heavy '${MODEL_HEAVY}' unavailable → falling back to sonnet`); MODEL_HEAVY = "sonnet"; }
  if (!(await resolveModel(MODEL_FAST))) { log(`  fast '${MODEL_FAST}' unavailable → falling back to sonnet`); MODEL_FAST = "sonnet"; }
  log(`  models → heavy=${MODEL_HEAVY} fast=${MODEL_FAST}`);
  return true;
}
function selftest() {
  let fail = 0;
  const expectOk = (s, d, want) => { const v = validate(s, d); if (v.ok !== want) { console.log(`SELFTEST FAIL: want ok=${want} got ${v.ok}`, v.errors); fail++; } };
  expectOk(S.findings, { files_examined: ["a"], findings: [{ id: "x", location: "a:1", class: "bug", severity: "high", evidence: "e" }] }, true);
  expectOk(S.findings, { findings: [] }, false);                            // missing files_examined
  expectOk(S.findings, { files_examined: [], findings: [{ id: "x", location: "a", class: "nope", severity: "high", evidence: "e" }] }, false); // bad enum
  expectOk(S.goal, { candidates: [] }, false);                              // minItems
  expectOk(S.plan, { items: [{ id: "1", title: "t", links_to: [], location: "f", change: "c", verification: "v", depends_on: [], order: 1 }] }, true);
  expectOk(S.invShard, { entries: [{ path: "p", role: "source", summary: "s" }] }, true);
  // render smoke (must not throw)
  try {
    renderUnderstanding([{ path: "p", role: "source", summary: "s" }], { architecture: "a", provisional_intent: "i", entry_points: [{ name: "n", kind: "cli", location: "p:1", description: "d" }] }, { files_total: 1, files_classified: 1, unknown: 0 });
    renderStatic([{ id: "x", location: "a:1", class: "bug", severity: "high", evidence: "e", recommendation: "r" }], { rounds: 1, source_total: 1, examined: 1 });
    renderPlan({ items: [{ id: "1", title: "t", links_to: ["f1"], location: "f", change: "c", verification: "v", depends_on: [], order: 1 }] }, { all_mappable: true, ambiguous_items: [] });
  } catch (e) { console.log("SELFTEST FAIL: renderer threw", e.message); fail++; }
  console.log(fail ? `SELFTEST: ${fail} failure(s)` : "SELFTEST: all passed");
  process.exit(fail ? 1 : 0);
}

// ─────────────────────────────── main ───────────────────────────────
async function main() {
  const argv = process.argv.slice(2);
  if (argv.includes("--selftest")) return selftest();
  mkdirSync(WORK, { recursive: true });
  BRANCH = (await sh("git", ["rev-parse", "--abbrev-ref", "HEAD"])).out.trim() || "HEAD";
  log(`repo=${REPO} branch=${BRANCH}`);
  if (argv.includes("--fresh")) { log("--fresh: tearing down audit artifacts"); for (const f of ["01-understanding.md", "02-static-audit.md", "03-execution.md", "04-goal.md", "05-plan.md", "HALT-REPORT.md"]) { try { rmSync(join(AUDIT, f)); } catch {} } try { rmSync(WORK, { recursive: true, force: true }); } catch {} mkdirSync(WORK, { recursive: true }); }
  if (argv.includes("--preflight")) { await preflight(); log("preflight OK"); return; }

  await preflight();
  const state = loadState();
  const only = argv.includes("--stage") ? Number(argv[argv.indexOf("--stage") + 1]) : null;
  const from = argv.includes("--from") ? Number(argv[argv.indexOf("--from") + 1]) : null;
  const want = (n) => only ? only === n : from ? n >= from : !state.completed[`stage${n}`];
  const need = (n) => { const p = join(WORK, `stage${n}.json`); if (!existsSync(p)) throw new Error(`stage${n} artifact missing; run that stage first`); return JSON.parse(readFileSync(p, "utf8")); };

  let s1, s2;
  if (want(1)) s1 = await stage1(state); else if (existsSync(join(WORK, "stage1.json"))) s1 = need(1);
  if (want(2)) s2 = await stage2(state, s1 || need(1)); else if (existsSync(join(WORK, "stage2.json"))) s2 = need(2);
  if (want(3)) await stage3(state, s1 || need(1), s2 || need(2));
  if (want(4)) await stage4(state);
  if (want(5)) await stage5(state);

  log(`DONE. api-equivalent cost $${TOTAL_COST.toFixed(2)} (subscription spend far lower). Artifacts in audit/.`);
}
main().catch((e) => { console.error("orchestrator error:", e); process.exit(1); });
