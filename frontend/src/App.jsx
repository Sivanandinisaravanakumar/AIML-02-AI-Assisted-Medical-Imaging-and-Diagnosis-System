import { useEffect, useRef, useState } from "react";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function api(path, options) {
  const res = await fetch(`${API}${path}`, options);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch { /* ignore */ }
    throw new Error(detail);
  }
  return res.json();
}

const pct = (v) => `${(v * 100).toFixed(1)}%`;

export default function App() {
  const [cases, setCases] = useState([]);
  const [current, setCurrent] = useState(null);
  const [health, setHealth] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [opacity, setOpacity] = useState(0.8);
  const [drag, setDrag] = useState(false);
  const fileRef = useRef(null);

  const refresh = () => api("/cases").then(setCases).catch(() => {});

  useEffect(() => {
    api("/health").then(setHealth).catch(() => setError(`Cannot reach the API at ${API}. Start the backend first.`));
    refresh();
  }, []);

  async function upload(file) {
    if (!file) return;
    setBusy(true);
    setError("");
    try {
      const body = new FormData();
      body.append("file", file);
      const c = await api("/predict", { method: "POST", body });
      setCurrent(c);
      refresh();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="app">
      <header className="top">
        <h1>Scan review workstation</h1>
        <div className="status">
          {health && (
            <span className={health.model_loaded ? "pill ok" : "pill warn"}>
              {health.model_loaded ? "Trained model loaded" : "Untrained demo model"}
            </span>
          )}
          {health?.orthanc && <span className="pill ok">Orthanc connected</span>}
        </div>
      </header>

      <div className="layout">
        <aside className="side">
          <div
            className={`drop ${drag ? "over" : ""}`}
            onClick={() => fileRef.current.click()}
            onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => { e.preventDefault(); setDrag(false); upload(e.dataTransfer.files[0]); }}
          >
            <strong>{busy ? "Analysing scan…" : "Add a scan"}</strong>
            <span>Drop a chest X-ray here, or click to browse. PNG, JPG and DICOM (.dcm) work.</span>
            <input ref={fileRef} type="file" accept=".png,.jpg,.jpeg,.dcm,.dicom" hidden
                   onChange={(e) => { upload(e.target.files[0]); e.target.value = ""; }} />
          </div>

          <h2>Case queue</h2>
          {cases.length === 0 && <p className="muted">No cases yet. Add a scan to start.</p>}
          <ul className="queue">
            {cases.map((c) => (
              <li key={c.id}>
                <button className={current?.id === c.id ? "row active" : "row"} onClick={() => setCurrent(c)}>
                  <span className="fname">{c.filename}</span>
                  <span className="meta">
                    {c.label} · {pct(c.confidence)}
                  </span>
                  <span className={c.status === "reviewed" ? "chip done" : "chip wait"}>
                    {c.status === "reviewed" ? "Reviewed" : "Needs review"}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </aside>

        <main className="main">
          {error && <div className="error" role="alert">{error}</div>}
          {!current ? (
            <div className="empty">
              <h2>No scan open</h2>
              <p>Add a scan to see the model's finding and the regions it focused on, then record your review.</p>
            </div>
          ) : (
            <CaseView key={current.id} c={current} opacity={opacity} setOpacity={setOpacity}
                      onSaved={(u) => { setCurrent(u); refresh(); }} />
          )}
        </main>
      </div>
      <footer>Research prototype. Not for clinical use.</footer>
    </div>
  );
}

function CaseView({ c, opacity, setOpacity, onSaved }) {
  const [decision, setDecision] = useState(c.decision || "agree");
  const [finalLabel, setFinalLabel] = useState(c.final_label || c.label);
  const [notes, setNotes] = useState(c.notes || "");
  const [reviewer, setReviewer] = useState(c.reviewer || "");
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState("");
  const reviewed = c.status === "reviewed";

  async function save(e) {
    e.preventDefault();
    setSaving(true);
    setErr("");
    try {
      const label = decision === "agree" ? c.label : finalLabel;
      const u = await api(`/cases/${c.id}/review`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ decision, final_label: label, notes, reviewer: reviewer || "Doctor" }),
      });
      onSaved(u);
    } catch (ex) {
      setErr(ex.message);
    } finally {
      setSaving(false);
    }
  }

  return (
    <>
      <div className="case-head">
        <h2>{c.filename}</h2>
        <span className={reviewed ? "chip done" : "chip wait"}>{reviewed ? "Reviewed" : "Needs review"}</span>
      </div>

      {!c.model_trained && (
        <div className="notice">
          This result comes from an untrained model, so the label and heatmap are not meaningful.
          Train the model (see README) and restart the backend.
        </div>
      )}

      <div className="viewers">
        <figure>
          <div className="stage"><img src={`${API}/cases/${c.id}/image`} alt="Original scan" /></div>
          <figcaption>Original scan</figcaption>
        </figure>
        <figure>
          <div className="stage">
            <img src={`${API}/cases/${c.id}/image`} alt="" />
            <img className="overlay" style={{ opacity }} src={`${API}/cases/${c.id}/heatmap`}
                 alt="Grad-CAM heatmap of the regions that drove the prediction" />
          </div>
          <figcaption>
            <label className="slider">
              Heatmap strength
              <input type="range" min="0" max="1" step="0.05" value={opacity}
                     onChange={(e) => setOpacity(Number(e.target.value))} />
            </label>
          </figcaption>
        </figure>
      </div>

      <section className="card">
        <h3>Model finding</h3>
        <p className="finding"><strong>{c.label}</strong> <span className="muted">at {pct(c.confidence)} confidence</span></p>
        <div className="bars">
          <Bar name="Normal" v={c.prob_normal} />
          <Bar name="Pneumonia" v={c.prob_pneumonia} />
        </div>
      </section>

      <form className="card" onSubmit={save}>
        <h3>Doctor review</h3>
        <div className="radios">
          <label><input type="radio" checked={decision === "agree"} onChange={() => setDecision("agree")} />
            Agree with the model ({c.label})</label>
          <label><input type="radio" checked={decision === "override"} onChange={() => setDecision("override")} />
            Override the model</label>
        </div>
        {decision === "override" && (
          <label className="field">Final diagnosis
            <select value={finalLabel} onChange={(e) => setFinalLabel(e.target.value)}>
              <option value="NORMAL">NORMAL</option>
              <option value="PNEUMONIA">PNEUMONIA</option>
            </select>
          </label>
        )}
        <label className="field">Reviewer name
          <input value={reviewer} onChange={(e) => setReviewer(e.target.value)} placeholder="Dr. Name" />
        </label>
        <label className="field">Notes
          <textarea rows="3" value={notes} onChange={(e) => setNotes(e.target.value)}
                    placeholder="Findings, follow-up, reason for override" />
        </label>
        {err && <div className="error">{err}</div>}
        <button className="primary" disabled={saving}>{saving ? "Saving…" : reviewed ? "Update review" : "Save review"}</button>
        {reviewed && <p className="muted small">Last reviewed by {c.reviewer} on {new Date(c.reviewed_at).toLocaleString()}.</p>}
      </form>
    </>
  );
}

function Bar({ name, v }) {
  return (
    <div className="bar">
      <span>{name}</span>
      <div className="track"><div className="fill" style={{ width: pct(v) }} /></div>
      <span className="num">{pct(v)}</span>
    </div>
  );
}
