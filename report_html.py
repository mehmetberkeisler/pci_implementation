"""Static HTML report generator and command-line entry point.

This is the standalone CLI path: it runs :func:`analyze_pci.analyze_file`
(or :func:`analyze_pci.analyze_multiple_files`) and renders a self-contained
interactive HTML report. The Streamlit app (``app.py`` + ``ui/``) renders its
own views and does not import anything from this module.

    python report_html.py sub01.vhdr -o report.html
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

import numpy as np

from analyze_pci import analyze_file, analyze_multiple_files

logger = logging.getLogger("pcist_analyzer")


# ═══════════════════════════════════════════════════════════════════════════
# §6 HTML REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

def generate_html_report(analysis: Dict[str, Any], output_path: str):
    """Generate interactive HTML report with Plotly."""

    results = analysis.get("results", [analysis])

    # Collect all session data for the report
    all_sessions_json = []
    all_eeg_data = []

    for result in results:
        for sess in result["sessions"]:
            all_sessions_json.append(sess)

        # Limit EEG display channels to 20 evenly spaced for file size
        ch_names = result["ch_names"]
        eeg_data = result["eeg_display_data"]
        n_total_ch = len(ch_names)
        if n_total_ch > 20:
            step = n_total_ch // 20
            display_indices = list(range(0, n_total_ch, step))[:20]
        else:
            display_indices = list(range(n_total_ch))

        all_eeg_data.append({
            "file": result["file"],
            "ch_names": [ch_names[i] for i in display_indices],
            "all_ch_names": ch_names,
            "sfreq": result["sfreq"],
            "duration": result["duration"],
            "n_stim": result["n_stim_total"],
            "stim_times": [round(t, 3) for t in result["stim_times"]],
            "eeg_times": [round(t, 3) for t in result["eeg_display_times"]],
            "eeg_data": [eeg_data[i] for i in display_indices],
            "ds_factor": result["ds_factor"],
            "all_marker_times": [round(t, 3) for t in result["all_marker_times"]],
            "all_marker_types": result["all_marker_types"],
            "display_indices": display_indices,
        })

    # Serialize session data (without heavy EEG data)
    sessions_for_json = []
    for s in all_sessions_json:
        entry = {k: v for k, v in s.items() if k not in ("evoked_data",)}
        # evoked_data can be large; we keep it for per-session evoked plots
        if "evoked_data" in s and s["evoked_data"] is not None:
            # Keep all channels but round values
            entry["evoked_data"] = [[round(v, 4) for v in ch] for ch in s["evoked_data"]]
        sessions_for_json.append(entry)

    html = _build_html(sessions_for_json, all_eeg_data)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"Report saved to {output_path}")


def _build_html(sessions: List[Dict], eeg_data_list: List[Dict]) -> str:
    """Build the complete HTML string."""

    sessions_json = json.dumps(sessions, default=_json_default)
    eeg_json = json.dumps(eeg_data_list, default=_json_default)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TMS-EEG PCI Analysis Report</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
:root {{
    --ink-900: #102231;
    --ink-700: #264255;
    --accent: #0b6e8a;
    --accent-light: #1483a5;
    --warm: #c86b33;
    --bg: #f5f8fb;
    --panel: #ffffff;
    --line: #d6e2e8;
    --green: #2E7D32;
    --orange: #E65100;
    --grey: #37474F;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--ink-900);
    line-height: 1.5;
}}
.container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
.header {{
    background: linear-gradient(135deg, #0b6e8a 0%, #1a3a4a 100%);
    color: white;
    padding: 30px 40px;
    border-radius: 16px;
    margin-bottom: 24px;
    box-shadow: 0 8px 24px rgba(11, 110, 138, 0.3);
}}
.header h1 {{ font-size: 1.8rem; font-weight: 700; margin-bottom: 4px; }}
.header .subtitle {{ font-size: 0.95rem; opacity: 0.85; }}
.card {{
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}}
.card h2 {{
    font-size: 1.15rem;
    color: var(--ink-900);
    border-bottom: 1px solid var(--line);
    padding-bottom: 8px;
    margin-bottom: 16px;
}}
.metrics-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px;
    margin-bottom: 16px;
}}
.metric {{
    background: linear-gradient(180deg, #f8fbfd 0%, #f0f5f8 100%);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 12px 16px;
    text-align: center;
}}
.metric-value {{
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--ink-900);
}}
.metric-label {{
    font-size: 0.75rem;
    color: #5b7282;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}
.pci-badge {{
    display: inline-block;
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
}}
.pci-conscious {{ background: #E8F5E9; color: #2E7D32; border: 1px solid #2E7D32; }}
.pci-intermediate {{ background: #FFF3E0; color: #E65100; border: 1px solid #E65100; }}
.pci-unconscious {{ background: #ECEFF1; color: #37474F; border: 1px solid #37474F; }}
.sessions-tabs {{
    display: flex;
    gap: 8px;
    margin-bottom: 16px;
    flex-wrap: wrap;
}}
.session-tab {{
    padding: 8px 20px;
    border: 2px solid var(--line);
    border-radius: 8px;
    cursor: pointer;
    font-weight: 600;
    font-size: 0.9rem;
    transition: all 0.2s;
    background: white;
}}
.session-tab:hover {{ border-color: var(--accent); }}
.session-tab.active {{
    background: var(--accent);
    color: white;
    border-color: var(--accent);
}}
.session-content {{ display: none; }}
.session-content.active {{ display: block; }}
.plot-container {{ width: 100%; min-height: 300px; }}
.summary-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
}}
.summary-table th {{
    background: #eaf2f6;
    color: #183448;
    font-weight: 650;
    padding: 8px 12px;
    text-align: left;
    border-bottom: 2px solid #c7d9e2;
}}
.summary-table td {{
    padding: 8px 12px;
    border-bottom: 1px solid #e5eef3;
}}
.summary-table tr:hover {{ background: #f8fbfd; }}
.eeg-controls {{
    display: flex;
    gap: 12px;
    align-items: center;
    margin-bottom: 8px;
    flex-wrap: wrap;
}}
.eeg-controls label {{ font-size: 0.85rem; font-weight: 600; }}
.eeg-controls select, .eeg-controls input {{
    padding: 4px 8px;
    border: 1px solid var(--line);
    border-radius: 6px;
    font-size: 0.85rem;
}}
</style>
</head>
<body>
<div class="container">

<div class="header">
    <h1>Perturbational Complexity Index - TMS-EEG Analysis</h1>
    <div class="subtitle">Casali et al. (2013) framework | Bootstrap significance + Lempel-Ziv complexity</div>
</div>

<!-- Summary Card -->
<div class="card" id="summary-card"></div>

<!-- Full EEG Display -->
<div class="card">
    <h2>Full EEG Recording with Stimulus Markers</h2>
    <div class="eeg-controls">
        <label>Channels:</label>
        <select id="ch-select" onchange="updateEEGPlot()">
            <option value="all">All channels</option>
            <option value="subset">Key channels (Fz, Cz, Pz, C3, C4)</option>
        </select>
        <label>Scale (µV):</label>
        <input type="range" id="scale-slider" min="10" max="500" value="100" oninput="updateEEGPlot()">
        <span id="scale-value">100</span>
    </div>
    <div id="eeg-full-plot" class="plot-container" style="min-height:500px;"></div>
</div>

<!-- Session Tabs -->
<div class="card">
    <h2>Session-Based Analysis</h2>
    <div class="sessions-tabs" id="session-tabs"></div>
    <div id="session-panels"></div>
</div>

<!-- Comparison Table -->
<div class="card">
    <h2>Session Comparison</h2>
    <div id="comparison-table"></div>
</div>

<!-- Interpretation -->
<div class="card">
    <h2>Interpretation Notes</h2>
    <table class="summary-table">
        <tr><th>Item</th><th>Meaning</th><th>Use</th></tr>
        <tr><td>PCIst</td><td>Sensor-level normalized state-transition sum</td><td>Compare sessions processed with identical settings and QC gates</td></tr>
        <tr><td>Scale</td><td>Not bounded to 0-1</td><td>Do not apply Casali LZ-PCI thresholds directly</td></tr>
        <tr><td>QC</td><td>SNR, rejected epochs, bad channels, and trigger timing</td><td>Interpret before ranking sessions</td></tr>
    </table>
</div>

</div>

<script>
const sessions = {sessions_json};
const eegDataList = {eeg_json};

// ── INITIALIZATION ──
function init() {{
    buildSummary();
    buildEEGPlot();
    buildSessionTabs();
    buildComparisonTable();
}}

// ── SUMMARY ──
function buildSummary() {{
    const card = document.getElementById('summary-card');
    let totalStim = 0;
    eegDataList.forEach(d => totalStim += d.n_stim);

    const validSessions = sessions.filter(s => s.pcist !== null && s.pcist !== undefined);
    const avgPCIst = validSessions.length > 0 ? (validSessions.reduce((a, s) => a + s.pcist, 0) / validSessions.length) : 0;

    let html = '<h2>Recording Overview</h2><div class="metrics-grid">';
    html += `<div class="metric"><div class="metric-value">${{eegDataList.length}}</div><div class="metric-label">File(s)</div></div>`;
    html += `<div class="metric"><div class="metric-value">${{eegDataList[0]?.ch_names?.length || 0}}</div><div class="metric-label">Channels</div></div>`;
    html += `<div class="metric"><div class="metric-value">${{(eegDataList[0]?.sfreq || 0).toFixed(0)}} Hz</div><div class="metric-label">Sampling Rate</div></div>`;
    html += `<div class="metric"><div class="metric-value">${{totalStim}}</div><div class="metric-label">Total Stimuli</div></div>`;
    html += `<div class="metric"><div class="metric-value">${{sessions.length}}</div><div class="metric-label">Sessions</div></div>`;
    if (validSessions.length > 0) {{
        html += `<div class="metric"><div class="metric-value">${{avgPCIst.toFixed(1)}}</div><div class="metric-label">Mean PCIst</div></div>`;
    }}
    html += '</div>';
    card.innerHTML = html;
}}

// ── FULL EEG PLOT ──
function buildEEGPlot() {{
    const eeg = eegDataList[0];
    if (!eeg) return;
    updateEEGPlot();
}}

function updateEEGPlot() {{
    const eeg = eegDataList[0];
    if (!eeg) return;

    const chSelect = document.getElementById('ch-select').value;
    const scale = parseInt(document.getElementById('scale-slider').value);
    document.getElementById('scale-value').textContent = scale;

    let channelIndices = [];
    const keyChannels = ['Fz', 'Cz', 'Pz', 'C3', 'C4', 'F3', 'F4', 'P3', 'P4', 'O1', 'O2'];

    if (chSelect === 'subset') {{
        keyChannels.forEach(name => {{
            const idx = eeg.ch_names.indexOf(name);
            if (idx >= 0) channelIndices.push(idx);
        }});
        if (channelIndices.length === 0) channelIndices = Array.from({{length: Math.min(10, eeg.ch_names.length)}}, (_, i) => i);
    }} else {{
        channelIndices = Array.from({{length: eeg.ch_names.length}}, (_, i) => i);
    }}

    const traces = [];
    const n_ch = channelIndices.length;

    // EEG traces with offset stacking
    channelIndices.forEach((chIdx, i) => {{
        const offset = (n_ch - 1 - i) * scale;
        const y = eeg.eeg_data[chIdx].map(v => v + offset);
        traces.push({{
            x: eeg.eeg_times,
            y: y,
            type: 'scattergl',
            mode: 'lines',
            line: {{ width: 0.5, color: '#333' }},
            name: eeg.ch_names[chIdx],
            hovertemplate: eeg.ch_names[chIdx] + ': %{{customdata:.1f}} µV<extra></extra>',
            customdata: eeg.eeg_data[chIdx],
            showlegend: false,
        }});
    }});

    // Stimulus markers as vertical lines
    const shapes = [];
    const annotations = [];

    // Session boundaries
    sessions.forEach((s, i) => {{
        if (s.start_time !== undefined) {{
            shapes.push({{
                type: 'rect',
                x0: s.start_time,
                x1: s.end_time,
                y0: -scale,
                y1: (n_ch) * scale,
                fillcolor: `rgba(11, 110, 138, 0.06)`,
                line: {{ color: 'rgba(11, 110, 138, 0.3)', width: 1, dash: 'dash' }},
            }});
            annotations.push({{
                x: (s.start_time + s.end_time) / 2,
                y: (n_ch) * scale + scale * 0.3,
                text: `<b>${{s.label}}</b>` + (s.pcist !== null && s.pcist !== undefined ? `<br>PCIst=${{s.pcist.toFixed(1)}}` : ''),
                showarrow: false,
                font: {{ size: 11, color: '#0b6e8a' }},
                bgcolor: 'rgba(255,255,255,0.8)',
                bordercolor: '#0b6e8a',
                borderwidth: 1,
                borderpad: 4,
            }});
        }}
    }});

    // Stimulus ticks
    eeg.stim_times.forEach(t => {{
        shapes.push({{
            type: 'line',
            x0: t, x1: t,
            y0: -scale * 0.5,
            y1: (n_ch) * scale,
            line: {{ color: 'rgba(198, 40, 40, 0.3)', width: 0.8 }},
        }});
    }});

    // Y-axis tick labels for channels
    const tickvals = channelIndices.map((_, i) => (n_ch - 1 - i) * scale);
    const ticktext = channelIndices.map(i => eeg.ch_names[i]);

    const layout = {{
        height: Math.max(400, n_ch * 18 + 80),
        margin: {{ l: 70, r: 30, t: 30, b: 50 }},
        xaxis: {{
            title: 'Time (s)',
            rangeslider: {{ visible: true, thickness: 0.06 }},
            showgrid: true,
            gridcolor: '#eee',
        }},
        yaxis: {{
            tickvals: tickvals,
            ticktext: ticktext,
            tickfont: {{ size: 8 }},
            showgrid: false,
            zeroline: false,
        }},
        shapes: shapes,
        annotations: annotations,
        dragmode: 'zoom',
        hovermode: 'x unified',
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
    }};

    Plotly.newPlot('eeg-full-plot', traces, layout, {{
        responsive: true,
        scrollZoom: true,
        displayModeBar: true,
        modeBarButtonsToAdd: ['drawrect', 'eraseshape'],
    }});
}}

// ── SESSION TABS ──
function buildSessionTabs() {{
    const tabsDiv = document.getElementById('session-tabs');
    const panelsDiv = document.getElementById('session-panels');

    sessions.forEach((s, i) => {{
        // Tab
        const tab = document.createElement('div');
        tab.className = 'session-tab' + (i === 0 ? ' active' : '');
        tab.textContent = s.label;
        tab.onclick = () => switchSession(i);
        tabsDiv.appendChild(tab);

        // Panel
        const panel = document.createElement('div');
        panel.className = 'session-content' + (i === 0 ? ' active' : '');
        panel.id = `session-panel-${{i}}`;
        panel.innerHTML = buildSessionPanel(s, i);
        panelsDiv.appendChild(panel);
    }});

    // Render plots for first session
    setTimeout(() => renderSessionPlots(0), 100);
}}

function switchSession(idx) {{
    document.querySelectorAll('.session-tab').forEach((t, i) => {{
        t.className = 'session-tab' + (i === idx ? ' active' : '');
    }});
    document.querySelectorAll('.session-content').forEach((p, i) => {{
        p.className = 'session-content' + (i === idx ? ' active' : '');
    }});
    renderSessionPlots(idx);
}}

function buildSessionPanel(s, idx) {{
    if (s.error && s.pcist == null) {{
        return `<div class="metrics-grid">
            <div class="metric"><div class="metric-value">${{s.n_events}}</div><div class="metric-label">Stimuli</div></div>
            <div class="metric"><div class="metric-value">${{s.n_accepted || 0}}</div><div class="metric-label">Epochs</div></div>
            <div class="metric"><div class="metric-value" style="color:red;">ERROR</div><div class="metric-label">${{s.error}}</div></div>
        </div>`;
    }}

    const nComp = s.n_components || 0;
    const dNST = s.dNST || [];
    const dNSTstr = dNST.length > 0 ? dNST.map(d => d.toFixed(1)).join(', ') : 'N/A';

    return `
    <div class="metrics-grid">
        <div class="metric">
            <div class="metric-value" style="font-size:2rem;">${{s.pcist != null ? s.pcist.toFixed(1) : 'N/A'}}</div>
            <div class="metric-label">PCIst</div>
        </div>
        <div class="metric"><div class="metric-value">${{nComp}}</div><div class="metric-label">SVD Components</div></div>
        <div class="metric"><div class="metric-value">${{s.snr?.toFixed(2) || 'N/A'}}</div><div class="metric-label">SNR</div></div>
        <div class="metric"><div class="metric-value">${{s.n_accepted || 0}} / ${{s.n_events}}</div><div class="metric-label">Epochs</div></div>
        <div class="metric"><div class="metric-value">${{s.n_channels_used || '?'}}</div><div class="metric-label">Channels</div></div>
        <div class="metric"><div class="metric-value">${{s.duration?.toFixed(1) || '?'}}s</div><div class="metric-label">Duration</div></div>
    </div>
    <div style="margin:8px 0; font-size:0.85em; color:#666;">
        <strong>ΔNST per component:</strong> [${{dNSTstr}}]
    </div>
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px;">
        <div id="evoked-plot-${{idx}}" class="plot-container" style="min-height:280px;"></div>
        <div id="ss-plot-${{idx}}" class="plot-container" style="min-height:280px;"></div>
    </div>
    `;
}}

function renderSessionPlots(idx) {{
    const s = sessions[idx];
    if (!s || s.pcist == null) return;

    // Check if already rendered
    const evokedDiv = document.getElementById(`evoked-plot-${{idx}}`);
    if (!evokedDiv || evokedDiv.dataset.rendered === 'true') return;
    evokedDiv.dataset.rendered = 'true';

    // Evoked butterfly + GFP
    if (s.evoked_data && s.evoked_times) {{
        const traces = [];
        const n_ch = s.evoked_data.length;
        const alpha = Math.max(0.08, Math.min(0.5, 8 / n_ch));

        // Individual channels
        for (let i = 0; i < n_ch; i++) {{
            traces.push({{
                x: s.evoked_times,
                y: s.evoked_data[i],
                type: 'scatter',
                mode: 'lines',
                line: {{ width: 0.4, color: `rgba(60,60,60,${{alpha}})` }},
                showlegend: false,
                hoverinfo: 'skip',
            }});
        }}

        // GFP
        traces.push({{
            x: s.evoked_times,
            y: s.evoked_gfp,
            type: 'scatter',
            mode: 'lines',
            line: {{ width: 2, color: '#000' }},
            name: 'GFP',
        }});

        Plotly.newPlot(`evoked-plot-${{idx}}`, traces, {{
            title: {{ text: 'TMS-Evoked Potentials', font: {{ size: 13 }} }},
            xaxis: {{ title: 'Time (ms)', zeroline: true }},
            yaxis: {{ title: 'Amplitude (µV)' }},
            shapes: [{{ type: 'line', x0: 0, x1: 0, y0: 0, y1: 1, yref: 'paper', line: {{ color: '#C62828', dash: 'dash', width: 1.5 }} }}],
            margin: {{ l: 50, r: 20, t: 40, b: 40 }},
            height: 280,
            plot_bgcolor: 'white',
            showlegend: true,
            legend: {{ x: 0.85, y: 0.95, font: {{ size: 9 }} }},
        }}, {{ responsive: true }});
    }}

    // ΔNST component bar chart (replaces SS matrix heatmap for PCIst)
    const ssDiv = document.getElementById(`ss-plot-${{idx}}`);
    if (ssDiv && s.dNST && s.dNST.length > 0) {{
        const compLabels = s.dNST.map((_, i) => `Comp ${{(s.components_kept || [])[i] || i + 1}}`);
        Plotly.newPlot(`ss-plot-${{idx}}`, [{{
            x: compLabels,
            y: s.dNST,
            type: 'bar',
            marker: {{ color: '#0b6e8a' }},
            text: s.dNST.map(d => d.toFixed(1)),
            textposition: 'outside',
        }}], {{
            title: {{ text: 'ΔNST per SVD Component', font: {{ size: 13 }} }},
            xaxis: {{ title: 'Component' }},
            yaxis: {{ title: 'ΔNST (state transitions)', rangemode: 'tozero' }},
            margin: {{ l: 50, r: 20, t: 40, b: 40 }},
            height: 280,
            plot_bgcolor: 'white',
        }}, {{ responsive: true }});
    }} else if (ssDiv) {{
        ssDiv.innerHTML = '<p style="text-align:center;color:#999;padding-top:80px;">No components survived SNR filter</p>';
    }}
}}

// ── COMPARISON TABLE ──
function buildComparisonTable() {{
    const div = document.getElementById('comparison-table');
    const validSessions = sessions.filter(s => s.pcist != null);

    if (validSessions.length === 0) {{
        div.innerHTML = '<p>No valid PCIst results to compare.</p>';
        return;
    }}

    let html = '<table class="summary-table"><tr>';
    html += '<th>Session</th><th>PCIst</th><th>SVD Comp.</th><th>SNR</th><th>Epochs</th><th>Channels</th><th>Duration</th><th>ISI (s)</th>';
    html += '</tr>';

    validSessions.forEach(s => {{
        html += `<tr>
            <td><strong>${{s.label}}</strong></td>
            <td><strong>${{s.pcist.toFixed(1)}}</strong></td>
            <td>${{s.n_components || 0}}</td>
            <td>${{s.snr?.toFixed(2) || '-'}}</td>
            <td>${{s.n_accepted}}/${{s.n_events}}</td>
            <td>${{s.n_channels_used || '-'}}</td>
            <td>${{s.duration?.toFixed(1) || '-'}}s</td>
            <td>${{s.median_isi?.toFixed(2) || '-'}}</td>
        </tr>`;
    }});

    html += '</table>';
    div.innerHTML = html;
}}

// ── START ──
document.addEventListener('DOMContentLoaded', init);
</script>
</body>
</html>"""


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    return str(obj)


# ═══════════════════════════════════════════════════════════════════════════
# §7 CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Command-line entry point: analyse file(s) and write an HTML report."""
    import argparse

    parser = argparse.ArgumentParser(description="TMS-EEG PCIst Analyzer (Comolatti 2019)")
    parser.add_argument("vhdr_files", nargs="+", help="BrainVision .vhdr file(s)")
    parser.add_argument("-o", "--output", default="pcist_report.html", help="Output HTML file")
    parser.add_argument("--gap", type=float, default=30.0, help="Gap threshold for session detection (seconds)")
    parser.add_argument("--reject", type=float, default=150.0, help="Artifact rejection threshold (µV)")
    parser.add_argument("--decimate-to", type=float, default=1000.0, help="Target sampling rate after decimation (Hz)")
    parser.add_argument("--pcist-k", type=float, default=1.2, help="PCIst baseline penalty factor")
    parser.add_argument("--pcist-min-snr", type=float, default=1.1, help="PCIst min component SNR")
    parser.add_argument("--pcist-max-var", type=float, default=99.0, help="PCIst max cumulative variance (percent)")
    parser.add_argument("--pcist-n-steps", type=int, default=100, help="PCIst number of threshold steps")

    args = parser.parse_args()

    kwargs = dict(
        gap_seconds=args.gap,
        reject_uv=args.reject,
        decimate_to=args.decimate_to,
        pcist_k=args.pcist_k,
        pcist_min_snr=args.pcist_min_snr,
        pcist_max_var=args.pcist_max_var,
        pcist_n_steps=args.pcist_n_steps,
    )

    if len(args.vhdr_files) == 1:
        result = analyze_file(args.vhdr_files[0], **kwargs)
        analysis = {"results": [result]}
    else:
        analysis = analyze_multiple_files(args.vhdr_files, **kwargs)

    generate_html_report(analysis, args.output)
    print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
