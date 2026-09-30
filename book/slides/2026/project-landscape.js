(() => {
  const storageKey = "mlgeo-2026-project-landscape";
  const taskOrder = [
    "Detect & classify",
    "Discover structure",
    "Forecast",
    "Emulate & invert",
  ];

  const palettes = {
    discipline: {
      "Earthquakes & volcanoes": "#b3402a",
      "Climate & weather": "#2f6fb0",
      "Cryosphere & remote sensing": "#4d8f95",
      "Hydrology & hazards": "#327a5f",
      "Solid Earth & geodesy": "#8a5a44",
      "Oceans & ecosystems": "#6558a6",
    },
    modality: {
      "Time series": "#116b66",
      Waveforms: "#b3402a",
      "Images & rasters": "#6a56a1",
      "Gridded fields": "#2f6fb0",
      "Tables & points": "#b17718",
      "Profiles & trajectories": "#327a5f",
    },
    task: {
      "Detect & classify": "#b3402a",
      "Discover structure": "#6a56a1",
      Forecast: "#2f6fb0",
      "Emulate & invert": "#116b66",
    },
  };

  const examples = [
    { idea: "Detect small earthquakes hidden in continuous waveforms", discipline: "Earthquakes & volcanoes", modality: "Waveforms", task: "Detect & classify" },
    { idea: "Discover ocean regimes from float profiles", discipline: "Oceans & ecosystems", modality: "Profiles & trajectories", task: "Discover structure" },
    { idea: "Forecast streamflow during atmospheric rivers", discipline: "Hydrology & hazards", modality: "Time series", task: "Forecast" },
    { idea: "Emulate a regional climate simulation", discipline: "Climate & weather", modality: "Gridded fields", task: "Emulate & invert" },
    { idea: "Map glacier retreat from satellite scenes", discipline: "Cryosphere & remote sensing", modality: "Images & rasters", task: "Detect & classify" },
    { idea: "Find transient deformation in GNSS stations", discipline: "Solid Earth & geodesy", modality: "Tables & points", task: "Discover structure" },
  ];

  let projects = [];
  let theme = "discipline";

  function safeParseStored() {
    try {
      const stored = JSON.parse(localStorage.getItem(storageKey) || "[]");
      return Array.isArray(stored) ? stored : [];
    } catch (_) {
      return [];
    }
  }

  function saveLocal() {
    localStorage.setItem(storageKey, JSON.stringify(projects));
  }

  function escapeHtml(value) {
    return String(value || "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function normalizeProject(row) {
    const lower = Object.fromEntries(Object.entries(row).map(([key, value]) => [key.trim().toLowerCase(), String(value || "").trim()]));
    return {
      idea: lower.idea || lower.question || lower["project idea"] || "",
      discipline: lower.discipline || lower.domain || "",
      modality: lower.modality || lower["data modality"] || "",
      task: lower.task || lower["ml task"] || "",
    };
  }

  function parseCsv(text) {
    const rows = [];
    let row = [];
    let field = "";
    let quoted = false;
    for (let i = 0; i < text.length; i += 1) {
      const char = text[i];
      const next = text[i + 1];
      if (char === '"' && quoted && next === '"') {
        field += '"';
        i += 1;
      } else if (char === '"') {
        quoted = !quoted;
      } else if (char === "," && !quoted) {
        row.push(field);
        field = "";
      } else if ((char === "\n" || char === "\r") && !quoted) {
        if (char === "\r" && next === "\n") i += 1;
        row.push(field);
        if (row.some((cell) => cell.trim())) rows.push(row);
        row = [];
        field = "";
      } else {
        field += char;
      }
    }
    row.push(field);
    if (row.some((cell) => cell.trim())) rows.push(row);
    if (rows.length < 2) return [];
    const headers = rows[0];
    return rows.slice(1).map((values) => normalizeProject(Object.fromEntries(headers.map((header, index) => [header, values[index] || ""]))));
  }

  function render() {
    const board = document.getElementById("landscape-board");
    const legend = document.getElementById("landscape-legend");
    const count = document.getElementById("project-count");
    if (count) count.textContent = `${projects.length} ${projects.length === 1 ? "idea" : "ideas"}`;
    if (!board || !legend) return;

    board.innerHTML = taskOrder.map((task) => {
      const laneProjects = projects.filter((project) => project.task === task);
      const cards = laneProjects.length
        ? laneProjects.map((project) => {
          const colorKey = project[theme];
          const color = palettes[theme][colorKey] || "#6e675c";
          return `<article class="project-card" style="--project-color:${color}"><strong>${escapeHtml(project.idea)}</strong><small>${escapeHtml(project.discipline)} · ${escapeHtml(project.modality)}</small></article>`;
        }).join("")
        : '<span class="empty-lane">Waiting for an idea</span>';
      return `<section class="task-lane"><h3>${escapeHtml(task)}</h3><div class="task-lane-projects">${cards}</div></section>`;
    }).join("");

    legend.innerHTML = Object.entries(palettes[theme]).map(([label, color]) =>
      `<span class="legend-item"><span class="legend-swatch" style="--legend-color:${color}"></span>${escapeHtml(label)}</span>`
    ).join("");
  }

  async function refreshShared(csvUrl) {
    const status = document.getElementById("landscape-status");
    try {
      const response = await fetch(csvUrl, { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const remoteProjects = parseCsv(await response.text()).filter((project) => project.idea && taskOrder.includes(project.task));
      projects = remoteProjects;
      if (status) status.textContent = `live · ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
      render();
    } catch (_) {
      if (status) status.textContent = "live source unavailable";
    }
  }

  function initialize() {
    const landscape = document.getElementById("project-landscape");
    const addButton = document.getElementById("project-add");
    const demoButton = document.getElementById("project-demo");
    const clearButton = document.getElementById("project-clear");
    if (!landscape) return;

    projects = safeParseStored();
    render();

    document.querySelectorAll(".theme-button").forEach((button) => {
      button.addEventListener("click", () => {
        theme = button.dataset.theme || "discipline";
        document.querySelectorAll(".theme-button").forEach((candidate) => candidate.classList.toggle("active", candidate === button));
        render();
      });
    });

    addButton?.addEventListener("click", () => {
      const ideaInput = document.getElementById("project-idea");
      const idea = ideaInput?.value.trim() || "";
      if (!idea) {
        ideaInput?.focus();
        return;
      }
      projects.push({
        idea,
        discipline: document.getElementById("project-discipline").value,
        modality: document.getElementById("project-modality").value,
        task: document.getElementById("project-task").value,
      });
      saveLocal();
      ideaInput.value = "";
      render();
    });

    demoButton?.addEventListener("click", () => {
      projects = examples.map((project) => ({ ...project }));
      saveLocal();
      render();
    });

    clearButton?.addEventListener("click", () => {
      if (!window.confirm("Clear the project ideas stored in this browser?")) return;
      projects = [];
      saveLocal();
      render();
    });

    const csvUrl = landscape.dataset.csv?.trim();
    if (csvUrl) {
      refreshShared(csvUrl);
      window.setInterval(() => refreshShared(csvUrl), 10000);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialize, { once: true });
  } else {
    initialize();
  }
})();
