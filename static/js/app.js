/**
 * app.js — Dynamic Mexico Municipality Wildfire Risk Choropleth Map with 2015–2026 Timeline Slider.
 */

(function () {
  "use strict";

  // ── State ──────────────────────────────────────────────────────────────────
  const state = {
    minDate: "2015-01-01",
    maxDate: "2026-09-19",
    today: "2026-09-04",
    currentDate: "2024-05-15",
    isPlaying: false,
    playInterval: null,
    playSpeed: 200, // ms per day
    riskCache: new Map(), // dateStr -> data
    selectedCvegeo: null,
    selectedLayer: null,
    cveLayerMap: new Map(), // cvegeo -> Leaflet layer
    municipalitiesList: [], // for search autocomplete
    currentRisks: {}, // active date's { cvegeo: { risk, prob, fires } }
  };

  // ── Risk Color Palette (Curated Light Theme) ───────────────────────────────
  const COLOR_HIGH = "#ef4444";   // Vibrant Red (High Risk)
  const COLOR_MED  = "#f97316";   // Amber Orange (Medium Risk)
  const COLOR_LOW  = "#10b981";   // Emerald Green (Low Risk)
  const COLOR_NONE = "#10b981";   // Emerald Green (Same Green for All Low/Baseline Areas)
  const COLOR_SELECT = "#2563eb"; // Blue highlight

  // ── DOM References ─────────────────────────────────────────────────────────
  const dom = {
    map: document.getElementById("map"),
    statHigh: document.getElementById("stat-high"),
    statMed: document.getElementById("stat-med"),
    statFires: document.getElementById("stat-fires"),
    searchInput: document.getElementById("muni-search-input"),
    searchResults: document.getElementById("search-results"),
    btnResetView: document.getElementById("btn-reset-view"),
    btnPrevDay: document.getElementById("btn-prev-day"),
    btnNextDay: document.getElementById("btn-next-day"),
    btnPlayPause: document.getElementById("btn-play-pause"),
    iconPlay: document.getElementById("icon-play"),
    iconPause: document.getElementById("icon-pause"),
    speedSelect: document.getElementById("speed-select"),
    dateSlider: document.getElementById("date-slider"),
    displayWeekday: document.getElementById("display-weekday"),
    displayDate: document.getElementById("display-date"),
    displaySeason: document.getElementById("display-season"),
    detailDrawer: document.getElementById("detail-drawer"),
    btnCloseDrawer: document.getElementById("btn-close-drawer"),
    btnZoomOut: document.getElementById("btn-zoom-out"),
    toast: document.getElementById("toast"),

    // Detail drawer fields
    dtState: document.getElementById("dt-state"),
    dtMunicipality: document.getElementById("dt-municipality"),
    dtCvegeo: document.getElementById("dt-cvegeo"),
    dtDate: document.getElementById("dt-date"),
    dtSourceBadge: document.getElementById("dt-source-badge"),
    dtProb: document.getElementById("dt-prob"),
    dtProbBar: document.getElementById("dt-prob-bar"),
    dtRiskBadge: document.getElementById("dt-risk-badge"),
    dtFireStatus: document.getElementById("dt-fire-status"),
    dtWTempMax: document.getElementById("dt-w-temp-max"),
    dtWTempMin: document.getElementById("dt-w-temp-min"),
    dtWTempMean: document.getElementById("dt-w-temp-mean"),
    dtWHumidity: document.getElementById("dt-w-humidity"),
    dtWPrecip: document.getElementById("dt-w-precip"),
    dtWWind: document.getElementById("dt-w-wind"),
    dtWGusts: document.getElementById("dt-w-gusts"),
    dtWEt0: document.getElementById("dt-w-et0"),
    dtWDry: document.getElementById("dt-w-dry"),
  };

  // ── Date Math Helpers ──────────────────────────────────────────────────────
  const MS_PER_DAY = 86400000;
  const START_TS = new Date("2015-01-01T00:00:00").getTime();

  function dateToOffset(dateStr) {
    const t = new Date(dateStr + "T00:00:00").getTime();
    return Math.round((t - START_TS) / MS_PER_DAY);
  }

  function offsetToDate(offset) {
    const d = new Date(START_TS + offset * MS_PER_DAY);
    return d.toISOString().split("T")[0];
  }

  function formatDisplayDate(dateStr) {
    const d = new Date(dateStr + "T12:00:00");
    const weekday = d.toLocaleDateString("en-US", { weekday: "long" });
    const calendar = d.toLocaleDateString("en-US", { month: "long", day: "numeric", year: "numeric" });
    
    // Determine season tag
    const m = d.getMonth() + 1;
    let season = "Standard Conditions";
    if (m >= 3 && m <= 5) {
      season = "Peak Wildfire Season (High Risk)";
    } else if (m >= 6 && m <= 10) {
      season = "Rainy Season (Lower Risk)";
    } else if (m >= 11 || m <= 2) {
      season = "Dry Winter Season";
    }

    const todayStr = state.today;
    if (dateStr > todayStr) {
      season = `16-Day Forecast Horizon (${season})`;
    }

    return { weekday, calendar, season };
  }

  // ── Leaflet Map Setup ──────────────────────────────────────────────────────
  const map = L.map(dom.map, {
    center: [23.6345, -102.5528],
    zoom: 5,
    minZoom: 4,
    maxZoom: 14,
    zoomControl: false,
  });

  L.control.zoom({ position: "topright" }).addTo(map);

  // Modern Light Theme Basemap (CartoDB Positron)
  L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
    attribution: '&copy; <a href="https://carto.com/">CARTO</a>, &copy; INEGI / CONAFOR',
    subdomains: "abcd",
    maxZoom: 19,
  }).addTo(map);

  let geojsonLayer = null;

  // ── Polygon Styling ────────────────────────────────────────────────────────
  function getFeatureStyle(feature) {
    const cve = feature.properties.cvegeo;
    const isSelected = state.selectedCvegeo === cve;
    const rInfo = state.currentRisks[cve];

    // Default to Green for municipalities without data so the entire Mexico map is fully colored
    let fillColor = COLOR_NONE;
    let fillOpacity = 0.55;

    if (rInfo) {
      if (rInfo.risk === "HIGH") {
        fillColor = COLOR_HIGH;
        fillOpacity = 0.8;
      } else if (rInfo.risk === "MEDIUM") {
        fillColor = COLOR_MED;
        fillOpacity = 0.7;
      } else if (rInfo.risk === "LOW") {
        fillColor = COLOR_LOW;
        fillOpacity = 0.65;
      }
      if (rInfo.fires > 0) {
        fillOpacity = 0.95;
      }
    }

    return {
      fillColor: fillColor,
      weight: isSelected ? 3.5 : 0.6,
      opacity: 1,
      color: isSelected ? COLOR_SELECT : "#ffffff",
      fillOpacity: fillOpacity,
    };
  }

  function updateChoroplethColors() {
    if (!geojsonLayer) return;
    geojsonLayer.eachLayer((layer) => {
      layer.setStyle(getFeatureStyle(layer.feature));
    });
  }

  // ── TopoJSON Loading & Layer Creation ──────────────────────────────────────
  async function loadTopoJsonMap() {
    try {
      const resp = await fetch("/static/data/mexico_municipalities.topojson");
      if (!resp.ok) throw new Error("Could not load Mexico TopoJSON boundaries");
      const topo = await resp.json();

      // Convert TopoJSON to GeoJSON using local topojson client
      const geojson = topojson.feature(topo, topo.objects.municipalities);

      // Populate municipality search catalog
      state.municipalitiesList = geojson.features.map((f) => ({
        cvegeo: f.properties.cvegeo,
        name: f.properties.mun_name,
        state: f.properties.state_name,
      }));

      geojsonLayer = L.geoJSON(geojson, {
        style: getFeatureStyle,
        onEachFeature: (feature, layer) => {
          const cve = feature.properties.cvegeo;
          state.cveLayerMap.set(cve, layer);

          // Lightweight hover tooltip
          layer.on({
            mouseover: (e) => {
              const target = e.target;
              if (state.selectedCvegeo !== cve) {
                target.setStyle({ weight: 2, color: "#334155" });
                target.bringToFront();
              }
              const r = state.currentRisks[cve];
              const riskStr = r ? `${r.risk} (${r.prob}%)` : "LOW (< 2.12%)";
              const fireNote = r && r.fires > 0 ? ` 🔥 ${r.fires} fire(s)` : "";
              target.bindTooltip(
                `<strong>${feature.properties.mun_name}</strong>, ${feature.properties.state_name}<br/>Risk: ${riskStr}${fireNote}`,
                { sticky: true, className: "choropleth-tooltip" }
              ).openTooltip();
            },
            mouseout: (e) => {
              const target = e.target;
              if (state.selectedCvegeo !== cve) {
                geojsonLayer.resetStyle(target);
              }
            },
            click: (e) => {
              selectMunicipality(feature.properties.cvegeo, layer);
            },
          });
        },
      }).addTo(map);

      // Fit map bounds to Mexico
      map.fitBounds(geojsonLayer.getBounds(), { padding: [10, 10] });

      // Load initial risk data
      await loadRiskForDate(state.currentDate);
    } catch (err) {
      console.error("Error loading boundaries:", err);
      showToast("Error loading Mexico geographic boundaries: " + err.message, "error");
    }
  }

  // ── Date Risk Fetching ─────────────────────────────────────────────────────
  let fetchDebounceTimer = null;

  async function loadRiskForDate(dateStr) {
    state.currentDate = dateStr;
    updateDateDisplay(dateStr);

    // Check client-side memory cache first
    if (state.riskCache.has(dateStr)) {
      const data = state.riskCache.get(dateStr);
      applyRiskData(data);
      return;
    }

    try {
      const resp = await fetch(`/api/map-risk?date=${encodeURIComponent(dateStr)}`);
      if (!resp.ok) throw new Error("Failed to load risk data");
      const data = await resp.json();
      state.riskCache.set(dateStr, data);
      applyRiskData(data);
    } catch (err) {
      console.error("Error fetching map risk:", err);
    }
  }

  function applyRiskData(data) {
    state.currentRisks = data.risks || {};

    // Update Topbar Stats
    if (data.summary) {
      dom.statHigh.textContent = (data.summary.high || 0).toLocaleString();
      dom.statMed.textContent = (data.summary.medium || 0).toLocaleString();
      dom.statFires.textContent = (data.summary.fires || 0).toLocaleString();
    }

    // Refresh Map Colors
    updateChoroplethColors();

    // If a municipality is currently selected, refresh its detail panel
    if (state.selectedCvegeo) {
      loadMunicipalityDetail(state.selectedCvegeo, state.currentDate, false);
    }
  }

  function updateDateDisplay(dateStr) {
    const info = formatDisplayDate(dateStr);
    dom.displayWeekday.textContent = info.weekday;
    dom.displayDate.textContent = info.calendar;
    dom.displaySeason.textContent = info.season;

    const offset = dateToOffset(dateStr);
    dom.dateSlider.value = offset;
  }

  // ── Municipality Selection & Zoom-In ───────────────────────────────────────
  async function selectMunicipality(cvegeo, layer, shouldZoom = true) {
    // Pause animation if playing
    if (state.isPlaying) {
      togglePlay();
    }

    // Unstyle previous selected layer
    if (state.selectedLayer && state.selectedLayer !== layer) {
      geojsonLayer.resetStyle(state.selectedLayer);
    }

    state.selectedCvegeo = cvegeo;
    state.selectedLayer = layer;

    if (layer) {
      layer.setStyle({
        weight: 3.5,
        color: COLOR_SELECT,
        fillOpacity: Math.max(layer.options.fillOpacity || 0.6, 0.7),
      });
      layer.bringToFront();

      if (shouldZoom) {
        map.fitBounds(layer.getBounds(), {
          padding: [80, 80],
          maxZoom: 10,
          duration: 0.8,
        });
      }
    }

    // Open Drawer & Load Detail
    dom.detailDrawer.classList.add("detail-drawer--open");
    await loadMunicipalityDetail(cvegeo, state.currentDate, true);
  }

  async function loadMunicipalityDetail(cvegeo, dateStr, showLoading = true) {
    if (showLoading) {
      dom.dtMunicipality.textContent = "Loading...";
      dom.dtRiskBadge.textContent = "…";
      dom.dtProb.textContent = "—";
      dom.dtProbBar.style.width = "0%";
    }

    try {
      const resp = await fetch(`/api/municipality-detail?cvegeo=${encodeURIComponent(cvegeo)}&date=${encodeURIComponent(dateStr)}`);
      if (!resp.ok) throw new Error("Could not retrieve municipality detail");
      const det = await resp.json();

      dom.dtState.textContent = det.estado_display || det.estado;
      dom.dtMunicipality.textContent = det.municipio_display || det.municipio;
      dom.dtCvegeo.textContent = det.cvegeo;
      dom.dtDate.textContent = det.date;
      dom.dtSourceBadge.textContent = det.data_source || "ERA5 Historical Reanalysis";

      const hasData = Boolean(state.currentRisks[cvegeo]);
      const prob = Number(det.probability) || 0;
      dom.dtProb.textContent = prob.toFixed(1) + "%";
      dom.dtProbBar.style.width = Math.min(prob, 100) + "%";

      const risk = det.risk_level || "LOW";
      dom.dtRiskBadge.textContent = risk;
      dom.dtRiskBadge.className = `risk-badge risk-badge--${risk.toLowerCase()}`;
      dom.dtProbBar.className = `prob-bar-fill prob-bar-fill--${risk.toLowerCase()}`;

      if (det.fires_count > 0) {
        dom.dtFireStatus.innerHTML = `🔥 <strong>${det.fires_count} wildfire ignition(s)</strong> confirmed on this date`;
        dom.dtFireStatus.className = "detail-risk-card__fire-status detail-risk-card__fire-status--active";
      } else {
        dom.dtFireStatus.textContent = "No active wildfire ignitions reported on this date";
        dom.dtFireStatus.className = "detail-risk-card__fire-status";
      }

      // Fill Weather Grid
      const w = det.weather || {};
      const fmt = (val, unit = "") => (val !== null && val !== undefined && !isNaN(val) ? `${Number(val).toFixed(1)}${unit}` : "—");

      dom.dtWTempMax.textContent = fmt(w.temp_max_c, " °C");
      dom.dtWTempMin.textContent = fmt(w.temp_min_c, " °C");
      dom.dtWTempMean.textContent = fmt(w.temp_mean_c, " °C");
      dom.dtWHumidity.textContent = fmt(w.relative_humidity_pct, " %");
      dom.dtWPrecip.textContent = fmt(w.precipitation_mm, " mm");
      dom.dtWWind.textContent = fmt(w.wind_speed_kmh, " km/h");
      dom.dtWGusts.textContent = fmt(w.wind_gusts_kmh, " km/h");
      dom.dtWEt0.textContent = fmt(w.evapotranspiration_mm, " mm");
      dom.dtWDry.textContent = w.consecutive_dry_days !== null && w.consecutive_dry_days !== undefined ? `${w.consecutive_dry_days} days` : "—";
    } catch (err) {
      console.error("Error loading municipality detail:", err);
      showToast("Could not load municipality details: " + err.message, "error");
    }
  }

  function deselectMunicipality() {
    if (state.selectedLayer && geojsonLayer) {
      geojsonLayer.resetStyle(state.selectedLayer);
    }
    state.selectedCvegeo = null;
    state.selectedLayer = null;
    dom.detailDrawer.classList.remove("detail-drawer--open");
  }

  // ── Timeline Slider & Playback ─────────────────────────────────────────────
  function stepDay(direction) {
    const currentOffset = dateToOffset(state.currentDate);
    const maxOffset = dateToOffset(state.maxDate);
    const minOffset = 0;

    let newOffset = currentOffset + direction;
    if (newOffset > maxOffset) newOffset = minOffset;
    if (newOffset < minOffset) newOffset = maxOffset;

    const newDate = offsetToDate(newOffset);
    loadRiskForDate(newDate);
  }

  function togglePlay() {
    state.isPlaying = !state.isPlaying;
    if (state.isPlaying) {
      dom.iconPlay.style.display = "none";
      dom.iconPause.style.display = "block";
      dom.btnPlayPause.classList.add("btn-play-pause--active");
      state.playInterval = setInterval(() => {
        stepDay(1);
      }, state.playSpeed);
    } else {
      dom.iconPlay.style.display = "block";
      dom.iconPause.style.display = "none";
      dom.btnPlayPause.classList.remove("btn-play-pause--active");
      if (state.playInterval) {
        clearInterval(state.playInterval);
        state.playInterval = null;
      }
    }
  }

  // ── Search Autocomplete ────────────────────────────────────────────────────
  function handleSearchInput(query) {
    const q = query.trim().toLowerCase();
    if (!q || q.length < 2) {
      dom.searchResults.style.display = "none";
      dom.searchResults.innerHTML = "";
      return;
    }

    const matches = state.municipalitiesList
      .filter((m) => m.name.toLowerCase().includes(q) || m.state.toLowerCase().includes(q))
      .slice(0, 8);

    if (matches.length === 0) {
      dom.searchResults.innerHTML = `<div class="search-item search-item--none">No municipalities found</div>`;
      dom.searchResults.style.display = "block";
      return;
    }

    dom.searchResults.innerHTML = matches
      .map(
        (m) => `
        <div class="search-item" data-cve="${m.cvegeo}">
          <strong>${m.name}</strong> <span class="search-item__state">· ${m.state}</span>
        </div>`
      )
      .join("");
    dom.searchResults.style.display = "block";
  }

  // ── Toast Notifications ────────────────────────────────────────────────────
  let toastTimer = null;
  function showToast(msg, type = "info") {
    dom.toast.textContent = msg;
    dom.toast.className = `toast toast--visible toast--${type}`;
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      dom.toast.className = "toast";
    }, 4000);
  }

  // ── Event Handlers ─────────────────────────────────────────────────────────
  function bindEvents() {
    // Slider Drag / Input
    dom.dateSlider.addEventListener("input", (e) => {
      const offset = parseInt(e.target.value, 10);
      const targetDate = offsetToDate(offset);
      updateDateDisplay(targetDate);

      // Debounce network request when user rapidly drags slider
      if (fetchDebounceTimer) clearTimeout(fetchDebounceTimer);
      fetchDebounceTimer = setTimeout(() => {
        loadRiskForDate(targetDate);
      }, 75);
    });

    // Prev / Next Day
    dom.btnPrevDay.addEventListener("click", () => stepDay(-1));
    dom.btnNextDay.addEventListener("click", () => stepDay(1));

    // Play / Pause
    dom.btnPlayPause.addEventListener("click", togglePlay);

    // Speed Select
    dom.speedSelect.addEventListener("change", (e) => {
      state.playSpeed = parseInt(e.target.value, 10);
      if (state.isPlaying) {
        clearInterval(state.playInterval);
        state.playInterval = setInterval(() => stepDay(1), state.playSpeed);
      }
    });

    // Preset Buttons
    document.querySelectorAll(".btn-preset").forEach((btn) => {
      btn.addEventListener("click", () => {
        const dt = btn.dataset.date;
        if (dt) loadRiskForDate(dt);
      });
    });

    // Reset View Button
    dom.btnResetView.addEventListener("click", () => {
      deselectMunicipality();
      map.flyTo([23.6345, -102.5528], 5, { duration: 0.8 });
    });

    // Close Drawer Buttons
    dom.btnCloseDrawer.addEventListener("click", deselectMunicipality);
    dom.btnZoomOut.addEventListener("click", () => {
      deselectMunicipality();
      map.flyTo([23.6345, -102.5528], 5, { duration: 0.8 });
    });

    // Search Input
    dom.searchInput.addEventListener("input", (e) => handleSearchInput(e.target.value));
    dom.searchResults.addEventListener("click", (e) => {
      const item = e.target.closest(".search-item");
      if (!item || !item.dataset.cve) return;
      const cve = item.dataset.cve;
      const layer = state.cveLayerMap.get(cve);
      dom.searchResults.style.display = "none";
      dom.searchInput.value = "";
      if (layer) {
        selectMunicipality(cve, layer, true);
      } else {
        showToast("Municipality geometry not available on map", "error");
      }
    });

    document.addEventListener("click", (e) => {
      if (!dom.searchInput.contains(e.target) && !dom.searchResults.contains(e.target)) {
        dom.searchResults.style.display = "none";
      }
    });
  }

  // ── Application Initialization ─────────────────────────────────────────────
  async function init() {
    bindEvents();

    // Fetch config to set slider bounds
    try {
      const resp = await fetch("/api/config");
      if (resp.ok) {
        const cfg = await resp.json();
        state.minDate = cfg.min_date || "2015-01-01";
        state.maxDate = cfg.max_date || "2026-09-19";
        state.today = cfg.today || "2026-09-04";

        const minOffset = 0;
        const maxOffset = dateToOffset(state.maxDate);
        dom.dateSlider.min = minOffset;
        dom.dateSlider.max = maxOffset;

        // Set preset today button
        const btnToday = document.getElementById("preset-today");
        if (btnToday) btnToday.dataset.date = state.today;
        const btnForecast = document.getElementById("preset-forecast");
        if (btnForecast) btnForecast.dataset.date = state.maxDate;

        if (cfg.thresholds) {
          const lLow = document.getElementById("legend-range-low");
          const lMed = document.getElementById("legend-range-med");
          const lHigh = document.getElementById("legend-range-high");
          if (lLow) lLow.textContent = `< ${cfg.thresholds.medium_threshold}%`;
          if (lMed) lMed.textContent = `${cfg.thresholds.medium_threshold}% – ${cfg.thresholds.high_threshold}%`;
          if (lHigh) lHigh.textContent = `≥ ${cfg.thresholds.high_threshold}%`;
        }
      }
    } catch (e) {
      console.warn("Using fallback config:", e);
    }

    // Load TopoJSON map and render initial view
    await loadTopoJsonMap();
  }

  // Launch on DOM ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
