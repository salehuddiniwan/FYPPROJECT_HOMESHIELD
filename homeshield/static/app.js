// HomeShield dashboard frontend
//
// - Subscribes to /events (Server-Sent Events) for live alerts
// - Polls /healthz once per second for system status
// - Loads /events/history on page load to backfill the alert list

(function () {
    "use strict";

    const MAX_ALERTS = 200;

    const els = {
        list:    document.getElementById("alert-list"),
        clear:   document.getElementById("clear-btn"),
        camPill: document.getElementById("pill-camera"),
        camLbl:  document.getElementById("cam-state"),
        fpsLbl:  document.getElementById("fps-val"),
        fireLbl: document.getElementById("fire-state"),
        fallLbl: document.getElementById("fall-state"),
        banner:  document.getElementById("banner"),
        uptime:  document.getElementById("stat-uptime"),
        frames:  document.getElementById("stat-frames"),
        err:     document.getElementById("stat-err"),
        alerts:  document.getElementById("stat-alerts"),
        modal:   document.getElementById("snap-modal"),
        modalImg: document.getElementById("snap-img"),
        modalCap: document.getElementById("snap-cap"),
    };

    const state = {
        kindFilter: { FIRE: true, FALL: true, LYING_MOTIONLESS: true, SYSTEM: true },
        critActive: false,
        critTimeout: null,
    };

    // ---- helpers --------------------------------------------------------

    function fmtTime(iso) {
        if (!iso) return "--";
        const d = new Date(iso);
        return d.toLocaleTimeString();
    }
    function fmtDur(s) {
        s = Math.max(0, Math.floor(s || 0));
        const h = Math.floor(s / 3600);
        const m = Math.floor((s % 3600) / 60);
        const sec = s % 60;
        return h > 0 ? `${h}h ${m}m ${sec}s` : (m > 0 ? `${m}m ${sec}s` : `${sec}s`);
    }
    function severityClass(ev) {
        if (ev.severity === "critical") return "crit";
        if (ev.severity === "warning") return "warn";
        return "info";
    }
    function applyKindFilter() {
        document.querySelectorAll(".alert[data-kind]").forEach(li => {
            const k = li.getAttribute("data-kind");
            li.style.display = state.kindFilter[k] ? "" : "none";
        });
    }

    // ---- alert list -----------------------------------------------------

    function renderAlert(ev) {
        const li = document.createElement("li");
        li.className = `alert alert-${ev.kind}`;
        li.setAttribute("data-kind", ev.kind);
        li.setAttribute("data-id", ev.id || "");

        const left = document.createElement("div");
        const msg = document.createElement("div");
        msg.className = "a-msg";
        const icon = ev.kind === "FIRE" ? "🔥"
                   : ev.kind === "FALL" ? "🚶‍♂️"
                   : ev.kind === "LYING_MOTIONLESS" ? "⚠️"
                   : "🛈";
        msg.textContent = `${icon} ${ev.kind.replace("_", " ")} — ${ev.message}`;
        left.appendChild(msg);

        const meta = document.createElement("div");
        meta.className = "a-meta";
        const bbox = ev.bbox ? ` · bbox [${ev.bbox.map(n => Math.round(n)).join(", ")}]` : "";
        meta.textContent = `Camera: ${ev.camera_id} · Severity: ${ev.severity}${bbox}`;
        left.appendChild(meta);

        const right = document.createElement("div");
        right.style.display = "flex";
        right.style.alignItems = "center";

        const time = document.createElement("div");
        time.className = "a-time";
        time.textContent = fmtTime(ev.ts_iso);
        right.appendChild(time);

        if (ev.snapshot_path) {
            const img = document.createElement("img");
            img.className = "a-snap";
            img.src = "/" + ev.snapshot_path;
            img.alt = "snapshot";
            img.addEventListener("click", () => {
                els.modal.hidden = false;
                els.modalImg.src = img.src;
                els.modalCap.textContent = `${ev.kind} · ${fmtTime(ev.ts_iso)} · ${ev.message}`;
            });
            right.appendChild(img);
        }

        li.appendChild(left);
        li.appendChild(right);

        // remove "empty" placeholder
        const empty = els.list.querySelector(".alert.empty");
        if (empty) empty.remove();

        els.list.prepend(li);
        while (els.list.children.length > MAX_ALERTS) {
            els.list.removeChild(els.list.lastChild);
        }
        if (!state.kindFilter[ev.kind]) li.style.display = "none";
    }

    function flashCritical(ev) {
        if (!(ev.severity === "critical")) return;
        if (ev.kind === "SYSTEM") return;
        els.banner.className = "banner banner-crit";
        els.banner.textContent = `${ev.kind.replace("_", " ")} ALERT - ${ev.message}`;
        clearTimeout(state.critTimeout);
        state.critTimeout = setTimeout(() => {
            els.banner.className = "banner banner-ok";
            els.banner.textContent = "SYSTEM NOMINAL";
        }, 6000);
    }

    // ---- SSE ------------------------------------------------------------

    function startStream() {
        const es = new EventSource("/events");
        es.addEventListener("alert", (e) => {
            try {
                const ev = JSON.parse(e.data);
                renderAlert(ev);
                flashCritical(ev);
            } catch (err) { console.error("alert parse failed", err); }
        });
        es.addEventListener("hello", () => {
            console.log("[homeshield] event stream connected");
        });
        es.onerror = () => {
            console.warn("[homeshield] event stream lost; auto-reconnecting...");
        };
    }

    // ---- backfill -------------------------------------------------------

    async function backfill() {
        try {
            const r = await fetch("/events/history?limit=50");
            const rows = await r.json();
            // history returns newest-first
            for (let i = rows.length - 1; i >= 0; i--) {
                renderAlert({
                    ...rows[i],
                    bbox: rows[i].bbox,
                });
            }
            els.alerts.textContent = String(rows.length);
        } catch (err) {
            console.warn("backfill failed", err);
        }
    }

    // ---- health polling -------------------------------------------------

    async function pollHealth() {
        try {
            const r = await fetch("/healthz");
            const h = await r.json();
            // camera pill
            els.camPill.classList.remove("pill-ok", "pill-crit", "pill-unknown", "pill-warn", "pill-info");
            els.camPill.classList.add(h.camera_connected ? "pill-ok" : "pill-crit");
            els.camLbl.textContent = h.camera_connected ? "online" : "offline";

            // fps
            els.fpsLbl.textContent = h.fps ? h.fps.toFixed(1) : "--";

            // model status pills
            els.fireLbl.textContent = h.fire_enabled ? "online" : "off";
            els.fallLbl.textContent = h.fall_enabled ? "online" : "off";

            // stats
            els.uptime.textContent = fmtDur(h.uptime_s);
            els.frames.textContent = h.frames_total;
            els.err.textContent = h.last_error ? h.last_error : "none";
        } catch (err) {
            els.camPill.classList.add("pill-crit");
            els.camLbl.textContent = "server unreachable";
        }
    }

    // ---- bootstrap ------------------------------------------------------

    function attachFilters() {
        document.querySelectorAll('.filter input[type="checkbox"]').forEach(cb => {
            cb.addEventListener("change", () => {
                state.kindFilter[cb.dataset.kind] = cb.checked;
                applyKindFilter();
            });
        });
        els.clear.addEventListener("click", () => {
            els.list.innerHTML = '<li class="alert empty">Listening for events…</li>';
        });
        document.querySelector(".modal-close").addEventListener("click", () => {
            els.modal.hidden = true;
        });
        els.modal.addEventListener("click", (e) => {
            if (e.target === els.modal) els.modal.hidden = true;
        });
    }

    document.addEventListener("DOMContentLoaded", () => {
        attachFilters();
        backfill();
        startStream();
        pollHealth();
        setInterval(pollHealth, 1000);
    });
})();
