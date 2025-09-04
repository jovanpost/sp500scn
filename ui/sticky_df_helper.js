// Sticky DataFrame/Table helper – runs in the PARENT page
(function (PWIN) {
  try {
    if (!PWIN) return;
    if (PWIN.__STICKY_READY__) { PWIN.__stickyAudit__?.(); return; }
    PWIN.__STICKY_READY__ = true;

    const DOC = PWIN.document;
    PWIN.__STICKY_DEBUG__ = PWIN.__STICKY_DEBUG__ ?? false;
    const STYLE_ID = "sticky-v3-css";
    const log = (...a) => { if (PWIN.__STICKY_DEBUG__) console.log(...a); };

    function ensureCSS() {
      if (DOC.getElementById(STYLE_ID)) return;
      const s = DOC.createElement("style");
      s.id = STYLE_ID;
      s.textContent = `
        /* Pin table headers */
        table thead th, table thead td { position: sticky; top: 0; z-index: 3; background: inherit; }
        /* Scroll wrapper the helper will tag */
        .sticky-scroll { overflow: auto; }
      `;
      DOC.head.appendChild(s);
    }

    function findScrollNode(el) {
      let n = el?.parentElement;
      const hasScroll = (e) => {
        const cs = PWIN.getComputedStyle(e);
        return /(auto|scroll)/.test(cs.overflow) || /(auto|scroll)/.test(cs.overflowY);
      };
      while (n && n !== DOC.documentElement) {
        if (hasScroll(n)) return n;
        n = n.parentElement;
      }
      return null;
    }

    function auditTable(tbl) {
      if (!tbl || !(tbl instanceof PWIN.Element)) return;
      const wrap = findScrollNode(tbl) || tbl;
      wrap.classList.add("sticky-scroll");
      tbl.querySelectorAll("thead th, thead td").forEach((h) => {
        h.style.position = "sticky";
        h.style.top = "0px";
        h.style.zIndex = "3";
        // keep current theme color
        h.style.background = PWIN.getComputedStyle(h).backgroundColor || "inherit";
      });
    }

    function collectTables() {
      // Streamlit DF containers + any plain tables your app renders
      const roots = DOC.querySelectorAll('div[data-testid="stDataFrame"], .table-wrapper, table.dark-table, table');
      const tables = [];
      roots.forEach((r) => {
        if (r.tagName?.toLowerCase() === "table") {
          tables.push(r);
        } else {
          r.querySelectorAll("table").forEach((t) => tables.push(t));
        }
      });
      return tables;
    }

    function apply() {
      ensureCSS();
      const tables = collectTables();
      tables.forEach(auditTable);
      log("[sticky] processed tables:", tables.length);
      return tables.length;
    }

    // Debounced observer across the whole parent page
    if (!PWIN.__stickyObserver__) {
      PWIN.__stickyObserver__ = new PWIN.MutationObserver(() => {
        clearTimeout(PWIN.__stickyPending__);
        PWIN.__stickyPending__ = PWIN.setTimeout(apply, 80);
      });
      PWIN.__stickyObserver__.observe(DOC.documentElement, { childList: true, subtree: true });
    }

    // Expose manual hook
    PWIN.__stickyAudit__ = apply;

    // Initial run (and also on DOM ready just in case)
    if (DOC.readyState === "loading") {
      DOC.addEventListener("DOMContentLoaded", apply, { once: false });
    }
    apply();
  } catch (e) {
    try { console.log("[sticky] helper crashed:", e); } catch {}
  }
})(window.parent || window);

