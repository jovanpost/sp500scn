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

    function collectRoots() {
      // Streamlit DataFrame containers + our own HTML tables
      return [...DOC.querySelectorAll(
        'div[data-testid="stDataFrame"], .table-wrapper, table.dark-table, table'
      )];
    }

    function auditStreamlitDF(root) {
      if (!root || !(root instanceof PWIN.Element)) return;
      // tag the scrollable wrapper so our CSS picks it up
      (findScrollNode(root) || root).classList.add('sticky-scroll');

      // virtualized headers in Streamlit: <div role="columnheader">
      const headers = root.querySelectorAll('[role="columnheader"]');
      headers.forEach((h) => {
        const cs = PWIN.getComputedStyle(h);
        h.style.position = 'sticky';
        h.style.top = '0px';
        h.style.zIndex = '3';
        h.style.background = cs.backgroundColor || PWIN.getComputedStyle(h.parentElement || root).backgroundColor || 'inherit';
      });
    }

    function apply() {
      ensureCSS();
      const roots = collectRoots();
      let count = 0;

      roots.forEach((r) => {
        if (r.tagName?.toLowerCase() === 'table') {
          auditTable(r); count++;
        } else {
          const innerTables = r.querySelectorAll('table');
          if (innerTables.length) {
            innerTables.forEach((t) => { auditTable(t); count++; });
          } else {
            auditStreamlitDF(r); count++;
          }
        }
      });

      log('[sticky] processed roots:', count);
      return count;
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

