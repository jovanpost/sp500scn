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

    function ensureCSS(doc = DOC) {
      if (doc.getElementById(STYLE_ID)) return;
      const s = doc.createElement("style");
      s.id = STYLE_ID;
      s.textContent = `
        table.dark-table.sticky-scroll, table.sticky-scroll { border-collapse: separate; }
        .sticky-scroll {
          position: relative;
          overflow: auto !important;
          z-index: 0;
        }
        .sticky-scroll thead th,
        .sticky-scroll thead td,
        [role="columnheader"] {
          position: sticky !important;
          top: 0 !important;
          z-index: 5 !important;
          backdrop-filter: saturate(140%);
        }
      `;
      doc.head && doc.head.appendChild(s);
    }

    function findScrollNode(el) {
      if (!el) return null;
      const doc = el.ownerDocument || DOC;
      const win = doc.defaultView || window;
      let n = el.parentElement;
      const hasScroll = (e) => {
        const cs = win.getComputedStyle(e);
        return /(auto|scroll)/.test(cs.overflow) || /(auto|scroll)/.test(cs.overflowY);
      };
      while (n && n !== doc.documentElement) {
        if (hasScroll(n)) return n;
        n = n.parentElement;
      }
      return null;
    }

    function auditTable(tbl) {
      if (!tbl) return;
      const wrap = findScrollNode(tbl) || tbl;
      wrap.classList.add("sticky-scroll");
      tbl.querySelectorAll("thead th, thead td").forEach((h) => {
        const W = (h.ownerDocument?.defaultView) || window;
        const bg =
          W.getComputedStyle(h).backgroundColor ||
          W.getComputedStyle(h.parentElement || tbl).backgroundColor ||
          "inherit";
        h.style.position = "sticky";
        h.style.top = "0px";
        h.style.zIndex = "5";
        h.style.background = bg;
      });
    }

    function collectRoots(doc = DOC) {
      return [...doc.querySelectorAll(
        'div[data-testid="stDataFrame"], .table-wrapper, table.dark-table, table'
      )];
    }

    function auditStreamlitDF(root) {
      if (!root) return;
      (findScrollNode(root) || root).classList.add('sticky-scroll');
      const headers = root.querySelectorAll('[role="columnheader"]');
      headers.forEach((h) => {
        const W = (h.ownerDocument?.defaultView) || window;
        const cs = W.getComputedStyle(h);
        const bg = cs.backgroundColor ||
          W.getComputedStyle(h.parentElement || root).backgroundColor ||
          'inherit';
        h.style.position = 'sticky';
        h.style.top = '0px';
        h.style.zIndex = '5';
        h.style.background = bg;
      });
    }

    function applyInDoc(doc) {
      ensureCSS(doc);
      let count = 0;
      collectRoots(doc).forEach((r) => {
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
      return count;
    }

    function apply() {
      let total = applyInDoc(DOC);
      DOC.querySelectorAll('iframe').forEach((ifr) => {
        try {
          const d = ifr.contentDocument;
          if (d) total += applyInDoc(d);
        } catch (_) {}
      });
      log('[sticky] TOTAL roots:', total);
      return total;
    }

    if (!PWIN.__stickyObserver__) {
      PWIN.__stickyObserver__ = new PWIN.MutationObserver(() => {
        clearTimeout(PWIN.__stickyPending__);
        PWIN.__stickyPending__ = PWIN.setTimeout(apply, 80);
      });
      PWIN.__stickyObserver__.observe(DOC.documentElement, { childList: true, subtree: true });
    }

    PWIN.__stickyAudit__ = apply;

    if (DOC.readyState === "loading") {
      DOC.addEventListener("DOMContentLoaded", apply, { once: false });
    }
    apply();
  } catch (e) {
    try { console.log("[sticky] helper crashed:", e); } catch {}
  }
})(window.parent || window);
