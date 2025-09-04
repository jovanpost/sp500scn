// Sticky DataFrame/Table helper – runs in the PARENT page
(function (PWIN) {
  try {
    const TOP = PWIN;
    const log = (...a) => { try { TOP.console.log(...a); } catch {} };

    // --- CSS per-document ---
    function ensureCSS(doc) {
      const id = '__sticky_css__';
      if (doc.getElementById(id)) return;
      const s = doc.createElement('style'); s.id = id;
      s.textContent = `
        .sticky-scroll { position: relative !important; overflow: auto !important; z-index: 0; }
        .sticky-scroll thead th, .sticky-scroll thead td, [role="columnheader"] {
          position: sticky !important; top: 0 !important; z-index: 6 !important;
          backdrop-filter: saturate(140%);
        }
      `;
      doc.head && doc.head.appendChild(s);
    }

    // --- find first scrolling/clipping ancestor in the element's own context ---
    function findScrollNode(el) {
      const DOC = el?.ownerDocument;
      const WIN = DOC?.defaultView || TOP;
      if (!DOC) return null;
      const isClip = (e) => {
        const cs = WIN.getComputedStyle(e);
        return /(auto|scroll|hidden)/.test(cs.overflowY) || /(auto|scroll|hidden)/.test(cs.overflow);
      };
      let n = el.parentElement;
      while (n && n !== DOC.documentElement) {
        if (isClip(n)) return n;
        n = n.parentElement;
      }
      return null;
    }

    // --- auditors ---
    function auditTable(tbl) {
      const DOC = tbl.ownerDocument, WIN = DOC.defaultView;
      const wrap = findScrollNode(tbl) || tbl;
      wrap.classList.add('sticky-scroll');
      if (WIN.getComputedStyle(wrap).position === 'static') wrap.style.position = 'relative';
      if (!/auto|scroll/.test(WIN.getComputedStyle(wrap).overflow + WIN.getComputedStyle(wrap).overflowY)) {
        wrap.style.overflow = 'auto';
      }
      tbl.querySelectorAll('thead th, thead td').forEach((h) => {
        const bg = WIN.getComputedStyle(h).backgroundColor;
        h.style.position = 'sticky'; h.style.top = '0px'; h.style.zIndex = '6';
        h.style.background = (bg === 'rgba(0, 0, 0, 0)' ? '#101425' : bg);
      });
      return 1;
    }

    function auditStreamlitDF(root) {
      const DOC = root.ownerDocument, WIN = DOC.defaultView;
      const headers = root.querySelectorAll('[role="columnheader"]');
      if (!headers.length) return 0;
      const wrap = findScrollNode(root) || root;
      wrap.classList.add('sticky-scroll');
      if (WIN.getComputedStyle(wrap).position === 'static') wrap.style.position = 'relative';
      if (!/auto|scroll/.test(WIN.getComputedStyle(wrap).overflow + WIN.getComputedStyle(wrap).overflowY)) {
        wrap.style.overflow = 'auto';
      }
      headers.forEach((h) => {
        const bg = WIN.getComputedStyle(h).backgroundColor;
        h.style.position = 'sticky'; h.style.top = '0px'; h.style.zIndex = '6';
        h.style.background = (bg === 'rgba(0, 0, 0, 0)' ? '#101425' : bg);
      });
      return 1;
    }

    // --- per-document driver ---
    function applyInDoc(DOC, WIN) {
      ensureCSS(DOC);

      let count = 0;

      // 1) regular tables
      DOC.querySelectorAll('table.dark-table, .table-wrapper table, table').forEach(t => { count += auditTable(t) });

      // 2) virtualized Streamlit DFs
      DOC.querySelectorAll('div[data-testid="stDataFrame"]').forEach(r => { count += auditStreamlitDF(r) });

      return count;
    }

    // --- top-level driver: walk same-origin iframes as well ---
    function apply() {
      let total = 0;
      total += applyInDoc(document, window);

      document.querySelectorAll('iframe').forEach((ifr, i) => {
        try {
          const d = ifr.contentDocument, w = ifr.contentWindow;
          if (d && w) total += applyInDoc(d, w);
        } catch { /* cross-origin, ignore */ }
      });

      log('[sticky] total processed roots:', total);
      return total;
    }

    // MutationObserver (debounced)
    if (!PWIN.__stickyObserver__) {
      PWIN.__stickyObserver__ = new PWIN.MutationObserver(() => {
        clearTimeout(PWIN.__stickyPending__);
        PWIN.__stickyPending__ = PWIN.setTimeout(apply, 80);
      });
      PWIN.__stickyObserver__.observe(document.documentElement, { childList: true, subtree: true });
    }

    // Expose manual hook + initial run
    PWIN.__stickyAudit__ = apply;
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', apply, { once: true });
    apply();
  } catch (e) { try { console.log('[sticky] crashed:', e); } catch {} }
})(window.parent || window);

