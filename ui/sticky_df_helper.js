/*!  
 * sticky_df_helper.js
 * Makes HTML/Pandas/Streamlit tables have sticky headers (and optional sticky first column).
 * Safe to include multiple times; it self-guards and re-applies to dynamic tables.
 */
(function StickyDFHelper () {
  const STYLE_ID = '__sticky_df_css__';
  const WRAP_CLASS = 'table-wrap';
  const TABLE_CLASS = 'sticky-table';
  const FIRST_COL_STICKY = false; // <- set true if you also want the first column sticky

  // ---------- CSS injection ----------
  function injectCSS(doc) {
    if (!doc || doc.getElementById(STYLE_ID)) return;

    const css = `
    /* Scroll container */
    .${WRAP_CLASS} { max-height: 72vh; overflow: auto; }

    /* Ensure sticky works reliably with borders */
    .${WRAP_CLASS} table { border-collapse: separate !important; border-spacing: 0 !important; }

    /* Sticky header cells */
    .${WRAP_CLASS} thead th {
      position: sticky; top: 0; z-index: 10;
      background: #374151 !important; /* Tailwind gray-700 to match UI */
      background-clip: padding-box;
    }

    /* Optional: sticky first column */
    ${FIRST_COL_STICKY ? `
    .${WRAP_CLASS} tbody td:first-child, .${WRAP_CLASS} thead th:first-child { position: sticky; left: 0; }
    .${WRAP_CLASS} tbody td:first-child { z-index: 5; }
    .${WRAP_CLASS} thead th:first-child { z-index: 11; }
    ` : ''}

    /* Prevent clipping of sticky layers when nested in flex/grid containers */
    .${WRAP_CLASS}, .${WRAP_CLASS} * { overflow: visible; }
    `.trim();

    const style = doc.createElement('style');
    style.id = STYLE_ID;
    style.textContent = css;
    doc.head && doc.head.appendChild(style);
  }

  // ---------- DOM utils ----------
  function isWrapped(table) {
    return table && table.parentElement && table.parentElement.classList.contains(WRAP_CLASS);
  }

  function wrapTable(table, doc) {
    if (!table || isWrapped(table)) return;

    const wrapper = doc.createElement('div');
    wrapper.className = WRAP_CLASS;

    // Keep current size to avoid layout jumps
    const rect = table.getBoundingClientRect();
    if (rect.width) wrapper.style.width = rect.width + 'px';

    table.parentNode.insertBefore(wrapper, table);
    wrapper.appendChild(table);
  }

  // Heuristic: which tables should we enhance?
  function isEnhanceableTable(table) {
    if (!(table instanceof HTMLTableElement)) return false;
    if (table.classList.contains(TABLE_CLASS)) return false; // already processed
    const hasThead = !!table.querySelector('thead');
    const hasTh = !!table.querySelector('thead th');
    const rows = table.querySelectorAll('tbody tr').length;
    return hasThead && hasTh && rows >= 2;
  }

  function enhanceTable(table, doc) {
    if (!isEnhanceableTable(table)) return;
    wrapTable(table, doc);
    table.classList.add(TABLE_CLASS);
  }

  function enhanceAllTables(doc) {
    injectCSS(doc);
    const tables = doc.querySelectorAll('table');
    tables.forEach(t => enhanceTable(t, doc));
  }

  // ---------- Mutation observer for dynamic tables ----------
  function observe(doc) {
    if (!doc || doc.__sticky_df_observing__) return;
    doc.__sticky_df_observing__ = true;

    const obs = new MutationObserver(mutations => {
      for (const m of mutations) {
        m.addedNodes.forEach(node => {
          if (node instanceof HTMLTableElement) {
            enhanceTable(node, doc);
          } else if (node instanceof HTMLElement) {
            node.querySelectorAll && node.querySelectorAll('table').forEach(t => enhanceTable(t, doc));
          }
        });
      }
    });

    obs.observe(doc.body || doc, { childList: true, subtree: true });
    doc.__sticky_df_observer__ = obs;
  }

  // ---------- Same-origin iframe support (e.g., Streamlit) ----------
  function enhanceSameOriginIframes(rootDoc) {
    const iframes = rootDoc.querySelectorAll('iframe');
    iframes.forEach(ifr => {
      try {
        const doc = ifr.contentDocument;
        if (!doc) return;
        enhanceAllTables(doc);
        observe(doc);
      } catch (_e) {
        // Cross-origin; skip silently
      }
    });
  }

  // ---------- Public init (idempotent) ----------
  function init(doc) {
    const d = doc || document;
    injectCSS(d);
    enhanceAllTables(d);
    observe(d);
    enhanceSameOriginIframes(d);
  }

  // Auto-run on current document
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => init(document), { once: true });
  } else {
    init(document);
  }

  // Expose minimal API (optional)
  window.StickyDF = { init, version: '1.0.0' };
})();

