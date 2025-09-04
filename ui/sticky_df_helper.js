// ui/sticky_df_helper.js
(function () {
  const PWIN = window.parent || window;
  const DOC  = (window.parent && window.parent.document) ? window.parent.document : document;

  if (PWIN.__STICKY_READY__) {
    PWIN.__stickyAudit__?.();
    return;
  }
  PWIN.__STICKY_READY__ = true;

  const DEBUG = PWIN.__STICKY_DEBUG__ ?? false;
  const log = (...args) => DEBUG && console.log('[sticky]', ...args);

  function findScrollNode(root) {
    // Try to find the div that actually scrolls; fallback to root.
    const nodes = root.querySelectorAll('div, section');
    for (const el of nodes) {
      const cs = getComputedStyle(el);
      if ((cs.overflowY === 'auto' || cs.overflowY === 'scroll')) return el;
    }
    return root;
  }

  function audit(root) {
    if (!root || !(root instanceof Element)) return;
    const scrollNode = findScrollNode(root);

    // Mark both so CSS can style either target
    root.classList.add('sticky-scroll');
    scrollNode.classList.add('sticky-scroll');

    // Ensure headers are sticky and visually on top
    const headers = root.querySelectorAll('thead th,[role="columnheader"]');
    headers.forEach(h => {
      h.style.position = 'sticky';
      h.style.top = '0px';
      h.style.zIndex = '3';
    });
  }

  function apply() {
    const roots = DOC.querySelectorAll('div[data-testid="stDataFrame"]');
    roots.forEach(audit);
    return roots.length;
  }

  // Expose for manual retries
  PWIN.__stickyAudit__ = apply;

  // Debounced global observer on parent DOM to catch re-renders
  if (!PWIN.__stickyObserver__) {
    PWIN.__stickyObserver__ = new MutationObserver(() => {
      clearTimeout(PWIN.__stickyPending__);
      PWIN.__stickyPending__ = setTimeout(apply, 80);
    });
    PWIN.__stickyObserver__.observe(DOC.documentElement, { childList: true, subtree: true });
  }

  const init = () => { log('helper loaded; processed', apply()); };
  if (DOC.readyState === 'loading') DOC.addEventListener('DOMContentLoaded', init, { once: true });
  else init();
})();
