(function() {
  if (window.__STICKY_READY__) return;
  window.__STICKY_READY__ = true;

  const CLASS = 'sticky-scroll';

  function findScrollNode(el) {
    const walkers = el.querySelectorAll('div');
    for (const node of walkers) {
      const style = getComputedStyle(node);
      if (/(auto|scroll)/.test(style.overflowY + style.overflowX)) {
        return node;
      }
    }
    return el;
  }

  function audit() {
    const frames = document.querySelectorAll('div[data-testid="stDataFrame"]');
    frames.forEach(el => {
      const scrollNode = findScrollNode(el);
      const root = el.closest('div[data-testid="stDataFrame"]') ?? el;
      root.classList.add(CLASS);
      scrollNode.classList.add(CLASS);
    });
    return frames.length;
  }

  window.__stickyAudit__ = audit;

  let timer;
  const observer = new MutationObserver(() => {
    clearTimeout(timer);
    timer = setTimeout(audit, 100);
  });
  observer.observe(document.documentElement, { childList: true, subtree: true });

  audit();
})();
