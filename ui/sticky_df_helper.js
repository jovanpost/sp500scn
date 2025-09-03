window.__STICKY_DEBUG__ = window.__STICKY_DEBUG__ ?? false;

(function() {
  function log(...args) {
    if (window.__STICKY_DEBUG__) {
      console.log(...args);
    }
  }

  function findScrollNode(root) {
    const nodes = root.querySelectorAll('*');
    for (const node of nodes) {
      const cs = getComputedStyle(node);
      if (node.scrollHeight > node.clientHeight && /(auto|scroll|overlay)/.test(cs.overflowY)) {
        return node;
      }
    }
    return null;
  }

  function audit(root) {
    const scrollNode = findScrollNode(root);
    if (!scrollNode) {
      return;
    }
    scrollNode.classList.add('sticky-scroll');

    // ensure ancestors don't block sticky
    let ancestor = scrollNode.parentElement;
    while (ancestor && ancestor !== root) {
      const cs = getComputedStyle(ancestor);
      if (cs.overflow !== 'visible') {
        ancestor.style.overflow = 'visible';
      }
      ancestor = ancestor.parentElement;
    }

    // mark headers
    root.querySelectorAll('thead th, [role="columnheader"]').forEach((th) => {
      th.classList.add('sticky-header');
    });

    if (window.__STICKY_DEBUG__) {
      const chain = [];
      let el = scrollNode;
      while (el) {
        const cs = getComputedStyle(el);
        chain.push(`${el.tagName.toLowerCase()}[data-testid="${el.dataset.testid || ''}"]{overflow:${cs.overflow};overflow-y:${cs.overflowY};position:${cs.position};display:${cs.display};z-index:${cs.zIndex}}`);
        if (el === root) break;
        el = el.parentElement;
      }
      log(chain.join(' -> '));
      const prev = scrollNode.style.outline;
      scrollNode.style.outline = '1px solid red';
      setTimeout(() => (scrollNode.style.outline = prev), 2000);
    }
  }

  function apply() {
    document.querySelectorAll('div[data-testid="stDataFrame"]').forEach((root) => audit(root));
  }

  function observe() {
    const observer = new MutationObserver(() => apply());
    document.querySelectorAll('div[data-testid="stDataFrame"]').forEach((root) => {
      observer.observe(root, { childList: true, subtree: true });
    });
  }

  window.__stickyAudit__ = apply;

  window.addEventListener('load', () => {
    apply();
    observe();
  });
})();
