(async () => {
  async function loadPartial(id, url) {
    const el = document.getElementById(id);
    if (!el) return;
    const res = await fetch(url, { cache: 'no-cache' });
    if (!res.ok) throw new Error(`Failed to load ${url}`);
    el.innerHTML = await res.text();
  }

  function setActiveNav() {
    const path = location.pathname.split('/').pop() || 'index.html';
    document.querySelectorAll('.navlink[data-route]').forEach(a => {
      const route = a.getAttribute('data-route');
      const isActive = route === path;
      a.classList.toggle('active', isActive);
      if (isActive) a.setAttribute('aria-current', 'page');
      else a.removeAttribute('aria-current');
    });
  }

  function initBurger() {
    const burger = document.getElementById('burger');
    const nav = document.getElementById('navlinks');
    if (!burger || !nav) return;

    const syncLabel = (open) => {
      burger.setAttribute('aria-expanded', open ? 'true' : 'false');
      burger.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
    };

    const close = () => {
      nav.classList.remove('open');
      syncLabel(false);
    };

    burger.addEventListener('click', () => {
      const open = nav.classList.toggle('open');
      syncLabel(open);
    });

    document.addEventListener('click', (e) => {
      if (!nav.classList.contains('open')) return;
      if (nav.contains(e.target) || burger.contains(e.target)) return;
      close();
    });

    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') close();
    });

    nav.querySelectorAll('a').forEach((link) => {
      link.addEventListener('click', close);
    });

    window.addEventListener('resize', () => {
      if (window.innerWidth > 980) close();
    });

    syncLabel(false);
  }

  function initReveal() {
    const reducedMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const seen = new WeakSet();
    let obs = null;

    const revealNow = (el) => {
      if (!el || !el.classList || !el.classList.contains('reveal')) return;
      el.classList.add('visible');
      if (obs) obs.unobserve(el);
    };

    const observeEl = (el) => {
      if (!(el instanceof Element) || !el.classList.contains('reveal') || seen.has(el)) return;
      seen.add(el);
      if (reducedMotion || !('IntersectionObserver' in window)) {
        revealNow(el);
        return;
      }
      obs.observe(el);
    };

    const scan = (root = document) => {
      if (root instanceof Element && root.classList.contains('reveal')) observeEl(root);
      if (root.querySelectorAll) root.querySelectorAll('.reveal').forEach(observeEl);
    };

    if (!reducedMotion && 'IntersectionObserver' in window) {
      obs = new IntersectionObserver((entries) => {
        entries.forEach((ent) => {
          if (ent.isIntersecting) revealNow(ent.target);
        });
      }, { threshold: 0.12 });
    }

    scan(document);

    if ('MutationObserver' in window && document.body) {
      const mo = new MutationObserver((entries) => {
        entries.forEach((entry) => {
          entry.addedNodes.forEach((node) => {
            if (node instanceof Element) scan(node);
          });
        });
      });
      mo.observe(document.body, { childList: true, subtree: true });
    }

    window.SiteUI = Object.assign(window.SiteUI || {}, {
      refreshReveal: scan,
    });
  }

  function initYear() {
    const y = document.getElementById('year');
    if (y) y.textContent = new Date().getFullYear();
  }

  initReveal();

  try {
    await Promise.all([
      loadPartial('site-nav', 'partials/nav.html'),
      loadPartial('site-footer', 'partials/footer.html'),
    ]);
    setActiveNav();
    initBurger();
    initYear();
  } catch (e) {
    // If partial fetch fails, fail gracefully; page content still renders.
    console.warn(e);
  }

  // The shared archive theme uses a static background, so this hook is intentionally quiet.
})();
