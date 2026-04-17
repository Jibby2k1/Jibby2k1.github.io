(() => {
  function currentRoute() {
    const path = location.pathname;
    if (path.includes('/projects/')) return 'research.html';
    if (path.includes('/notes/')) return 'about.html';
    const file = path.split('/').pop() || 'index.html';
    if (file === 'publications.html') return 'research.html';
    if (['blog.html', 'awards.html', 'photography.html'].includes(file)) return 'about.html';
    return file;
  }

  function setActiveNav() {
    const path = currentRoute();
    document.querySelectorAll('.navlink[data-route]').forEach((link) => {
      const isActive = link.getAttribute('data-route') === path;
      link.classList.toggle('active', isActive);
      if (isActive) link.setAttribute('aria-current', 'page');
      else link.removeAttribute('aria-current');
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

    document.addEventListener('click', (event) => {
      if (!nav.classList.contains('open')) return;
      if (nav.contains(event.target) || burger.contains(event.target)) return;
      close();
    });

    window.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') close();
    });

    nav.querySelectorAll('a').forEach((link) => link.addEventListener('click', close));
    window.addEventListener('resize', () => {
      if (window.innerWidth > 980) close();
    });

    syncLabel(false);
  }

  function initReveal() {
    const reducedMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reducedMotion || !('IntersectionObserver' in window)) {
      document.querySelectorAll('.reveal').forEach((node) => node.classList.add('visible'));
      return;
    }

    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.12 });

    const scan = (root = document) => {
      if (root instanceof Element && root.classList.contains('reveal')) observer.observe(root);
      if (root.querySelectorAll) root.querySelectorAll('.reveal').forEach((node) => observer.observe(node));
    };

    scan(document);
    window.SiteUI = Object.assign(window.SiteUI || {}, { refreshReveal: scan });
  }

  function initYear() {
    const node = document.getElementById('year');
    if (node) node.textContent = new Date().getFullYear();
  }

  function initFilters() {
    document.querySelectorAll('[data-filter-root]').forEach((root) => {
      const searchInput = root.querySelector('[data-filter-search]');
      const buttons = [...root.querySelectorAll('[data-filter-button]')];
      const grid = root.parentElement.querySelector('[data-filter-grid]');
      const count = root.querySelector('[data-filter-count]');
      const empty = root.querySelector('[data-filter-empty]');
      if (!grid) return;

      let active = buttons.find((button) => button.classList.contains('active'))?.dataset.filterValue || 'all';

      const apply = () => {
        const query = (searchInput?.value || '').trim().toLowerCase();
        let visible = 0;

        grid.querySelectorAll('[data-filter-item]').forEach((item) => {
          const tags = (item.dataset.tags || '').split(',').map((value) => value.trim()).filter(Boolean);
          const matchesTag = active === 'all' || tags.includes(active);
          const matchesQuery = !query || (item.dataset.search || '').includes(query);
          const show = matchesTag && matchesQuery;
          item.classList.toggle('is-hidden', !show);
          if (show) visible += 1;
        });

        if (count) count.textContent = `${visible} item${visible === 1 ? '' : 's'}`;
        if (empty) empty.hidden = visible !== 0;
      };

      buttons.forEach((button) => {
        button.addEventListener('click', () => {
          active = button.dataset.filterValue || 'all';
          buttons.forEach((peer) => peer.classList.toggle('active', peer === button));
          apply();
        });
      });

      searchInput?.addEventListener('input', apply);
      apply();
    });
  }

  initReveal();
  setActiveNav();
  initBurger();
  initYear();
  initFilters();
})();
