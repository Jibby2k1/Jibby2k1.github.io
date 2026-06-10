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
    if (reducedMotion) {
      document.querySelectorAll('.reveal').forEach((node) => node.classList.add('visible'));
      return;
    }

    // Geometry sweep instead of IntersectionObserver: IO notifications can be
    // dropped during fast scrolling, permanently leaving content invisible.
    const MARGIN = 200;
    const sweep = () => {
      document.querySelectorAll('.reveal:not(.visible)').forEach((node) => {
        const rect = node.getBoundingClientRect();
        if (rect.top < window.innerHeight + MARGIN && rect.bottom > -MARGIN) {
          node.classList.add('visible');
        }
      });
    };

    let pending = false;
    const scheduleSweep = () => {
      if (pending) return;
      pending = true;
      requestAnimationFrame(() => {
        sweep();
        pending = false;
      });
    };

    window.addEventListener('scroll', scheduleSweep, { passive: true });
    window.addEventListener('resize', scheduleSweep, { passive: true });
    window.addEventListener('load', scheduleSweep);

    // Last-resort safety net: nothing stays hidden for more than a few seconds.
    window.addEventListener('load', () => {
      setTimeout(() => {
        document.querySelectorAll('.reveal:not(.visible)').forEach((node) => node.classList.add('visible'));
      }, 5000);
    });

    sweep();
    window.SiteUI = Object.assign(window.SiteUI || {}, { refreshReveal: sweep });
  }

  function initYear() {
    const node = document.getElementById('year');
    if (node) node.textContent = new Date().getFullYear();
  }

  function initPrint() {
    document.querySelectorAll('[data-print]').forEach((button) => {
      button.addEventListener('click', () => window.print());
    });
  }

  function initLightbox() {
    const triggers = [...document.querySelectorAll('[data-lightbox]')];
    if (!triggers.length) return;

    const overlay = document.createElement('div');
    overlay.className = 'lightbox';
    overlay.setAttribute('role', 'dialog');
    overlay.setAttribute('aria-modal', 'true');
    overlay.setAttribute('aria-label', 'Photo viewer');
    overlay.innerHTML = `
      <button class="lightbox-btn lightbox-close" type="button" aria-label="Close viewer">✕</button>
      <button class="lightbox-btn lightbox-prev" type="button" aria-label="Previous photo">‹</button>
      <figure class="lightbox-figure">
        <img class="lightbox-img" alt="" />
        <figcaption class="lightbox-caption"></figcaption>
      </figure>
      <button class="lightbox-btn lightbox-next" type="button" aria-label="Next photo">›</button>`;
    document.body.appendChild(overlay);

    const img = overlay.querySelector('.lightbox-img');
    const caption = overlay.querySelector('.lightbox-caption');
    const closeBtn = overlay.querySelector('.lightbox-close');
    let current = 0;
    let lastFocused = null;

    const render = () => {
      const trigger = triggers[current];
      img.src = trigger.dataset.full;
      img.alt = trigger.dataset.title || '';
      const title = trigger.dataset.title || '';
      const text = trigger.dataset.caption || '';
      caption.innerHTML = '';
      const strong = document.createElement('strong');
      strong.textContent = title;
      caption.appendChild(strong);
      if (text) caption.appendChild(document.createTextNode(` — ${text}`));
      caption.appendChild(document.createTextNode(` (${current + 1}/${triggers.length})`));
    };

    const open = (index) => {
      current = index;
      lastFocused = document.activeElement;
      render();
      overlay.classList.add('open');
      document.body.style.overflow = 'hidden';
      closeBtn.focus();
    };

    const close = () => {
      overlay.classList.remove('open');
      img.src = '';
      document.body.style.overflow = '';
      if (lastFocused) lastFocused.focus();
    };

    const step = (delta) => {
      current = (current + delta + triggers.length) % triggers.length;
      render();
    };

    triggers.forEach((trigger, index) => {
      trigger.addEventListener('click', () => open(index));
    });
    closeBtn.addEventListener('click', close);
    overlay.querySelector('.lightbox-prev').addEventListener('click', () => step(-1));
    overlay.querySelector('.lightbox-next').addEventListener('click', () => step(1));
    overlay.addEventListener('click', (event) => {
      if (event.target === overlay) close();
    });

    window.addEventListener('keydown', (event) => {
      if (!overlay.classList.contains('open')) return;
      if (event.key === 'Escape') close();
      else if (event.key === 'ArrowLeft') step(-1);
      else if (event.key === 'ArrowRight') step(1);
      else if (event.key === 'Tab') {
        const focusable = [...overlay.querySelectorAll('button')];
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first.focus();
        }
      }
    });
  }

  function initTheme() {
    const buttons = document.querySelectorAll('[data-theme-toggle]');
    if (!buttons.length) return;

    const isDark = () => document.documentElement.getAttribute('data-theme') === 'dark';
    const sync = () => {
      const label = isDark() ? 'Switch to light theme' : 'Switch to dark theme';
      buttons.forEach((button) => button.setAttribute('aria-label', label));
    };

    buttons.forEach((button) => {
      button.addEventListener('click', () => {
        const next = isDark() ? 'light' : 'dark';
        if (next === 'dark') document.documentElement.setAttribute('data-theme', 'dark');
        else document.documentElement.removeAttribute('data-theme');
        try { localStorage.setItem('theme', next); } catch { /* private mode */ }
        sync();
      });
    });

    sync();
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
  initPrint();
  initTheme();
  initLightbox();
  initFilters();
})();
