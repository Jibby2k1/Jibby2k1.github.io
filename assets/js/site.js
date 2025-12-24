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

    const close = () => {
      nav.classList.remove('open');
      burger.setAttribute('aria-expanded', 'false');
    };

    burger.addEventListener('click', () => {
      const open = nav.classList.toggle('open');
      burger.setAttribute('aria-expanded', open ? 'true' : 'false');
    });

    document.addEventListener('click', (e) => {
      if (!nav.classList.contains('open')) return;
      if (nav.contains(e.target) || burger.contains(e.target)) return;
      close();
    });

    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') close();
    });
  }

  function initReveal() {
    const els = document.querySelectorAll('.reveal');
    if (!('IntersectionObserver' in window)) {
      els.forEach(el => el.classList.add('visible'));
      return;
    }
    const obs = new IntersectionObserver((entries) => {
      entries.forEach(ent => {
        if (ent.isIntersecting) ent.target.classList.add('visible');
      });
    }, { threshold: 0.12 });
    els.forEach(el => obs.observe(el));
  }

  function initYear() {
    const y = document.getElementById('year');
    if (y) y.textContent = new Date().getFullYear();
  }

  try {
    await loadPartial('site-nav', 'partials/nav.html');
    await loadPartial('site-footer', 'partials/footer.html');
    setActiveNav();
    initBurger();
    initYear();
  } catch (e) {
    // If partial fetch fails, fail gracefully; page content still renders.
    console.warn(e);
  }

  // Background animation (canvas)
  try {
    if (window.SiteBackground && window.SiteBackground.startBackground) {
      window.SiteBackground.startBackground();
    }
  } catch (e) {
    console.warn(e);
  }

  initReveal();
})();
