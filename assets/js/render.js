(() => {
  const cache = new Map();
  const PLACEHOLDER_IMG = 'assets/img/placeholder.svg';

  async function loadJSON(url) {
    if (cache.has(url)) return cache.get(url);
    const res = await fetch(url, { cache: 'no-cache' });
    if (!res.ok) throw new Error(`Failed to load ${url}`);
    const data = await res.json();
    cache.set(url, data);
    return data;
  }

  function el(tag, attrs = {}, children = []) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === 'class') node.className = v;
      else if (k.startsWith('on') && typeof v === 'function') node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v);
    }
    for (const c of children) {
      if (c == null) continue;
      if (typeof c === 'string') node.appendChild(document.createTextNode(c));
      else node.appendChild(c);
    }
    return node;
  }

  function safeImgSrc(src) {
    return src ? src : PLACEHOLDER_IMG;
  }

  function imgNode({ src, alt = '', className = 'card-img', fit = null, aspect = null } = {}) {
    let cls = className || 'card-img';

    // Prefer explicit fit control (useful for diagrams/logos vs photos).
    if (fit === 'cover' && !cls.includes('fit-cover')) cls += ' fit-cover';
    if (fit === 'contain' && !cls.includes('fit-contain')) cls += ' fit-contain';

    const attrs = {
      class: cls.trim(),
      src: safeImgSrc(src),
      alt,
      loading: 'lazy',
      decoding: 'async',
    };
    if (aspect) attrs.style = `aspect-ratio: ${aspect};`;

    const img = el('img', attrs);
    img.addEventListener('error', () => {
      // avoid infinite loops
      if (img.getAttribute('src') !== PLACEHOLDER_IMG) img.setAttribute('src', PLACEHOLDER_IMG);
    });
    return img;
  }

  function toggleFlip(node) {
    const next = !node.classList.contains('is-flipped');
    node.classList.toggle('is-flipped', next);
    node.setAttribute('aria-pressed', next ? 'true' : 'false');
  }

  function renderProjects({ mountId, dataUrl, defaultFilter = 'all' }) {
    const mount = document.getElementById(mountId);
    if (!mount) return;

    const filtersEl = mount.querySelector('[data-project-filters]');
    const gridEl = mount.querySelector('[data-project-grid]');
    if (!filtersEl || !gridEl) return;

    loadJSON(dataUrl).then((data) => {
      const items = data.items || [];
      const tags = new Set();
      items.forEach(it => (it.tags || []).forEach(t => tags.add(t)));

      const allTags = ['all', ...Array.from(tags).sort((a, b) => a.localeCompare(b))];
      let active = defaultFilter;
      let query = '';
      let countEl = null;

      function drawFilters() {
        filtersEl.innerHTML = '';

        // Header + lightweight search so the sidebar feels like an actual directory.
        filtersEl.appendChild(el('div', { class: 'filter-title' }, ['Filters']));

        const input = el('input', {
          class: 'filter-input',
          type: 'search',
          placeholder: 'Search projects…',
          'aria-label': 'Search projects',
        });
        input.value = query;
        input.addEventListener('input', () => {
          query = (input.value || '').trim().toLowerCase();
          drawGrid();
        });
        filtersEl.appendChild(el('div', { class: 'filter-search' }, [input]));

        countEl = el('div', { class: 'filter-count' }, ['']);
        filtersEl.appendChild(countEl);

        allTags.forEach(t => {
          const btn = el('button', { class: `filter-btn${t === active ? ' active' : ''}`, type: 'button' }, [t.toUpperCase()]);
          btn.addEventListener('click', () => {
            active = t;
            drawFilters();
            drawGrid();
          });
          filtersEl.appendChild(btn);
        });
      }

      function card(it) {
        // Flip card: front shows image + title; back shows description + metadata.
        const root = el('article', {
          class: 'card flip-card reveal',
          tabindex: '0',
          role: 'button',
          'aria-pressed': 'false',
        }, []);

        const front = el('div', { class: 'flip-face flip-front' }, [
          imgNode({ src: it.img, alt: it.imgAlt || it.title || '', className: (it.imgClass || 'card-img'), fit: (it.imgFit || null), aspect: (it.imgAspect || null) }),
          el('h3', {}, [it.title || 'Untitled']),
          it.meta ? el('div', { class: 'meta' }, [it.meta]) : null,
          el('div', { class: 'flip-hint' }, ['Click to flip'])
        ]);

        const children = [
          el('h3', {}, [it.title || 'Untitled']),
          el('p', {}, [it.desc || '']),
        ];

        if (it.pills && it.pills.length) {
          children.push(el('div', { class: 'pill-row' }, (it.pills || []).map(p => el('span', { class: 'pill' }, [p]))));
        }
        if (it.meta) children.push(el('div', { class: 'meta' }, [it.meta]));

        if (it.links && it.links.length) {
          children.push(el('div', { class: 'pill-row' }, it.links.map(l =>
            el('a', { class: 'pill', href: l.href, target: '_blank', rel: 'noreferrer' }, [l.label])
          )));
        }

        const back = el('div', { class: 'flip-face flip-back' }, children);
        const inner = el('div', { class: 'flip-inner' }, [front, back]);
        root.appendChild(inner);

        root.addEventListener('click', (e) => {
          // Don't flip when user clicks a link.
          if (e.target && e.target.closest && e.target.closest('a')) return;
          toggleFlip(root);
        });

        root.addEventListener('keydown', (e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            toggleFlip(root);
          }
        });

        return root;
      }

      function drawGrid() {
        gridEl.innerHTML = '';
        const filtered = items.filter(it => {
          const tagOk = (active === 'all') ? true : (it.tags || []).includes(active);
          if (!tagOk) return false;

          if (!query) return true;
          const hay = [
            it.title || '',
            it.desc || '',
            it.meta || '',
            (it.pills || []).join(' '),
            (it.tags || []).join(' '),
          ].join(' ').toLowerCase();
          return hay.includes(query);
        });

        if (countEl) {
          const n = filtered.length;
          countEl.textContent = `${n} project${n === 1 ? '' : 's'}`;
        }
        if (!filtered.length) {
          gridEl.appendChild(el('div', { class: 'notice' }, ['No projects match this filter.']));
          return;
        }
        filtered.forEach(it => gridEl.appendChild(card(it)));
      }

      drawFilters();
      drawGrid();
    }).catch((e) => {
      gridEl.innerHTML = '';
      gridEl.appendChild(el('div', { class: 'notice' }, [`Failed to load project data (${dataUrl}).`]));
      console.warn(e);
    });
  }

  function renderBlog({ mountId, dataUrl }) {
    const mount = document.getElementById(mountId);
    if (!mount) return;

    loadJSON(dataUrl).then((data) => {
      const items = data.items || [];
      mount.innerHTML = '';
      if (!items.length) {
        mount.appendChild(el('div', { class: 'notice' }, ['No blog posts yet.']));
        return;
      }

      items.forEach(it => {
        const details = el('details', { class: 'post reveal' }, []);

        const title = el('div', { class: 'post-title' }, [it.title || 'Untitled']);
        const meta = el('div', { class: 'post-meta' }, [`${it.date || ''}${it.tag ? ' · ' + it.tag : ''}`]);
        const summaryText = it.summary ? el('div', { class: 'post-summary' }, [it.summary]) : null;

        const header = el('div', { class: 'post-header' }, [
          it.img ? imgNode({ src: it.img, alt: it.imgAlt || it.title || '', className: (it.imgClass || 'post-img'), fit: (it.imgFit || null), aspect: (it.imgAspect || null) }) : null,
          el('div', { class: 'post-text' }, [title, meta, summaryText])
        ]);

        const summary = el('summary', {}, [header]);
        const content = it.content ? el('div', { class: 'post-content' }, [it.content]) : null;

        details.appendChild(summary);
        if (content) details.appendChild(content);

        mount.appendChild(details);
      });
    }).catch((e) => {
      mount.innerHTML = '';
      mount.appendChild(el('div', { class: 'notice' }, [`Failed to load blog data (${dataUrl}).`]));
      console.warn(e);
    });
  }

  function renderAwards({ mountId, dataUrl }) {
    const mount = document.getElementById(mountId);
    if (!mount) return;

    loadJSON(dataUrl).then((data) => {
      const items = data.items || [];
      mount.innerHTML = '';
      if (!items.length) {
        mount.appendChild(el('div', { class: 'notice' }, ['No awards listed yet.']));
        return;
      }

      items.forEach(it => {
        mount.appendChild(el('article', { class: 'card reveal' }, [
          it.img ? imgNode({ src: it.img, alt: it.imgAlt || it.title || '', className: (it.imgClass || 'card-img'), fit: (it.imgFit || null), aspect: (it.imgAspect || null) }) : null,
          el('h3', {}, [it.title || 'Untitled']),
          el('p', {}, [it.desc || '']),
          it.meta ? el('div', { class: 'meta' }, [it.meta]) : null,
        ]));
      });
    }).catch((e) => {
      mount.innerHTML = '';
      mount.appendChild(el('div', { class: 'notice' }, [`Failed to load awards data (${dataUrl}).`]));
      console.warn(e);
    });
  }

  function renderHomeFeatured({ mountId, dataUrl, limit = 3 }) {
    const mount = document.getElementById(mountId);
    if (!mount) return;

    loadJSON(dataUrl).then((data) => {
      const items = (data.items || []).filter(it => it.featured);
      const picked = items.slice(0, limit);

      mount.innerHTML = '';
      if (!picked.length) {
        mount.appendChild(el('div', { class: 'notice' }, ['No featured items yet.']));
        return;
      }

      picked.forEach(it => {
        mount.appendChild(el('article', { class: 'card reveal' }, [
          it.img ? imgNode({ src: it.img, alt: it.imgAlt || it.title || '', className: (it.imgClass || 'card-img compact'), fit: (it.imgFit || null), aspect: (it.imgAspect || null) }) : null,
          el('h3', {}, [it.title || 'Untitled']),
          el('p', {}, [it.desc || '']),
          it.meta ? el('div', { class: 'meta' }, [it.meta]) : null,
        ]));
      });
    }).catch((e) => {
      mount.innerHTML = '';
      mount.appendChild(el('div', { class: 'notice' }, [`Failed to load featured data (${dataUrl}).`]));
      console.warn(e);
    });
  }

  window.SiteRender = { renderProjects, renderBlog, renderAwards, renderHomeFeatured };
})();
