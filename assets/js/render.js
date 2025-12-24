(() => {
  const cache = new Map();

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

      const allTags = ['all', ...Array.from(tags).sort((a,b)=>a.localeCompare(b))];

      let active = defaultFilter;

      function drawFilters() {
        filtersEl.innerHTML = '';
        allTags.forEach(t => {
          const btn = el('button', { class: `filter-btn${t===active?' active':''}`, type: 'button' }, [t.toUpperCase()]);
          btn.addEventListener('click', () => {
            active = t;
            drawFilters();
            drawGrid();
          });
          filtersEl.appendChild(btn);
        });
      }

      function card(it) {
        const pills = el('div', { class: 'pill-row' }, (it.pills || []).map(p => el('span', { class: 'pill' }, [p])));
        const meta = it.meta ? el('div', { class: 'meta' }, [it.meta]) : null;

        const children = [
          el('h3', {}, [it.title || 'Untitled']),
          el('p', {}, [it.desc || '']),
        ];
        if (it.pills && it.pills.length) children.push(pills);
        if (meta) children.push(meta);

        if (it.links && it.links.length) {
          const row = el('div', { class: 'pill-row' }, it.links.map(l =>
            el('a', { class: 'pill', href: l.href, target: '_blank', rel: 'noreferrer' }, [l.label])
          ));
          children.push(row);
        }

        return el('article', { class: 'card reveal' }, children);
      }

      function drawGrid() {
        gridEl.innerHTML = '';
        const filtered = items.filter(it => active === 'all' ? true : (it.tags || []).includes(active));
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
        const summary = el('summary', {}, [
          el('div', { class: 'post-title' }, [it.title || 'Untitled']),
          el('div', { class: 'post-meta' }, [`${it.date || ''}${it.tag ? ' · ' + it.tag : ''}`]),
          it.summary ? el('div', { class: 'post-summary' }, [it.summary]) : null,
        ]);
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
