(() => {
  const cache = new Map();
  const PLACEHOLDER_IMG = 'assets/img/placeholder.svg';
  const LABEL_MAP = {
    all: 'All Projects',
    eeg: 'EEG',
    ml: 'ML',
    nlp: 'NLP',
    gpu: 'GPU',
    sps: 'SPS',
    cnel: 'CNEL',
  };

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

  function inferSrcFromDataUrl(dataUrl) {
    const u = String(dataUrl || '').toLowerCase();
    if (u.includes('cnel')) return 'cnel';
    if (u.includes('sps')) return 'sps';
    return '';
  }

  function projectHref(src, slug) {
    if (!src || !slug) return '';
    return `project.html?src=${encodeURIComponent(src)}&slug=${encodeURIComponent(slug)}`;
  }

  function prettyLabel(value) {
    const raw = String(value || '').trim();
    const lower = raw.toLowerCase();
    if (LABEL_MAP[lower]) return LABEL_MAP[lower];
    return lower
      .split(/[-_\s]+/)
      .filter(Boolean)
      .map((part) => LABEL_MAP[part] || `${part.charAt(0).toUpperCase()}${part.slice(1)}`)
      .join(' ');
  }

  function notifyContentRendered(root) {
    if (window.SiteUI && typeof window.SiteUI.refreshReveal === 'function') {
      window.SiteUI.refreshReveal(root || document);
    }
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

  // ---- Modal (used for "Read More" on project cards) ----
  let _modal = null;

  function ensureModal() {
    if (_modal) return _modal;

    const overlay = el('div', {
      class: 'modal-overlay',
      role: 'dialog',
      'aria-modal': 'true',
      'aria-hidden': 'true',
    }, []);

    // Inline style fallbacks (in case CSS is missing or stale-cached).
    overlay.style.position = 'fixed';
    overlay.style.inset = '0';
    overlay.style.display = 'none';
    overlay.style.alignItems = 'center';
    overlay.style.justifyContent = 'center';
    overlay.style.padding = '24px';
    overlay.style.background = 'rgba(0,0,0,0.55)';
    overlay.style.backdropFilter = 'blur(6px)';
    overlay.style.zIndex = '9999';

    const dialog = el('div', { class: 'modal' }, []);
    dialog.style.width = 'min(920px, 100%)';
    dialog.style.maxHeight = 'min(82vh, 760px)';
    dialog.style.overflow = 'hidden';
    dialog.style.borderRadius = '22px';
    dialog.style.border = '1px solid rgba(255,255,255,0.12)';
    dialog.style.background = 'rgba(15,15,18,0.92)';
    dialog.style.boxShadow = '0 20px 60px rgba(0,0,0,0.55)';

    const header = el('div', { class: 'modal-header' }, []);
    header.style.display = 'flex';
    header.style.alignItems = 'center';
    header.style.justifyContent = 'space-between';
    header.style.gap = '16px';
    header.style.padding = '18px 18px 10px 18px';
    header.style.borderBottom = '1px solid rgba(255,255,255,0.10)';

    const title = el('h3', { class: 'modal-title' }, ['']);
    const closeBtn = el('button', { class: 'modal-close', type: 'button', 'aria-label': 'Close' }, ['×']);
    closeBtn.style.width = '38px';
    closeBtn.style.height = '38px';
    closeBtn.style.borderRadius = '12px';
    closeBtn.style.border = '1px solid rgba(255,255,255,0.14)';
    closeBtn.style.background = 'rgba(255,255,255,0.06)';
    closeBtn.style.color = '#fff';
    closeBtn.style.cursor = 'pointer';
    closeBtn.style.fontSize = '22px';
    closeBtn.style.lineHeight = '1';
    closeBtn.style.display = 'flex';
    closeBtn.style.alignItems = 'center';
    closeBtn.style.justifyContent = 'center';

    const body = el('div', { class: 'modal-body' }, []);
    body.style.padding = '18px';
    body.style.overflow = 'auto';
    body.style.maxHeight = 'calc(min(82vh, 760px) - 64px)';

    header.appendChild(title);
    header.appendChild(closeBtn);
    dialog.appendChild(header);
    dialog.appendChild(body);
    overlay.appendChild(dialog);
    document.body.appendChild(overlay);

    let lastFocus = null;

    function close() {
      overlay.classList.remove('open');
      overlay.style.display = 'none';
      overlay.setAttribute('aria-hidden', 'true');
      document.body.classList.remove('modal-open');
      body.innerHTML = '';
      if (lastFocus && typeof lastFocus.focus === 'function') lastFocus.focus();
      lastFocus = null;
    }

    function open({ titleText = '', contentNodes = [] } = {}) {
      lastFocus = document.activeElement;
      title.textContent = titleText;
      body.innerHTML = '';
      (contentNodes || []).forEach(n => n && body.appendChild(n));
      overlay.style.display = 'flex';
      overlay.classList.add('open');
      overlay.setAttribute('aria-hidden', 'false');
      document.body.classList.add('modal-open');
      closeBtn.focus();
    }

    overlay.addEventListener('click', (e) => {
      // Close when clicking outside the dialog container.
      if (e.target === overlay) close();
    });

    closeBtn.addEventListener('click', close);

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && overlay.classList.contains('open')) close();
    });

    _modal = { open, close };
    return _modal;
  }

  function projectModalContent(it) {
    const nodes = [];
    if (!it) return nodes;
    const detailHref = projectHref(it._srcKey, it.slug);

    if (it.subtitle) nodes.push(el('p', { class: 'modal-subtitle' }, [it.subtitle]));
    if (it.desc) nodes.push(el('p', {}, [it.desc]));

    const d = it.details || {};
    if (d.overview) {
      nodes.push(el('h4', {}, ['Overview']));
      nodes.push(el('p', {}, [d.overview]));
    }

    if (Array.isArray(d.what_i_built) && d.what_i_built.length) {
      nodes.push(el('h4', {}, ['What I built']));
      nodes.push(el('ul', { class: 'modal-list' }, d.what_i_built.map(x => el('li', {}, [String(x)]))));
    }

    if (Array.isArray(d.stack) && d.stack.length) {
      nodes.push(el('h4', {}, ['Stack']));
      nodes.push(el('ul', { class: 'modal-list' }, d.stack.map(x => el('li', {}, [String(x)]))));
    }

    if (Array.isArray(d.collaborators) && d.collaborators.length) {
      nodes.push(el('h4', {}, ['Collaborators']));
      nodes.push(el('ul', { class: 'modal-list' }, d.collaborators.map(x => el('li', {}, [String(x)]))));
    }

    if (d.status) {
      nodes.push(el('h4', {}, ['Status']));
      nodes.push(el('p', {}, [String(d.status)]));
    }

    if (Array.isArray(d.glossary) && d.glossary.length) {
      nodes.push(el('h4', {}, ['Glossary']));
      const dlKids = [];
      d.glossary.forEach(g => {
        if (!g) return;
        const term = g.term ? String(g.term) : '';
        const def = g.definition ? String(g.definition) : '';
        if (!term && !def) return;
        dlKids.push(el('dt', {}, [term]));
        dlKids.push(el('dd', {}, [def]));
      });
      nodes.push(el('dl', { class: 'modal-dl' }, dlKids));
    }

    if (it.links && it.links.length) {
      nodes.push(el('h4', {}, ['Links']));
      nodes.push(el('div', { class: 'pill-row' }, it.links.map(l =>
        el('a', { class: 'pill', href: l.href, target: '_blank', rel: 'noreferrer' }, [l.label])
      )));
    }

    const primary = (it.links || []).find(l => l && l.href);
    const modalCtas = [];

    if (detailHref) {
      modalCtas.push(el('a', { class: 'btn small', href: detailHref }, ['Open detail page']));
    }

    if (primary) {
      modalCtas.push(
        el(
          'a',
          { class: 'btn small primary', href: primary.href, target: '_blank', rel: 'noreferrer' },
          ['Take me there']
        )
      );
    }

    if (modalCtas.length) {
      nodes.push(el('div', { class: 'cta-row' }, modalCtas));
    }

    return nodes;
  }

  function openProjectModal(it) {
    const modal = ensureModal();
    modal.open({
      titleText: (it && it.title) ? it.title : 'Project details',
      contentNodes: projectModalContent(it),
    });
  }



  function renderProjects({ mountId, dataUrl, defaultFilter = 'all' }) {
    const mount = document.getElementById(mountId);
    if (!mount) return;

    const filtersEl = mount.querySelector('[data-project-filters]');
    const gridEl = mount.querySelector('[data-project-grid]');
    if (!filtersEl || !gridEl) return;

    loadJSON(dataUrl).then((data) => {
      const sourceKey = inferSrcFromDataUrl(dataUrl);
      const items = (data.items || []).map(it => ({ ...it, _srcKey: sourceKey }));
      const tags = new Set();
      items.forEach(it => (it.tags || []).forEach(t => tags.add(t)));

      const allTags = ['all', ...Array.from(tags).sort((a, b) => a.localeCompare(b))];
      let active = defaultFilter;
      let query = '';
      let countEl = null;

      function drawFilters() {
        filtersEl.innerHTML = '';

        filtersEl.appendChild(el('div', { class: 'filter-title' }, ['Filters']));
        filtersEl.appendChild(el('p', { class: 'filter-hint' }, ['Search titles, summaries, and tags.']));

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

        countEl = el('div', { class: 'filter-count', 'aria-live': 'polite' }, ['']);
        filtersEl.appendChild(countEl);

        allTags.forEach(t => {
          const btn = el('button', { class: `filter-btn${t === active ? ' active' : ''}`, type: 'button' }, [prettyLabel(t)]);
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

        const frontChildren = [
          imgNode({ src: it.img, alt: it.imgAlt || it.title || '', className: (it.imgClass || 'card-img'), fit: (it.imgFit || null), aspect: (it.imgAspect || null) }),
          el('h3', {}, [it.title || 'Untitled']),
          it.meta ? el('div', { class: 'meta' }, [it.meta]) : null,
          el('div', { class: 'flip-hint' }, ['Click to flip'])
        ];

        const front = el('div', { class: 'flip-face flip-front' }, frontChildren);

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

        const ctas = [];
        const hasMore = Boolean(it.details && Object.keys(it.details).length) || Boolean(it.subtitle);
        const detailHref = projectHref(it._srcKey, it.slug);

        if (hasMore) {
          const readMoreBtn = el('button', { class: 'btn small primary', type: 'button', 'data-no-flip': 'true' }, ['Read More']);
          readMoreBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            openProjectModal(it);
          });
          ctas.push(readMoreBtn);
        }

        if (detailHref) {
          ctas.push(el('a', { class: 'btn small', href: detailHref, 'data-no-flip': 'true' }, ['Open Page']));
        }

        if (ctas.length) {
          children.push(el('div', { class: 'cta-row' }, ctas));
        }

        const back = el('div', { class: 'flip-face flip-back' }, children);
        const inner = el('div', { class: 'flip-inner' }, [front, back]);
        root.appendChild(inner);

        root.addEventListener('click', (e) => {
          // Don't flip when user clicks a link.
          if (e.target && e.target.closest && e.target.closest('a, button, [data-no-flip]')) return;
          toggleFlip(root);
        });

        root.addEventListener('keydown', (e) => {
          if (e.target && e.target.closest && e.target.closest('a, button, input, textarea, select')) return;
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
            it.subtitle || '',
            it.meta || '',
            (it.details && it.details.overview) ? it.details.overview : '',
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
          gridEl.appendChild(el('div', { class: 'notice' }, ['No projects match this filter yet. Try clearing the search or choosing a different tag.']));
          notifyContentRendered(gridEl);
          return;
        }
        filtered.forEach(it => gridEl.appendChild(card(it)));
        notifyContentRendered(gridEl);
      }

      drawFilters();
      drawGrid();
    }).catch((e) => {
      gridEl.innerHTML = '';
      gridEl.appendChild(el('div', { class: 'notice' }, [`Failed to load project data (${dataUrl}).`]));
      notifyContentRendered(gridEl);
      console.warn(e);
    });
  }

  function renderBlog({ mountId, dataUrl, limit = null }) {
    const mount = document.getElementById(mountId);
    if (!mount) return;

    loadJSON(dataUrl).then((data) => {
      const items = (data.items || []).slice(0, limit || undefined);
      mount.innerHTML = '';
      if (!items.length) {
        mount.appendChild(el('div', { class: 'notice' }, ['No blog posts yet.']));
        notifyContentRendered(mount);
        return;
      }

      items.forEach(it => {
        const cardChildren = [
          it.img ? imgNode({ src: it.img, alt: it.imgAlt || it.title || '', className: (it.imgClass || 'blog-media'), fit: (it.imgFit || 'cover'), aspect: (it.imgAspect || null) }) : null,
          el('div', { class: 'blog-copy' }, [
            el('div', { class: 'blog-meta' }, [`${it.date || ''}${it.tag ? ' · ' + it.tag : ''}`]),
            el('h3', { class: 'blog-title' }, [it.title || 'Untitled']),
            it.summary ? el('p', { class: 'blog-summary' }, [it.summary]) : null,
          ]),
        ];

        if (it.content) {
          const details = el('details', { class: 'blog-details' }, [
            el('summary', {}, ['Read note']),
            el('div', { class: 'blog-content' }, [it.content]),
          ]);
          cardChildren.push(details);
        }

        mount.appendChild(el('article', { class: 'card blog-card reveal' }, cardChildren.filter(Boolean)));
      });
      notifyContentRendered(mount);
    }).catch((e) => {
      mount.innerHTML = '';
      mount.appendChild(el('div', { class: 'notice' }, [`Failed to load blog data (${dataUrl}).`]));
      notifyContentRendered(mount);
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
        notifyContentRendered(mount);
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
      notifyContentRendered(mount);
    }).catch((e) => {
      mount.innerHTML = '';
      mount.appendChild(el('div', { class: 'notice' }, [`Failed to load awards data (${dataUrl}).`]));
      notifyContentRendered(mount);
      console.warn(e);
    });
  }

  function renderHomeFeatured({ mountId, dataUrl, limit = 3 }) {
    const mount = document.getElementById(mountId);
    if (!mount) return;

    loadJSON(dataUrl).then((data) => {
      const sourceKey = inferSrcFromDataUrl(dataUrl);
      const items = (data.items || [])
        .map(it => ({ ...it, _srcKey: sourceKey }))
        .filter(it => it.featured);
      const picked = items.slice(0, limit);

      mount.innerHTML = '';
      if (!picked.length) {
        mount.appendChild(el('div', { class: 'notice' }, ['No featured items yet.']));
        notifyContentRendered(mount);
        return;
      }

      picked.forEach(it => {
        const kids = [
          it.img ? imgNode({ src: it.img, alt: it.imgAlt || it.title || '', className: (it.imgClass || 'card-img'), fit: (it.imgFit || null), aspect: (it.imgAspect || null) }) : null,
          el('h3', {}, [it.title || 'Untitled']),
          el('p', {}, [it.desc || '']),
          it.meta ? el('div', { class: 'meta' }, [it.meta]) : null,
        ];

        const hasMore = Boolean(it.details && Object.keys(it.details).length) || Boolean(it.subtitle) || Boolean(it.desc);
        const detailHref = projectHref(it._srcKey, it.slug);

        if (hasMore) {
          const readMoreBtn = el('button', { class: 'btn small primary', type: 'button' }, ['Read More']);
          readMoreBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            openProjectModal(it);
          });
          kids.push(el('div', { class: 'cta-row' }, [readMoreBtn]));
        }

        if (detailHref) {
          const ctaRow = kids.find(node => node && node.className === 'cta-row');
          const link = el('a', { class: 'btn small', href: detailHref }, ['Open Page']);
          if (ctaRow) ctaRow.appendChild(link);
          else kids.push(el('div', { class: 'cta-row' }, [link]));
        }

        mount.appendChild(el('article', { class: 'card reveal' }, kids));
      });
      notifyContentRendered(mount);
    }).catch((e) => {
      mount.innerHTML = '';
      mount.appendChild(el('div', { class: 'notice' }, [`Failed to load featured data (${dataUrl}).`]));
      notifyContentRendered(mount);
      console.warn(e);
    });
  }

  function renderPhotography({ mountId, dataUrl }) {
    const mount = document.getElementById(mountId);
    if (!mount) return;

    loadJSON(dataUrl).then((data) => {
      const items = data.items || [];
      mount.innerHTML = '';

      if (!items.length) {
        mount.appendChild(el('article', { class: 'card photo-empty reveal' }, [
          el('h3', {}, [data.emptyTitle || 'No photographs published yet']),
          el('p', {}, [data.emptyText || 'This gallery is ready for published images.']),
          el('div', { class: 'meta' }, ['Once you add entries to data/photography.json, they will appear here automatically.']),
        ]));
        notifyContentRendered(mount);
        return;
      }

      items.forEach((it) => {
        const media = imgNode({
          src: it.img,
          alt: it.imgAlt || it.title || 'Photograph',
          className: 'photo-media',
          fit: it.imgFit || 'cover',
          aspect: it.imgAspect || it.aspect || null,
        });

        const mediaNode = it.link
          ? el('a', { class: 'photo-media-link', href: it.link, target: '_blank', rel: 'noreferrer' }, [media])
          : media;

        const metaParts = [it.location, it.year].filter(Boolean).join(' · ');

        mount.appendChild(el('figure', { class: 'photo-card reveal' }, [
          mediaNode,
          el('figcaption', { class: 'photo-copy' }, [
            el('div', { class: 'photo-category' }, [it.collection || 'Photography']),
            el('h3', {}, [it.title || 'Untitled']),
            it.caption ? el('p', { class: 'photo-caption' }, [it.caption]) : null,
            metaParts ? el('div', { class: 'photo-meta' }, [metaParts]) : null,
          ]),
        ]));
      });

      notifyContentRendered(mount);
    }).catch((e) => {
      mount.innerHTML = '';
      mount.appendChild(el('div', { class: 'notice' }, [`Failed to load photography data (${dataUrl}).`]));
      notifyContentRendered(mount);
      console.warn(e);
    });
  }

  window.SiteRender = { renderProjects, renderBlog, renderAwards, renderHomeFeatured, renderPhotography };
})();
