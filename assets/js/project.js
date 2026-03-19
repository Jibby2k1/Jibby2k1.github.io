(() => {
  const PLACEHOLDER_IMG = 'assets/img/placeholder.svg';
  const SOURCES = {
    cnel: {
      url: 'data/cnel_projects.json',
      kicker: 'Research · CNEL',
      backHref: 'research-cnel.html',
      backLabel: 'Back to CNEL projects',
      crumbLabel: 'CNEL projects',
    },
    sps: {
      url: 'data/sps_projects.json',
      kicker: 'Research · IEEE SPS @ UF',
      backHref: 'research-sps.html',
      backLabel: 'Back to IEEE SPS projects',
      crumbLabel: 'IEEE SPS projects',
    },
  };

  function el(tag, attrs = {}, children = []) {
    const node = document.createElement(tag);
    Object.entries(attrs).forEach(([key, value]) => {
      if (key === 'class') node.className = value;
      else node.setAttribute(key, value);
    });
    children.forEach((child) => {
      if (child == null) return;
      if (typeof child === 'string') node.appendChild(document.createTextNode(child));
      else node.appendChild(child);
    });
    return node;
  }

  function listCard(title, items) {
    if (!Array.isArray(items) || !items.length) return null;
    return el('article', { class: 'card reveal' }, [
      el('h2', { class: 'project-section-title' }, [title]),
      el('ul', {}, items.map((item) => el('li', {}, [String(item)]))),
    ]);
  }

  function kvRow(key, value) {
    return el('div', { class: 'kv-row' }, [
      el('div', { class: 'kv-key' }, [key]),
      el('div', { class: 'kv-val' }, [value]),
    ]);
  }

  function showNotice(message) {
    const notice = document.getElementById('project-notice');
    if (!notice) return;
    notice.style.display = 'block';
    notice.textContent = message;
  }

  function setMetaDescription(text) {
    const meta = document.querySelector('meta[name="description"]');
    if (meta && text) meta.setAttribute('content', text);
  }

  function renderProject(item, sourceKey) {
    const source = SOURCES[sourceKey];
    const details = item.details || {};
    const detailMount = document.getElementById('project-detail');
    const cta = document.getElementById('project-cta');
    const crumbs = document.getElementById('project-breadcrumbs');
    const heroImg = document.getElementById('project-hero-img');

    document.title = `${item.title || 'Project'} · Raul Valle`;
    setMetaDescription(item.desc || item.subtitle || 'Project details');
    document.getElementById('project-kicker').textContent = source ? source.kicker : 'Project';
    document.getElementById('project-title').textContent = item.title || 'Untitled project';
    document.getElementById('project-subtitle').textContent = item.subtitle || item.desc || '';

    heroImg.src = item.img || PLACEHOLDER_IMG;
    heroImg.alt = item.imgAlt || item.title || 'Project image';
    heroImg.addEventListener('error', () => {
      if (heroImg.src && !heroImg.src.includes(PLACEHOLDER_IMG)) heroImg.src = PLACEHOLDER_IMG;
    });

    crumbs.innerHTML = '';
    crumbs.appendChild(el('a', { href: 'research.html' }, ['Research']));
    if (source) {
      crumbs.appendChild(document.createTextNode(' / '));
      crumbs.appendChild(el('a', { href: source.backHref }, [source.crumbLabel]));
    }
    crumbs.appendChild(document.createTextNode(' / '));
    crumbs.appendChild(el('span', {}, [item.title || 'Project']));

    cta.innerHTML = '';
    if (source) {
      cta.appendChild(el('a', { class: 'btn', href: source.backHref }, [source.backLabel]));
    }
    (item.links || []).forEach((link, index) => {
      if (!link || !link.href) return;
      cta.appendChild(el('a', {
        class: `btn${index === 0 ? ' primary' : ''}`,
        href: link.href,
        target: '_blank',
        rel: 'noreferrer',
      }, [link.label || 'Open link']));
    });

    const main = el('div', { class: 'project-main' }, []);
    const side = el('div', { class: 'project-side' }, []);

    main.appendChild(el('article', { class: 'card reveal' }, [
      el('h2', { class: 'project-section-title' }, ['Overview']),
      item.desc ? el('p', {}, [item.desc]) : null,
      details.overview ? el('p', {}, [details.overview]) : null,
    ].filter(Boolean)));

    [
      listCard('What I built', details.what_i_built),
      listCard('How it works', details.how_it_works),
      listCard('Deliverables', details.deliverables),
    ].forEach((card) => {
      if (card) main.appendChild(card);
    });

    if (Array.isArray(details.glossary) && details.glossary.length) {
      side.appendChild(el('article', { class: 'card reveal' }, [
        el('h2', { class: 'project-section-title' }, ['Glossary']),
        el('dl', { class: 'glossary' }, details.glossary.flatMap((entry) => {
          if (!entry) return [];
          return [
            el('dt', {}, [entry.term || '']),
            el('dd', {}, [entry.definition || '']),
          ];
        })),
      ]));
    }

    const snapshotRows = [];
    if (item.meta) snapshotRows.push(kvRow('Track', item.meta));
    if (details.status) snapshotRows.push(kvRow('Status', details.status));
    if (item.pills && item.pills.length) {
      snapshotRows.push(el('div', { class: 'kv-row' }, [
        el('div', { class: 'kv-key' }, ['Focus']),
        el('div', { class: 'kv-val pill-row' }, item.pills.map((pill) => el('span', { class: 'pill' }, [pill]))),
      ]));
    }
    if (snapshotRows.length) {
      side.appendChild(el('article', { class: 'card reveal' }, [
        el('h2', { class: 'project-section-title' }, ['Snapshot']),
        el('div', { class: 'kv' }, snapshotRows),
      ]));
    }

    if (Array.isArray(details.stack) && details.stack.length) {
      side.appendChild(el('article', { class: 'card reveal' }, [
        el('h2', { class: 'project-section-title' }, ['Stack']),
        el('ul', {}, details.stack.map((entry) => el('li', {}, [String(entry)]))),
      ]));
    }

    if (Array.isArray(details.collaborators) && details.collaborators.length) {
      side.appendChild(el('article', { class: 'card reveal' }, [
        el('h2', { class: 'project-section-title' }, ['Collaborators']),
        el('ul', {}, details.collaborators.map((entry) => el('li', {}, [String(entry)]))),
      ]));
    }

    if (Array.isArray(item.links) && item.links.length) {
      side.appendChild(el('article', { class: 'card reveal' }, [
        el('h2', { class: 'project-section-title' }, ['Links']),
        el('div', { class: 'pill-row' }, item.links.map((link) =>
          el('a', { class: 'pill', href: link.href, target: '_blank', rel: 'noreferrer' }, [link.label || 'Open'])
        )),
      ]));
    }

    detailMount.innerHTML = '';
    detailMount.appendChild(main);
    detailMount.appendChild(side);

    if (window.SiteUI && typeof window.SiteUI.refreshReveal === 'function') {
      window.SiteUI.refreshReveal(document.getElementById('project-hero'));
      window.SiteUI.refreshReveal(detailMount);
    }
  }

  async function loadSource(key) {
    const source = SOURCES[key];
    if (!source) return null;
    const res = await fetch(source.url, { cache: 'no-cache' });
    if (!res.ok) throw new Error(`Failed to load ${source.url}`);
    return res.json();
  }

  async function findProject() {
    const params = new URLSearchParams(window.location.search);
    const slug = params.get('slug');
    const requestedSource = params.get('src');

    if (!slug) {
      showNotice('No project was specified. Use the research pages to browse available work.');
      return;
    }

    const sourceKeys = requestedSource && SOURCES[requestedSource]
      ? [requestedSource]
      : Object.keys(SOURCES);

    for (const sourceKey of sourceKeys) {
      const data = await loadSource(sourceKey);
      const match = (data.items || []).find((item) => item.slug === slug);
      if (match) {
        renderProject(match, sourceKey);
        return;
      }
    }

    showNotice('That project could not be found. Use the research pages to browse available work.');
  }

  findProject().catch((error) => {
    console.warn(error);
    showNotice('The project page could not be loaded right now.');
  });
})();
