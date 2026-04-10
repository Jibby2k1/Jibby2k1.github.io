import fs from 'node:fs/promises';
import path from 'node:path';

const root = process.cwd();
const site = JSON.parse(await fs.readFile(path.join(root, 'data/site.json'), 'utf8'));
const nowIso = new Date().toISOString().slice(0, 10);

const staticPages = [
  { file: 'about.html', type: 'ProfilePage' },
  { file: 'cv.html', type: 'WebPage' },
  { file: 'contact.html', type: 'ContactPage' },
  { file: 'publications.html', type: 'CollectionPage' }
];

const tagLabels = {
  'paper-note': 'Paper note',
  'research-journal': 'Research journal',
  'state-of-research': 'State of research'
};

const trackMeta = {
  cnel: {
    slug: 'research-cnel.html',
    label: 'CNEL',
    name: 'CNEL projects',
    kicker: 'Computational NeuroEngineering Lab',
    description: 'EEG, voltage imaging, biological time-series modeling, and experiment infrastructure connected to CNEL research at UF.'
  },
  sps: {
    slug: 'research-sps.html',
    label: 'IEEE SPS @ UF',
    name: 'IEEE SPS projects',
    kicker: 'IEEE Signal Processing Society at UF',
    description: 'Student-facing research engineering, workshops, neurotechnology prototypes, robotics, and open-source systems through IEEE SPS at UF.'
  }
};

function escapeHtml(value = '') {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

function formatDate(value) {
  return new Date(`${value}T00:00:00`).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric'
  });
}

function canonicalFor(outputPath) {
  if (outputPath === 'index.html') return `${site.siteUrl}/`;
  return `${site.siteUrl}/${outputPath}`;
}

function prefixFor(outputPath) {
  const depth = outputPath.split('/').length - 1;
  return depth > 0 ? '../'.repeat(depth) : '';
}

function nav(outputPath) {
  const prefix = prefixFor(outputPath);
  return `<header class="site-header">
  <a class="skip-link" href="#main">Skip to content</a>
  <div class="container navbar" role="navigation" aria-label="Primary">
    <a class="brand" href="${prefix}index.html">
      <span class="brand-copy">
        <span class="brand-name">${site.name}</span>
        <span class="brand-eyebrow">${escapeHtml(site.tagline)}</span>
      </span>
    </a>

    <button class="burger" id="burger" aria-label="Open menu" aria-expanded="false" aria-controls="navlinks">
      <span aria-hidden="true">☰</span>
    </button>

    <nav class="navlinks" id="navlinks">
      <a class="navlink" href="${prefix}index.html" data-route="index.html">Home</a>
      <a class="navlink" href="${prefix}about.html" data-route="about.html">About</a>
      <a class="navlink" href="${prefix}cv.html" data-route="cv.html">CV</a>
      <a class="navlink" href="${prefix}research.html" data-route="research.html">Research</a>
      <a class="navlink" href="${prefix}publications.html" data-route="publications.html">Publications</a>
      <a class="navlink" href="${prefix}blog.html" data-route="blog.html">Writing</a>
      <a class="navlink" href="${prefix}awards.html" data-route="awards.html">Awards</a>
      <a class="navlink" href="${prefix}photography.html" data-route="photography.html">Photography</a>
      <a class="navlink" href="${prefix}contact.html" data-route="contact.html">Contact</a>
    </nav>

    <div class="nav-right">
      <a class="icon-link" href="https://github.com/Jibby2k1" target="_blank" rel="me noreferrer">GitHub</a>
      <a class="icon-link" href="https://www.linkedin.com/in/raul-valle1/" target="_blank" rel="me noreferrer">LinkedIn</a>
      <a class="icon-link" href="https://www.ece.ufl.edu/" target="_blank" rel="noreferrer">UF ECE</a>
    </div>
  </div>
</header>`;
}

function footer(outputPath) {
  const prefix = prefixFor(outputPath);
  return `<footer class="footer">
  <div class="container footer-row">
    <div class="footer-copy">
      <div class="footer-label">About This Site</div>
      <p>${escapeHtml(site.shortBio)}</p>
      <div class="footer-meta">© <span id="year"></span> ${site.name} · Built for GitHub Pages</div>
    </div>

    <div class="footer-group">
      <div class="footer-label">Browse</div>
      <div class="footer-links">
        <a href="${prefix}research.html">Research</a>
        <a href="${prefix}blog.html">Writing</a>
        <a href="${prefix}publications.html">Publications</a>
        <a href="${prefix}photography.html">Photography</a>
        <a href="${prefix}contact.html">Contact</a>
      </div>
    </div>

    <div class="footer-group">
      <div class="footer-label">Profiles</div>
      <div class="footer-links">
        ${site.profiles.map((profile) => `<a href="${profile.href}" target="_blank" rel="${profile.rel}">${profile.label}</a>`).join('')}
      </div>
    </div>
  </div>
</footer>`;
}

function personSchema() {
  return {
    '@type': 'Person',
    '@id': `${site.siteUrl}/#person`,
    name: site.name,
    url: `${site.siteUrl}/`,
    image: `${site.siteUrl}/assets/img/me/Raul_me.jpeg`,
    jobTitle: site.jobTitle,
    description: site.longBio,
    homeLocation: {
      '@type': 'Place',
      name: site.location
    },
    affiliation: site.affiliations.map((entry) => ({
      '@type': entry.type,
      name: entry.name,
      ...(entry.url ? { url: entry.url } : {})
    })),
    sameAs: site.sameAs
  };
}

function websiteSchema() {
  return {
    '@type': 'WebSite',
    '@id': `${site.siteUrl}/#website`,
    url: `${site.siteUrl}/`,
    name: site.name,
    description: site.longBio,
    publisher: { '@id': `${site.siteUrl}/#person` },
    inLanguage: 'en-US'
  };
}

function pageShell({
  outputPath,
  title,
  description,
  type = 'website',
  pageType = 'WebPage',
  image = `${site.siteUrl}/assets/img/me/Raul_me.jpeg`,
  imageAlt = `Portrait of ${site.name}`,
  main,
  schemaExtras = []
}) {
  const prefix = prefixFor(outputPath);
  const canonical = canonicalFor(outputPath);
  const graph = [
    personSchema(),
    websiteSchema(),
    {
      '@type': pageType,
      '@id': `${canonical}#webpage`,
      url: canonical,
      name: title,
      description,
      inLanguage: 'en-US',
      about: { '@id': `${site.siteUrl}/#person` },
      isPartOf: { '@id': `${site.siteUrl}/#website` }
    },
    ...schemaExtras
  ];

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>${escapeHtml(title)}</title>
  <meta name="description" content="${escapeHtml(description)}" />
  <meta name="author" content="${site.name}" />
  <meta name="robots" content="index, follow, max-image-preview:large" />
  <meta property="og:locale" content="en_US" />
  <meta property="og:type" content="${type}" />
  <meta property="og:title" content="${escapeHtml(title)}" />
  <meta property="og:description" content="${escapeHtml(description)}" />
  <meta property="og:url" content="${canonical}" />
  <meta property="og:site_name" content="${site.name}" />
  <meta property="og:image" content="${image}" />
  <meta property="og:image:alt" content="${escapeHtml(imageAlt)}" />
  <meta name="twitter:card" content="summary_large_image" />
  <meta name="twitter:title" content="${escapeHtml(title)}" />
  <meta name="twitter:description" content="${escapeHtml(description)}" />
  <meta name="twitter:image" content="${image}" />
  <meta name="twitter:image:alt" content="${escapeHtml(imageAlt)}" />
  <link rel="canonical" href="${canonical}" />
  <link rel="alternate" type="application/rss+xml" title="${escapeHtml(site.name)} Writing RSS" href="${prefix}rss.xml" />
  <link rel="stylesheet" href="${prefix}assets/css/styles.css" />
  <script type="application/ld+json">${JSON.stringify({ '@context': 'https://schema.org', '@graph': graph })}</script>
</head>
<body>
  ${nav(outputPath)}
  <main id="main">${main}</main>
  ${footer(outputPath)}
  <script src="${prefix}assets/js/site.js"></script>
</body>
</html>`;
}

async function loadJson(file) {
  return JSON.parse(await fs.readFile(path.join(root, file), 'utf8'));
}

function uniqueTags(items) {
  return [...new Set(items.flatMap((item) => item.tags || []))].sort((a, b) => a.localeCompare(b));
}

function searchText(parts) {
  return parts.filter(Boolean).join(' ').toLowerCase();
}

function extractMain(html) {
  const match = html.match(/<main id="main">([\s\S]*?)<\/main>/i);
  return match ? match[1].trim() : '';
}

function extractTitle(html) {
  return html.match(/<title>([\s\S]*?)<\/title>/i)?.[1]?.trim() || '';
}

function extractDescription(html) {
  return html.match(/<meta name="description" content="([\s\S]*?)"/i)?.[1]?.trim() || '';
}

function renderMarkdown(markdown) {
  const lines = markdown.trim().split('\n');
  const blocks = [];
  let paragraph = [];
  let list = [];

  const flushParagraph = () => {
    if (!paragraph.length) return;
    blocks.push(`<p>${paragraph.join(' ')}</p>`);
    paragraph = [];
  };

  const flushList = () => {
    if (!list.length) return;
    blocks.push(`<ul>${list.map((item) => `<li>${item}</li>`).join('')}</ul>`);
    list = [];
  };

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) {
      flushParagraph();
      flushList();
      continue;
    }
    if (line.startsWith('- ')) {
      flushParagraph();
      list.push(inlineMarkdown(line.slice(2)));
      continue;
    }
    flushList();
    if (line.startsWith('## ')) {
      flushParagraph();
      blocks.push(`<h2>${inlineMarkdown(line.slice(3))}</h2>`);
      continue;
    }
    if (line.startsWith('### ')) {
      flushParagraph();
      blocks.push(`<h3>${inlineMarkdown(line.slice(4))}</h3>`);
      continue;
    }
    paragraph.push(inlineMarkdown(line));
  }

  flushParagraph();
  flushList();
  return blocks.join('\n');
}

function inlineMarkdown(value) {
  return escapeHtml(value)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');
}

function parseFrontmatter(text) {
  const match = text.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
  if (!match) throw new Error('Missing frontmatter');
  const metaLines = match[1].split('\n');
  const meta = {};
  let currentKey = null;
  for (const line of metaLines) {
    if (line.startsWith('  - ')) {
      meta[currentKey] ||= [];
      meta[currentKey].push(line.slice(4).trim());
      continue;
    }
    const keyMatch = line.match(/^([A-Za-z0-9_]+):\s*(.*)$/);
    if (!keyMatch) continue;
    currentKey = keyMatch[1];
    const value = keyMatch[2].trim();
    if (!value) {
      meta[currentKey] = [];
      continue;
    }
    meta[currentKey] = value;
  }
  return { meta, body: match[2].trim() };
}

async function loadWriting() {
  const dir = path.join(root, 'content/writing');
  const entries = await fs.readdir(dir);
  const posts = [];
  for (const entry of entries) {
    if (!entry.endsWith('.md')) continue;
    const text = await fs.readFile(path.join(dir, entry), 'utf8');
    const { meta, body } = parseFrontmatter(text);
    posts.push({
      ...meta,
      topics: Array.isArray(meta.topics) ? meta.topics : [],
      relatedProjects: Array.isArray(meta.relatedProjects) ? meta.relatedProjects : [],
      takeaways: Array.isArray(meta.takeaways) ? meta.takeaways : [],
      body,
      bodyHtml: renderMarkdown(body),
      url: `notes/${meta.slug}.html`
    });
  }
  return posts.sort((a, b) => b.date.localeCompare(a.date));
}

function cardProject(project, prefix = '', extraClass = '') {
  const href = `${prefix}projects/${project.slug}.html`;
  return `<article class="card record-card reveal filter-card ${extraClass}" data-filter-item data-tags="${escapeHtml((project.tags || []).join(','))}" data-search="${escapeHtml(searchText([project.title, project.subtitle, project.desc, ...(project.tags || []), ...(project.pills || [])]))}">
  <img class="card-img" src="${prefix}${project.img}" alt="${escapeHtml(project.imgAlt || project.title)}" loading="lazy" decoding="async" />
  <div class="meta">${escapeHtml(project.meta || '')}</div>
  <h3>${escapeHtml(project.title)}</h3>
  <p>${escapeHtml(project.desc || project.subtitle || '')}</p>
  <div class="pill-row">${(project.pills || []).map((pill) => `<span class="pill">${escapeHtml(pill)}</span>`).join('')}</div>
  <div class="cta-row">
    <a class="btn primary" href="${href}">Open project</a>
  </div>
</article>`;
}

function cardWriting(post, prefix = '') {
  const href = `${prefix}${post.url}`;
  return `<article class="card blog-card reveal filter-card" data-filter-item data-tags="${escapeHtml([post.kind, ...(post.topics || [])].join(','))}" data-search="${escapeHtml(searchText([post.title, post.summary, post.description, post.kind, ...(post.topics || [])]))}">
  <img class="blog-media" src="${prefix}${post.heroImage}" alt="${escapeHtml(post.heroAlt || post.title)}" loading="lazy" decoding="async" />
  <div class="blog-copy">
    <div class="blog-meta">${formatDate(post.date)} · ${escapeHtml(post.label || tagLabels[post.kind] || post.kind)}</div>
    <h3 class="blog-title">${escapeHtml(post.title)}</h3>
    <p class="blog-summary">${escapeHtml(post.summary || post.description)}</p>
  </div>
  <div class="pill-row">${(post.topics || []).map((topic) => `<span class="pill">${escapeHtml(topic)}</span>`).join('')}</div>
  <div class="cta-row">
    <a class="btn primary" href="${href}">Read entry</a>
  </div>
</article>`;
}

function filterBar({ searchLabel, buttons, emptyMessage }) {
  return `<div class="filter-shell" data-filter-root>
  <div class="filter-bar">
    <label class="filter-search">
      <span class="filter-title">${escapeHtml(searchLabel)}</span>
      <input class="filter-input" type="search" placeholder="Search..." data-filter-search />
    </label>
    <div class="filters">
      ${buttons.map((button, index) => `<button class="filter-btn${index === 0 ? ' active' : ''}" type="button" data-filter-button data-filter-value="${escapeHtml(button.value)}">${escapeHtml(button.label)}</button>`).join('')}
    </div>
    <div class="filter-count" data-filter-count></div>
  </div>
  <div class="filter-empty notice" data-filter-empty hidden>${escapeHtml(emptyMessage)}</div>
</div>`;
}

async function buildStaticPage(file, type) {
  const current = await fs.readFile(path.join(root, file), 'utf8');
  const body = extractMain(current);
  const title = extractTitle(current);
  const description = extractDescription(current);
  await fs.writeFile(path.join(root, file), pageShell({
    outputPath: file,
    title,
    description,
    pageType: type,
    main: body
  }));
}

async function writePage(outputPath, content) {
  await fs.mkdir(path.dirname(path.join(root, outputPath)), { recursive: true });
  await fs.writeFile(path.join(root, outputPath), content);
}

async function main() {
  const [cnelData, spsData, awardsData, photographyData, writing] = await Promise.all([
    loadJson('data/cnel_projects.json'),
    loadJson('data/sps_projects.json'),
    loadJson('data/awards.json'),
    loadJson('data/photography.json'),
    loadWriting()
  ]);

  const projects = [
    ...cnelData.items.map((item) => ({ ...item, track: 'cnel' })),
    ...spsData.items.map((item) => ({ ...item, track: 'sps' }))
  ];
  const featuredProjects = projects.filter((item) => item.featured).slice(0, 4);

  for (const page of staticPages) {
    await buildStaticPage(page.file, page.type);
  }

  await writePage('index.html', pageShell({
    outputPath: 'index.html',
    title: 'Raul Valle | Signal Processing, Neuroengineering, and Research Writing',
    description: 'Research site and writing archive for Raul Valle, a University of Florida Ph.D. student working on signal processing, neuroengineering, research software, and technical commentary.',
    image: `${site.siteUrl}/assets/img/me/Raul_me.jpeg`,
    main: `<div class="container">
      <section class="page-hero archive-hero reveal accent-cool">
        <div class="archive-hero-main">
          <div class="kicker">Raul Valle · University of Florida · Gainesville, Florida</div>
          <h1 class="h1">Signal processing, neuroengineering, and research systems that hold up outside the benchmark.</h1>
          <p class="lead">${escapeHtml(site.shortBio)}</p>
          <div class="cta-row">
            <a class="btn primary" href="research.html">Explore research</a>
            <a class="btn" href="blog.html">Read writing</a>
            <a class="btn" href="publications.html">Publications and talks</a>
          </div>
        </div>
        <aside class="card archive-ledger reveal">
          <div class="archive-ledger-label">Current focus</div>
          <dl class="archive-stats">
            <div><dt>Research</dt><dd>EEG, voltage imaging, time-series models, and reproducible experimentation.</dd></div>
            <div><dt>Engineering</dt><dd>Research tooling, biosignal systems, workshop infrastructure, and deployable prototypes.</dd></div>
            <div><dt>Writing</dt><dd>Paper commentary, research journal notes, and state-of-research reflections.</dd></div>
          </dl>
        </aside>
      </section>

      <section class="accent-mint">
        <h2 class="section-title reveal">Featured projects</h2>
        <p class="section-subtitle reveal">The highest-signal project pages that explain what I am building and why it matters.</p>
        <div class="grid two" style="margin-top:14px; gap:14px;">
          ${featuredProjects.map((project) => cardProject(project)).join('')}
        </div>
      </section>

      <section class="accent-amber">
        <h2 class="section-title reveal">Recent writing</h2>
        <p class="section-subtitle reveal">A public research journal for technical reflections, paper notes, and commentary on what is worth paying attention to.</p>
        <div class="grid three" style="margin-top:14px; gap:14px;">
          ${writing.slice(0, 3).map((post) => cardWriting(post)).join('')}
        </div>
        <div class="cta-row">
          <a class="btn primary" href="blog.html">Open writing archive</a>
        </div>
      </section>

      <section class="accent-cool">
        <h2 class="section-title reveal">Proof and context</h2>
        <div class="grid three" style="margin-top:14px;">
          <article class="card reveal">
            <div class="archive-ledger-label">Publications</div>
            <h3>Talks, preprints, and official coverage</h3>
            <p>One page for papers, talk videos, UF profiles, and public references tied to the work on this site.</p>
            <div class="cta-row"><a class="btn primary" href="publications.html">Open publications</a></div>
          </article>
          <article class="card reveal">
            <div class="archive-ledger-label">Research tracks</div>
            <h3>CNEL and IEEE SPS @ UF</h3>
            <p>The project archive is organized by lab-focused work and student-facing engineering work so the site stays legible.</p>
            <div class="cta-row"><a class="btn primary" href="research.html">Open research overview</a></div>
          </article>
          <article class="card reveal">
            <div class="archive-ledger-label">Secondary portfolio</div>
            <h3>Selected photography</h3>
            <p>Portrait and character photography remains part of the site, but clearly secondary to the research and writing surfaces.</p>
            <div class="cta-row"><a class="btn primary" href="photography.html">Open photography</a></div>
          </article>
        </div>
      </section>
    </div>`,
    schemaExtras: [{
      '@type': 'ItemList',
      '@id': `${site.siteUrl}/#featured`,
      itemListElement: featuredProjects.map((item, index) => ({
        '@type': 'ListItem',
        position: index + 1,
        item: {
          '@type': 'CreativeWork',
          name: item.title,
          url: `${site.siteUrl}/projects/${item.slug}.html`
        }
      }))
    }]
  }));

  await writePage('research.html', pageShell({
    outputPath: 'research.html',
    title: 'Raul Valle Research | Signal Processing, Neuroengineering, and Research Systems',
    description: 'Research overview for Raul Valle at the University of Florida, covering CNEL work, IEEE SPS projects, publications, and technical writing.',
    pageType: 'CollectionPage',
    image: `${site.siteUrl}/assets/img/projects/HLDS_Inf.png`,
    imageAlt: 'Research visual for Raul Valle',
    main: `<div class="container">
      <section class="page-hero reveal accent-cool">
        <div class="kicker">Research overview</div>
        <h1 class="h1">Research and engineering tracks</h1>
        <p class="lead">My work is organized around two tracks: lab-focused research in CNEL and engineering, workshop, and prototype work through IEEE SPS at UF. The common thread is signal processing that remains useful when the data is noisy, the hardware is imperfect, and the deployment constraints are real.</p>
      </section>

      <div class="grid two accent-mint" style="margin-top:18px; gap:14px;">
        ${Object.entries(trackMeta).map(([key, track]) => `<article class="card reveal">
          <div class="archive-ledger-label">${track.label}</div>
          <h3>${track.name}</h3>
          <p>${track.description}</p>
          <div class="cta-row">
            <a class="btn primary" href="${track.slug}">Open track</a>
          </div>
        </article>`).join('')}
      </div>

      <section class="accent-amber">
        <h2 class="section-title reveal">Featured project pages</h2>
        <p class="section-subtitle reveal">Canonical project pages with collaborators, scope, and related work.</p>
        <div class="grid two" style="margin-top:14px; gap:14px;">
          ${featuredProjects.map((project) => cardProject(project)).join('')}
        </div>
      </section>

      <section class="accent-cool">
        <h2 class="section-title reveal">Writing connected to the work</h2>
        <p class="section-subtitle reveal">Short public notes that explain what I care about, what I am learning, and where the research is moving.</p>
        <div class="grid three" style="margin-top:14px; gap:14px;">
          ${writing.slice(0, 3).map((post) => cardWriting(post)).join('')}
        </div>
        <div class="cta-row">
          <a class="btn primary" href="blog.html">Open writing archive</a>
        </div>
      </section>
    </div>`
  }));

  for (const [key, track] of Object.entries(trackMeta)) {
    const items = projects.filter((item) => item.track === key);
    const buttons = [
      { value: 'all', label: 'All projects' },
      ...uniqueTags(items).map((tag) => ({ value: tag, label: tag }))
    ];
    await writePage(track.slug, pageShell({
      outputPath: track.slug,
      title: `${track.name} | Raul Valle`,
      description: track.description,
      pageType: 'CollectionPage',
      image: `${site.siteUrl}/${items[0]?.img || 'assets/img/me/Raul_me.jpeg'}`,
      imageAlt: `${track.name} visual`,
      main: `<div class="container">
        <section class="page-hero reveal accent-cool">
          <div class="kicker">${track.kicker}</div>
          <h1 class="h1">${track.name}</h1>
          <p class="lead">${track.description}</p>
        </section>
        ${filterBar({
          searchLabel: `Search ${track.label} work`,
          buttons,
          emptyMessage: 'No projects match that filter yet.'
        })}
        <section class="accent-mint">
          <div class="grid two filter-grid" style="margin-top:14px; gap:14px;" data-filter-grid>
            ${items.map((project) => cardProject(project)).join('')}
          </div>
        </section>
      </div>`,
      schemaExtras: [{
        '@type': 'ItemList',
        '@id': `${site.siteUrl}/${track.slug}#itemlist`,
        itemListElement: items.map((item, index) => ({
          '@type': 'ListItem',
          position: index + 1,
          item: {
            '@type': 'CreativeWork',
            name: item.title,
            url: `${site.siteUrl}/projects/${item.slug}.html`
          }
        }))
      }]
    }));
  }

  await writePage('blog.html', pageShell({
    outputPath: 'blog.html',
    title: 'Raul Valle Writing | Research Journal, Paper Notes, and Commentary',
    description: 'Public writing by Raul Valle: paper notes, research journal entries, and state-of-research commentary on signal processing, machine learning, and neuroengineering.',
    pageType: 'CollectionPage',
    image: `${site.siteUrl}/${writing[0]?.heroImage || 'assets/img/me/Raul_me.jpeg'}`,
    imageAlt: 'Raul Valle writing archive',
    main: `<div class="container">
      <section class="page-hero reveal accent-cool">
        <div class="kicker">Writing and commentary</div>
        <h1 class="h1">A public research journal</h1>
        <p class="lead">This section is where I write more often: paper reactions, research notes, hardware lessons, and commentary on what seems important or overhyped in the current research landscape.</p>
      </section>
      ${filterBar({
        searchLabel: 'Search writing',
        buttons: [
          { value: 'all', label: 'All entries' },
          { value: 'research-journal', label: 'Research journal' },
          { value: 'paper-note', label: 'Paper notes' },
          { value: 'state-of-research', label: 'State of research' }
        ],
        emptyMessage: 'No writing entries match that filter yet.'
      })}
      <section class="accent-mint">
        <div class="grid three filter-grid" style="margin-top:14px; gap:14px;" data-filter-grid>
          ${writing.map((post) => cardWriting(post)).join('')}
        </div>
      </section>
    </div>`,
    schemaExtras: [{
      '@type': 'ItemList',
      '@id': `${site.siteUrl}/blog.html#itemlist`,
      itemListElement: writing.map((post, index) => ({
        '@type': 'ListItem',
        position: index + 1,
        item: {
          '@type': 'BlogPosting',
          headline: post.title,
          url: `${site.siteUrl}/${post.url}`
        }
      }))
    }]
  }));

  const projectMap = new Map(projects.map((project) => [project.slug, project]));
  for (const project of projects) {
    const relatedWriting = writing.filter((post) => (post.relatedProjects || []).includes(project.slug));
    const track = trackMeta[project.track];
    const details = project.details || {};
    await writePage(`projects/${project.slug}.html`, pageShell({
      outputPath: `projects/${project.slug}.html`,
      title: `${project.title} | Raul Valle Research Project`,
      description: project.desc || project.subtitle,
      image: `${site.siteUrl}/${project.img}`,
      imageAlt: project.imgAlt || project.title,
      main: `<div class="container">
        <section class="page-hero reveal hero-split accent-cool">
          <div>
            <div class="kicker">${track.label}</div>
            <h1 class="h1">${escapeHtml(project.title)}</h1>
            <p class="lead">${escapeHtml(project.subtitle || project.desc)}</p>
            <div class="breadcrumbs"><a href="../research.html">Research</a> / <a href="../${track.slug}">${track.name}</a> / <span>${escapeHtml(project.title)}</span></div>
            <div class="cta-row">
              <a class="btn" href="../${track.slug}">Back to ${track.name}</a>
              ${(project.links || []).map((link, index) => `<a class="btn${index === 0 ? ' primary' : ''}" href="${link.href}" target="_blank" rel="noreferrer">${escapeHtml(link.label)}</a>`).join('')}
            </div>
          </div>
          <img class="card-img compact project-hero-img" src="../${project.img}" alt="${escapeHtml(project.imgAlt || project.title)}" loading="lazy" decoding="async" />
        </section>

        <section class="project-detail reveal accent-mint">
          <div class="project-main">
            <article class="card reveal">
              <h2 class="project-section-title">Overview</h2>
              <p>${escapeHtml(project.desc || '')}</p>
              ${details.overview ? `<p>${escapeHtml(details.overview)}</p>` : ''}
            </article>
            ${Array.isArray(details.what_i_built) && details.what_i_built.length ? `<article class="card reveal"><h2 class="project-section-title">What I built</h2><ul>${details.what_i_built.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</ul></article>` : ''}
            ${Array.isArray(details.how_it_works) && details.how_it_works.length ? `<article class="card reveal"><h2 class="project-section-title">How it works</h2><ul>${details.how_it_works.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</ul></article>` : ''}
            ${Array.isArray(details.deliverables) && details.deliverables.length ? `<article class="card reveal"><h2 class="project-section-title">Deliverables</h2><ul>${details.deliverables.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</ul></article>` : ''}
            ${relatedWriting.length ? `<article class="card reveal"><h2 class="project-section-title">Related writing</h2><div class="grid two">${relatedWriting.map((post) => `<a class="related-link" href="../${post.url}"><strong>${escapeHtml(post.title)}</strong><span>${escapeHtml(post.summary || post.description)}</span></a>`).join('')}</div></article>` : ''}
          </div>
          <div class="project-side">
            <article class="card reveal">
              <h2 class="project-section-title">Snapshot</h2>
              <div class="kv">
                ${project.meta ? `<div class="kv-row"><div class="kv-key">Track</div><div class="kv-val">${escapeHtml(project.meta)}</div></div>` : ''}
                ${details.status ? `<div class="kv-row"><div class="kv-key">Status</div><div class="kv-val">${escapeHtml(details.status)}</div></div>` : ''}
                ${(project.pills || []).length ? `<div class="kv-row"><div class="kv-key">Focus</div><div class="kv-val pill-row">${project.pills.map((pill) => `<span class="pill">${escapeHtml(pill)}</span>`).join('')}</div></div>` : ''}
              </div>
            </article>
            ${Array.isArray(details.stack) && details.stack.length ? `<article class="card reveal"><h2 class="project-section-title">Stack</h2><ul>${details.stack.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</ul></article>` : ''}
            ${Array.isArray(details.collaborators) && details.collaborators.length ? `<article class="card reveal"><h2 class="project-section-title">Collaborators</h2><ul>${details.collaborators.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</ul></article>` : ''}
            ${Array.isArray(details.glossary) && details.glossary.length ? `<article class="card reveal"><h2 class="project-section-title">Glossary</h2><dl class="glossary">${details.glossary.map((entry) => `<dt>${escapeHtml(entry.term)}</dt><dd>${escapeHtml(entry.definition)}</dd>`).join('')}</dl></article>` : ''}
          </div>
        </section>
      </div>`,
      schemaExtras: [
        {
          '@type': 'CreativeWork',
          '@id': `${site.siteUrl}/projects/${project.slug}.html#project`,
          url: `${site.siteUrl}/projects/${project.slug}.html`,
          name: project.title,
          headline: project.title,
          description: project.desc || project.subtitle,
          image: `${site.siteUrl}/${project.img}`,
          creator: { '@id': `${site.siteUrl}/#person` },
          keywords: [...(project.tags || []), ...(project.pills || [])]
        },
        {
          '@type': 'BreadcrumbList',
          itemListElement: [
            { '@type': 'ListItem', position: 1, name: 'Home', item: `${site.siteUrl}/` },
            { '@type': 'ListItem', position: 2, name: 'Research', item: `${site.siteUrl}/research.html` },
            { '@type': 'ListItem', position: 3, name: track.name, item: `${site.siteUrl}/${track.slug}` },
            { '@type': 'ListItem', position: 4, name: project.title, item: `${site.siteUrl}/projects/${project.slug}.html` }
          ]
        }
      ]
    }));
  }

  for (const post of writing) {
    const relatedProjects = post.relatedProjects.map((slug) => projectMap.get(slug)).filter(Boolean);
    await writePage(post.url, pageShell({
      outputPath: post.url,
      title: `${post.title} | Raul Valle Writing`,
      description: post.description,
      type: 'article',
      pageType: 'WebPage',
      image: `${site.siteUrl}/${post.heroImage}`,
      imageAlt: post.heroAlt || post.title,
      main: `<div class="container">
        <section class="page-hero hero-split reveal accent-cool">
          <div>
            <div class="kicker">${escapeHtml(post.label || tagLabels[post.kind] || post.kind)}</div>
            <h1 class="h1">${escapeHtml(post.title)}</h1>
            <p class="lead">${escapeHtml(post.description)}</p>
            <div class="breadcrumbs"><a href="../blog.html">Writing</a> / <span>${escapeHtml(post.title)}</span></div>
            <div class="meta">${formatDate(post.date)} · ${(post.topics || []).map((topic) => escapeHtml(topic)).join(' · ')}</div>
          </div>
          <img class="card-img compact project-hero-img" src="../${post.heroImage}" alt="${escapeHtml(post.heroAlt || post.title)}" loading="lazy" decoding="async" />
        </section>

        <section class="project-detail reveal accent-mint">
          <div class="project-main">
            <article class="card reveal prose-card">
              <h2 class="project-section-title">Entry</h2>
              ${post.bodyHtml}
            </article>
          </div>
          <div class="project-side">
            ${post.takeaways.length ? `<article class="card reveal"><h2 class="project-section-title">Key takeaways</h2><ul>${post.takeaways.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</ul></article>` : ''}
            ${relatedProjects.length ? `<article class="card reveal"><h2 class="project-section-title">Related projects</h2><div class="stack-links">${relatedProjects.map((project) => `<a class="related-link" href="../projects/${project.slug}.html"><strong>${escapeHtml(project.title)}</strong><span>${escapeHtml(project.desc || project.subtitle)}</span></a>`).join('')}</div></article>` : ''}
          </div>
        </section>
      </div>`,
      schemaExtras: [
        {
          '@type': 'BlogPosting',
          '@id': `${site.siteUrl}/${post.url}#post`,
          headline: post.title,
          description: post.description,
          datePublished: post.date,
          dateModified: post.date,
          author: { '@id': `${site.siteUrl}/#person` },
          publisher: { '@id': `${site.siteUrl}/#person` },
          image: `${site.siteUrl}/${post.heroImage}`,
          mainEntityOfPage: `${site.siteUrl}/${post.url}`,
          articleSection: post.label || tagLabels[post.kind] || post.kind,
          keywords: post.topics || []
        },
        {
          '@type': 'BreadcrumbList',
          itemListElement: [
            { '@type': 'ListItem', position: 1, name: 'Home', item: `${site.siteUrl}/` },
            { '@type': 'ListItem', position: 2, name: 'Writing', item: `${site.siteUrl}/blog.html` },
            { '@type': 'ListItem', position: 3, name: post.title, item: `${site.siteUrl}/${post.url}` }
          ]
        }
      ]
    }));
  }

  await writePage('awards.html', pageShell({
    outputPath: 'awards.html',
    title: 'Awards and Recognition | Raul Valle',
    description: 'Selected awards and recognitions connected to research, hackathons, and computing at the University of Florida.',
    pageType: 'CollectionPage',
    main: `<div class="container">
      <section class="page-hero reveal accent-cool">
        <div class="kicker">Recognition and milestones</div>
        <h1 class="h1">Awards</h1>
        <p class="lead">Selected awards tied to research, computing, and public-facing project work.</p>
      </section>
      <section class="accent-mint">
        <div class="grid three" style="margin-top:14px; gap:14px;">
          ${awardsData.items.map((award) => `<article class="card reveal">
            <img class="card-img" src="${award.img}" alt="${escapeHtml(award.imgAlt || award.title)}" loading="lazy" decoding="async" />
            <div class="meta">${escapeHtml(award.meta || '')}</div>
            <h3>${escapeHtml(award.title)}</h3>
            <p>${escapeHtml(award.desc)}</p>
          </article>`).join('')}
        </div>
      </section>
    </div>`
  }));

  await writePage('photography.html', pageShell({
    outputPath: 'photography.html',
    title: 'Photography | Raul Valle',
    description: 'Selected photography by Raul Valle, presented as a secondary portfolio surface alongside research and writing.',
    pageType: 'CollectionPage',
    main: `<div class="container">
      <section class="page-hero reveal accent-cool">
        <div class="kicker">Secondary portfolio</div>
        <h1 class="h1">Selected photography</h1>
        <p class="lead">Photography stays on the site as a secondary surface: a visual archive that complements the research work without competing with it.</p>
      </section>
      <section class="accent-mint">
        <div class="photo-grid" style="margin-top:14px;">
          ${photographyData.items.map((item) => `<figure class="photo-card reveal">
            <img class="photo-media" src="${item.img}" alt="${escapeHtml(item.imgAlt || item.title)}" loading="lazy" decoding="async" />
            <figcaption class="photo-copy">
              <div class="photo-category">${escapeHtml(item.collection || 'Selected Work')}</div>
              <h3>${escapeHtml(item.title)}</h3>
              <p class="photo-caption">${escapeHtml(item.caption || '')}</p>
              <div class="photo-meta">${escapeHtml([item.location, item.year].filter(Boolean).join(' · '))}</div>
            </figcaption>
          </figure>`).join('')}
        </div>
      </section>
    </div>`
  }));

  await writePage('project.html', `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Legacy Project Link | Raul Valle</title>
  <meta name="description" content="Legacy project route for Raul Valle. Use the canonical static project pages under /projects/ instead." />
  <meta name="robots" content="noindex, follow, max-image-preview:large" />
  <link rel="canonical" href="${site.siteUrl}/research.html" />
  <link rel="stylesheet" href="assets/css/styles.css" />
</head>
<body>
  ${nav('project.html')}
  <main id="main">
    <div class="container">
      <section class="page-hero reveal accent-cool">
        <div class="kicker">Legacy route</div>
        <h1 class="h1">This project link has moved</h1>
        <p class="lead">Project pages now live under canonical static URLs in <code>/projects/</code>. If the redirect does not happen automatically, use the research index below.</p>
        <div class="cta-row">
          <a class="btn primary" href="research.html">Research overview</a>
          <a class="btn" href="research-cnel.html">CNEL projects</a>
          <a class="btn" href="research-sps.html">IEEE SPS projects</a>
        </div>
      </section>
    </div>
  </main>
  ${footer('project.html')}
  <script>
    (() => {
      const slug = new URLSearchParams(window.location.search).get('slug');
      if (slug) window.location.replace('projects/' + encodeURIComponent(slug) + '.html');
    })();
  </script>
  <script src="assets/js/site.js"></script>
</body>
</html>`);

  const sitemapUrls = [
    'index.html',
    ...staticPages.map((page) => page.file),
    'research.html',
    ...Object.values(trackMeta).map((track) => track.slug),
    'blog.html',
    'awards.html',
    'photography.html',
    'project.html',
    ...projects.map((project) => `projects/${project.slug}.html`),
    ...writing.map((post) => post.url)
  ];
  const sitemap = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">${sitemapUrls.map((page) => `
  <url>
    <loc>${canonicalFor(page)}</loc>
    <lastmod>${nowIso}</lastmod>
  </url>`).join('')}
</urlset>`;
  await writePage('sitemap.xml', sitemap);

  const rss = `<?xml version="1.0" encoding="UTF-8" ?>
<rss version="2.0">
  <channel>
    <title>${escapeHtml(site.name)} Writing</title>
    <link>${site.siteUrl}/blog.html</link>
    <description>${escapeHtml(site.shortBio)}</description>
    ${writing.map((post) => `
    <item>
      <title>${escapeHtml(post.title)}</title>
      <link>${site.siteUrl}/${post.url}</link>
      <guid>${site.siteUrl}/${post.url}</guid>
      <pubDate>${new Date(`${post.date}T00:00:00Z`).toUTCString()}</pubDate>
      <description>${escapeHtml(post.description)}</description>
    </item>`).join('')}
  </channel>
</rss>`;
  await writePage('rss.xml', rss);
}

await main();
