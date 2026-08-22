// Templates for the standalone pages (about, cv, contact).
// Each function receives a context with shared helpers and data:
// { site, escapeHtml, imgTag, awards, projects, writing }

export function aboutPage({ imgTag }) {
  return `<div class="container about-shell">
    <section class="page-hero about-hero reveal accent-cool">
      <div class="about-hero-head">
        <div class="about-hero-title">
          <div class="kicker">About</div>
          <h1 class="h1">About Raul Valle</h1>
        </div>
        ${imgTag('assets/img/me/portrait-680.webp', { className: 'portrait', alt: 'Portrait of Raul Valle', eager: true })}
      </div>
      <div class="about-hero-body">
        <p class="lead">I am a Ph.D. student in Electrical and Computer Engineering at the University of Florida. My current research in the SmartDATA Lab builds machine-learning surrogate models for computational fluid dynamics — fast learned stand-ins for expensive physics solvers that are audited against the physics they approximate. That work sits on a foundation of signal processing, time-series machine learning, and neuroengineering built in the Computational NeuroEngineering Lab.</p>
        <p class="lead">The work that matters most to me sits at the intersection of physical modeling, signals and systems, and machine learning. I care about models that remain interpretable under noise, limited data, and real deployment constraints instead of only looking good in idealized benchmarks.</p>
      </div>
    </section>

    <div class="grid two accent-mint" style="margin-top:18px;">
      <article class="card reveal">
        <h3>Education</h3>
        <p><strong>University of Florida</strong> — Ph.D., Electrical &amp; Computer Engineering (expected 2030)</p>
        <p style="margin-top:10px;"><strong>University of Florida</strong> — B.S., Electrical &amp; Computer Engineering (2020–2025)</p>
        <p style="margin-top:10px;"><strong>Doral Academy Charter School</strong> — STEM diploma (2013–2020)</p>
        <div class="meta">Concentration: analysis, signal processing, and machine learning</div>
      </article>

      <article class="card reveal">
        <h3>Affiliations and profiles</h3>
        <p>My current work is connected to UF ECE, the SmartDATA Lab, the Computational NeuroEngineering Lab, and IEEE Signal Processing Society activities at UF.</p>
        <div class="pill-row">
          <a class="pill" href="https://www.ufl.edu/" target="_blank" rel="noreferrer">University of Florida</a>
          <a class="pill" href="https://www.ece.ufl.edu/" target="_blank" rel="noreferrer">UF ECE</a>
          <a class="pill" href="https://smartdata.ece.ufl.edu/" target="_blank" rel="noreferrer">SmartDATA Lab</a>
          <a class="pill" href="https://github.com/Jibby2k1" target="_blank" rel="me noreferrer">GitHub</a>
          <a class="pill" href="https://www.linkedin.com/in/raul-valle1/" target="_blank" rel="me noreferrer">LinkedIn</a>
          <a class="pill" href="https://x.com/Jibby2k1" target="_blank" rel="me noreferrer">X</a>
        </div>
      </article>
    </div>

    <div class="accent-amber">
      <h2 class="section-title reveal">Research approach</h2>
      <article class="card reveal">
        <h3>How I think about the work</h3>
        <p>I favor problems where modeling assumptions, signal quality, and system design all interact. In practice that means thinking about the data collection stack, the mathematics, and the evaluation pipeline together rather than pretending they are separate tasks.</p>
        <p>That mindset carries into teaching and community work too. If a concept cannot survive being explained clearly, tested, and reused by someone else, the pipeline is probably not rigorous enough yet.</p>
      </article>

      <div class="grid three" style="margin-top:14px; gap:14px;">
        <article class="card reveal">
          <h3>Primary topics</h3>
          <div class="pill-row">
            <span class="pill">CFD surrogate modeling</span>
            <span class="pill">Scientific ML</span>
            <span class="pill">Signal processing</span>
            <span class="pill">Time-series ML</span>
            <span class="pill">Neuroengineering</span>
          </div>
        </article>
        <article class="card reveal">
          <h3>Secondary interests</h3>
          <div class="pill-row">
            <span class="pill">Controls</span>
            <span class="pill">Embedded systems</span>
            <span class="pill">GPU acceleration</span>
            <span class="pill">Technical writing</span>
          </div>
        </article>
        <article class="card reveal">
          <h3>Outside the lab</h3>
          <p>Photography, music, and training matter to me for the same reason research does: they reward discipline, iteration, and attention to detail.</p>
        </article>
      </div>
    </div>

    <section class="accent-cool">
      <h2 class="section-title reveal">Writing, recognition, and visual work</h2>
      <div class="grid three" style="margin-top:14px; gap:14px;">
        <article class="card reveal">
          <div class="archive-ledger-label">Writing</div>
          <h3>Research notes and commentary</h3>
          <p>Short reflections, paper notes, and research journal entries connected to signal processing, machine learning, and neuroengineering.</p>
          <div class="cta-row">
            <a class="btn primary" href="blog.html">Open writing archive</a>
          </div>
        </article>
        <article class="card reveal">
          <div class="archive-ledger-label">Recognition</div>
          <h3>Awards and milestones</h3>
          <p>Selected recognitions tied to research, hackathons, computing, and public-facing project work.</p>
          <div class="cta-row">
            <a class="btn primary" href="awards.html">Open awards archive</a>
          </div>
        </article>
        <article class="card reveal">
          <div class="archive-ledger-label">Photography</div>
          <h3>Selected visual work</h3>
          <p>A small personal portfolio: live music, portraits, and the occasional experiment with light.</p>
          <div class="cta-row">
            <a class="btn primary" href="photography.html">Open photography archive</a>
          </div>
        </article>
      </div>
    </section>
  </div>`;
}

export function cvPage({ escapeHtml, awards }) {
  const recognition = awards.items.map((award) => `<li><strong>${escapeHtml(award.title)}</strong> — ${escapeHtml(award.meta)}</li>`).join('\n            ');
  return `<div class="container cv-shell">
    <section class="page-hero reveal accent-cool">
      <div class="kicker">Curriculum vitae</div>
      <h1 class="h1">Curriculum vitae</h1>
      <p class="lead">Ph.D. student in Electrical and Computer Engineering at the University of Florida, working on machine-learning surrogates for physics simulation, signal processing, time-series machine learning, and neuroengineering.</p>
      <div class="cta-row no-print">
        <button class="btn primary" type="button" data-print>Download as PDF</button>
        <a class="btn" href="contact.html">Contact</a>
      </div>
    </section>

    <section class="accent-mint">
      <h2 class="section-title reveal">Education</h2>
      <article class="card reveal">
        <ul class="cv-list">
          <li><strong>Ph.D., Electrical &amp; Computer Engineering</strong> — University of Florida, Gainesville, FL (expected 2030). Concentration: analysis, signal processing, and machine learning.</li>
          <li><strong>B.S., Electrical &amp; Computer Engineering</strong> — University of Florida (2020–2025)</li>
          <li><strong>STEM diploma</strong> — Doral Academy Charter School (2013–2020)</li>
        </ul>
      </article>
    </section>

    <section class="accent-amber">
      <h2 class="section-title reveal">Research experience</h2>
      <article class="card reveal">
        <h3>SmartDATA Lab, University of Florida</h3>
        <div class="meta">Graduate researcher · Advisor: Dr. Joel B. Harley · Fall 2025–present</div>
        <ul class="cv-list">
          <li>Surrogate modeling for computational fluid dynamics: learned stand-ins for expensive CFD solvers spanning neural operators (FNO/DeepONet), mesh-based graph networks, physics-informed training, and reduced-order latent dynamics, evaluated on rollout stability, physical consistency, and generalization.</li>
        </ul>
      </article>
      <article class="card reveal">
        <h3>Computational NeuroEngineering Lab (CNEL), University of Florida</h3>
        <div class="meta">Graduate researcher · Gainesville, FL</div>
        <!-- TODO(raul): confirm start date for CNEL work -->
        <ul class="cv-list">
          <li>Latent-signal classification with hierarchical dynamical systems: state-space modeling (Hierarchical Linear Dynamical Systems) that turns noisy multichannel observations into compact hidden time-series trajectories evaluated for prediction and separability.</li>
          <li>Zebrafish voltage imaging: statistical event detection for 2D voltage-imaging video using CFAR-style detectors and a Quadratic Gamma Discriminator, producing interpretable maps for downstream analysis.</li>
          <li>Time-series ML experiment infrastructure: standardized configs, runs, metrics, and plots so model comparisons stay fair and repeatable, with GPU acceleration.</li>
        </ul>
      </article>
      <article class="card reveal">
        <h3>IEEE Signal Processing Society at UF</h3>
        <div class="meta">Workshops, prototypes, and open-source systems</div>
        <ul class="cv-list">
          <li>Designed and taught the "Foundations of Signal Processing" workshop series connecting analysis, probability, and machine learning to applied signal-processing case studies (recorded at the DSI Spring Symposium 2025).</li>
          <li>Led student-facing research engineering projects including Aude (end-to-end audio research workflow: synchronized microphone-array data, benchmark state-of-the-art baselines, source separation and localization models).</li>
          <li>Taught analysis and machine learning with Jupyter to UF students, pairing theory with runnable notebooks.</li>
        </ul>
      </article>
    </section>

    <section class="accent-cool">
      <h2 class="section-title reveal">Publications and talks</h2>
      <article class="card reveal">
        <ul class="cv-list">
          <li><strong>Plato's Cave: A Human-Centered Research Verification System</strong> — arXiv preprint, March 2026. <a href="https://arxiv.org/abs/2603.23526" target="_blank" rel="noreferrer">arxiv.org/abs/2603.23526</a></li>
          <li><strong>Foundations of Signal Processing</strong> (talk) — UF Data Science &amp; Informatics, DSI Spring Symposium 2025. <a href="https://www.youtube.com/watch?v=yuJaMaA18js" target="_blank" rel="noreferrer">Recording</a></li>
        </ul>
      </article>
    </section>

    <section class="accent-mint">
      <h2 class="section-title reveal">Recognition</h2>
      <article class="card reveal">
        <ul class="cv-list">
            ${recognition}
        </ul>
      </article>
    </section>

    <section class="accent-amber">
      <h2 class="section-title reveal">Skills</h2>
      <div class="grid two" style="margin-top:14px; gap:14px;">
        <article class="card reveal">
          <h3>Methods</h3>
          <div class="pill-row">
            <span class="pill">Surrogate modeling</span>
            <span class="pill">Physics-informed ML</span>
            <span class="pill">Signal processing</span>
            <span class="pill">State-space modeling</span>
            <span class="pill">Time-series ML</span>
            <span class="pill">Statistical detection</span>
          </div>
        </article>
        <article class="card reveal">
          <h3>Tools</h3>
          <div class="pill-row">
            <span class="pill">Python</span>
            <span class="pill">NumPy</span>
            <span class="pill">PyTorch / CuPy (GPU)</span>
            <span class="pill">Jupyter</span>
            <span class="pill">Experiment tracking</span>
            <span class="pill">Embedded systems</span>
          </div>
        </article>
      </div>
    </section>

    <section class="accent-cool">
      <h2 class="section-title reveal">Affiliations and identifiers</h2>
      <div class="grid two" style="margin-top:14px; gap:14px;">
        <article class="card reveal">
          <h3>Affiliations</h3>
          <ul class="cv-list">
            <li>University of Florida</li>
            <li>UF Department of Electrical &amp; Computer Engineering</li>
            <li>SmartDATA Lab</li>
            <li>Computational NeuroEngineering Lab</li>
            <li>IEEE Signal Processing Society at the University of Florida</li>
          </ul>
        </article>
        <article class="card reveal">
          <h3>Identifiers</h3>
          <ul class="cv-list">
            <li>ORCID: <a href="https://orcid.org/0009-0004-0487-0086" target="_blank" rel="me noreferrer">0009-0004-0487-0086</a></li>
            <li><a href="https://scholar.google.com/citations?user=v5_9hm8AAAAJ&amp;hl=en" target="_blank" rel="me noreferrer">Google Scholar</a></li>
            <li><a href="https://github.com/Jibby2k1" target="_blank" rel="me noreferrer">GitHub</a></li>
            <li><a href="https://www.linkedin.com/in/raul-valle1/" target="_blank" rel="me noreferrer">LinkedIn</a></li>
          </ul>
        </article>
      </div>
      <article class="card reveal" style="margin-top:14px;">
        <h3>Official UF mentions</h3>
        <ul class="cv-list">
          <li><a href="https://ai.ufl.edu/teaching-with-ai/for-uf-faculty/ai-faculty-awards/biography/raul-valle.html" target="_blank" rel="noreferrer">UF AI biography</a> — HiPerGator Early Career Award profile.</li>
          <li><a href="https://news.ece.ufl.edu/2025/11/10/uf-student-hackers-enter-platos-cave-for-first-place-win/" target="_blank" rel="noreferrer">UF student hackers enter 'Plato's Cave' for first-place win</a> — UF ECE coverage.</li>
          <li><a href="https://www.ufdsi.com/symposium" target="_blank" rel="noreferrer">DSI Spring Symposium 2025</a> — workshop talk context.</li>
        </ul>
      </article>
      <div class="cta-row no-print" style="margin-top:14px;">
        <a class="btn primary" href="research.html">Research and publications</a>
        <a class="btn" href="blog.html">Notes and writing</a>
      </div>
    </section>
  </div>`;
}

export function contactPage({ site, escapeHtml }) {
  return `<div class="container">
      <section class="page-hero reveal accent-cool">
        <div class="kicker">Contact</div>
        <h1 class="h1">Contact Raul Valle</h1>
        <p class="lead">The fastest way to reach me is email. For research questions, start with the project pages so we can talk specifics; for collaboration, teaching, or anything UF-related, a short note with context is plenty.</p>
        <div class="cta-row">
          <a class="btn primary" href="mailto:${site.email}">${escapeHtml(site.email)}</a>
          <a class="btn" href="https://www.linkedin.com/in/raul-valle1/" target="_blank" rel="me noreferrer">LinkedIn</a>
        </div>
      </section>
      <section class="accent-mint">
        <h2 class="section-title reveal">Profiles</h2>
        <div class="grid two" style="gap:14px; margin-top:14px;">
          <article class="card reveal">
  <h3>ORCID</h3>
  <p>Persistent research identifier.</p>
  <div class="cta-row">
    <a class="btn primary" href="https://orcid.org/0009-0004-0487-0086" target="_blank" rel="me noreferrer">Open ORCID</a>
  </div>
</article><article class="card reveal">
  <h3>Google Scholar</h3>
  <p>Scholar profile connected to a verified ufl.edu identity.</p>
  <div class="cta-row">
    <a class="btn primary" href="https://scholar.google.com/citations?user=v5_9hm8AAAAJ&amp;hl=en" target="_blank" rel="me noreferrer">Open Scholar</a>
  </div>
</article><article class="card reveal">
  <h3>GitHub</h3>
  <p>Code, research tooling, and project work.</p>
  <div class="cta-row">
    <a class="btn primary" href="https://github.com/Jibby2k1" target="_blank" rel="me noreferrer">Open GitHub</a>
  </div>
</article><article class="card reveal">
  <h3>X</h3>
  <p>Public short-form updates.</p>
  <div class="cta-row">
    <a class="btn primary" href="https://x.com/Jibby2k1" target="_blank" rel="me noreferrer">Open X</a>
  </div>
</article><article class="card reveal">
  <h3>YouTube</h3>
  <p>Talks and UF research coverage.</p>
  <div class="cta-row">
    <a class="btn primary" href="https://www.youtube.com/watch?v=yuJaMaA18js" target="_blank" rel="noreferrer">Open talk video</a>
  </div>
</article><article class="card reveal">
  <h3>Instagram</h3>
  <p>Short-form research and photography updates.</p>
  <div class="cta-row">
    <a class="btn primary" href="https://www.instagram.com/raul.research/" target="_blank" rel="me noreferrer">Open Instagram</a>
  </div>
</article>
        </div>
      </section>
      <section class="accent-amber">
        <h2 class="section-title reveal">Verification</h2>
        <p class="section-subtitle reveal">If you need to confirm you are looking at the right Raul Valle, these official University of Florida references are the shortest path.</p>
        <div class="grid two" style="gap:14px; margin-top:14px;">
          <article class="card reveal">
  <h3>UF AI biography</h3>
  <p>Official University of Florida profile for the HiPerGator Early Career Award.</p>
  <div class="cta-row">
    <a class="btn primary" href="https://ai.ufl.edu/teaching-with-ai/for-uf-faculty/ai-faculty-awards/biography/raul-valle.html" target="_blank" rel="noreferrer">Open UF AI bio</a>
  </div>
</article><article class="card reveal">
  <h3>UF ECE news</h3>
  <p>UF Electrical &amp; Computer Engineering coverage of the Plato's Cave team and first-place finish.</p>
  <div class="cta-row">
    <a class="btn primary" href="https://news.ece.ufl.edu/2025/11/10/uf-student-hackers-enter-platos-cave-for-first-place-win/" target="_blank" rel="noreferrer">Open UF ECE news</a>
  </div>
</article><article class="card reveal">
  <h3>UF ECE</h3>
  <p>Department context for Electrical and Computer Engineering at UF.</p>
  <div class="cta-row">
    <a class="btn primary" href="https://www.ece.ufl.edu/" target="_blank" rel="noreferrer">Open UF ECE</a>
  </div>
</article><article class="card reveal">
  <h3>DSI Spring Symposium</h3>
  <p>Event context for the signal-processing workshop talk and public video.</p>
  <div class="cta-row">
    <a class="btn primary" href="https://www.ufdsi.com/symposium" target="_blank" rel="noreferrer">Open symposium page</a>
  </div>
</article>
        </div>
        <article class="card reveal" style="margin-top:14px;">
          <h3>Research outputs</h3>
          <p>Papers, talk videos, and official mentions are collected with the research overview.</p>
          <div class="cta-row">
            <a class="btn primary" href="research.html">Open research</a>
            <a class="btn" href="cv.html">Open CV</a>
          </div>
        </article>
      </section>
    </div>`;
}
