// Landing page for NeuRev Review, the browser tool reviewers use to label calcium events.
// The tool itself is vendor/neurev-review/app.html.vendored (a copy of the NeuRev-Workbench labeler), published unchanged
// next to this page. Neither page carries data: reviewers open the pack they were sent on their own computer.
const DOCS = 'https://github.com/Jibby2k1/NeuRev-Workbench/blob/main/neurobench/tools/event_labeler/README.md';

export function neurevReviewPage({ imgTag, prefix }) {
  return `<div class="container">
      <section class="page-hero reveal accent-cool">
        <div class="kicker">NeuRev · Computational NeuroEngineering Lab</div>
        <h1 class="h1">NeuRev Review</h1>
        <p class="lead">A browser tool for labelling calcium events in zebrafish light-sheet recordings. A detector ranks the moments most likely to be a neuron firing; you look at each one as a short clip and answer <strong>Accept</strong>, <strong>Decline</strong> or <strong>Unsure</strong>, usually in a few seconds.</p>
        <div class="cta-row">
          <a class="btn primary" href="app.html">Open NeuRev Review</a>
          <a class="btn" href="#how">How it works</a>
          <a class="btn" href="${DOCS}" target="_blank" rel="noreferrer">Documentation</a>
        </div>
      </section>

      <figure class="card reveal neurev-shot">
        ${imgTag('assets/img/projects/neurev-review.webp', {
          prefix,
          eager: true,
          alt: 'NeuRev Review showing one candidate: the raw clip with centre and annulus circles, the same clip after local standardization glowing at the centre, and a trace where the centre rises far above its surroundings, with Decline, Unsure and Accept buttons.'
        })}
        <figcaption>One candidate, as a reviewer sees it (synthetic recording).</figcaption>
      </figure>

      <section id="how" class="accent-mint">
        <h2 class="section-title reveal">How it works</h2>
        <div class="grid two">
          <article class="card reveal">
            <h3>1 · Get your pack</h3>
            <p>The study coordinator sends you a pack: one <code>.zip</code> file of short clips. Keep it zipped.</p>
          </article>
          <article class="card reveal">
            <h3>2 · Open it here</h3>
            <p>Open NeuRev Review, enter your name and your experience with calcium imaging, and choose the pack.</p>
          </article>
          <article class="card reveal">
            <h3>3 · Review at your own pace</h3>
            <p>Pick a goal for the sitting (25 items, 20 minutes, …) and how much of the ranked queue to cover. Every answer is saved in your browser as you go; close the tab and come back any time.</p>
          </article>
          <article class="card reveal">
            <h3>4 · Send your results</h3>
            <p>Press <strong>Download results</strong> and send the zip back. It holds your answers, any notes, and a record of how you reviewed each clip.</p>
          </article>
        </div>
      </section>

      <section class="accent-amber">
        <h2 class="section-title reveal">Good to know</h2>
        <div class="grid two">
          <article class="card reveal">
            <h3>Nothing is uploaded</h3>
            <p>The pack is read on your own computer and your answers stay in your browser until you download them. This site never receives the recording, the pack, or your results.</p>
          </article>
          <article class="card reveal">
            <h3>Deciding an item</h3>
            <p>Accept when a cell at the centre lights up: the centre brightens more than the ring around it, then fades. Decline noise, flicker, movement, or a brightening of the whole area. Use Unsure, and a short note, when you cannot tell.</p>
          </article>
          <article class="card reveal">
            <h3>Keyboard</h3>
            <p><kbd>A</kbd> accept · <kbd>D</kbd> decline · <kbd>U</kbd> unsure · <kbd>N</kbd> note · <kbd>B</kbd> back · <kbd>Space</kbd> play or stop · <kbd>P</kbd> pause or change your goal.</p>
          </article>
          <article class="card reveal">
            <h3>For study coordinators</h3>
            <p>Packs are built with the NeuRev-Workbench event labeler (<code>build_pack.py --zip</code>), which keeps the answer key on your side; <code>summarize.py</code> reads the results zips reviewers send back.</p>
            <div class="cta-row">
              <a class="btn" href="${DOCS}" target="_blank" rel="noreferrer">Read the guide</a>
              <a class="btn text" href="${prefix}projects/zebrafish-voltage-imaging-detection.html">About the NeuRev project</a>
            </div>
          </article>
        </div>
      </section>
    </div>`;
}
