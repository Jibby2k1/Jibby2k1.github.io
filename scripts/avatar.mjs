// Avatar playground: a Gaussian-splat avatar of me, built from two phone videos.
// The page plays frames the model rendered offline (assets/avatar/*.webp sprite
// sheets, rows = turn angle, columns = animation frame); assets/js/avatar-viewer.js
// only chooses which frame to draw.
const REPO = 'https://github.com/Jibby2k1/Avatar';
const GSM = 'https://github.com/computational-imaging/GSM';

export function avatarPage({ prefix }) {
  const moves = [
    ['wave', 'Wave both hands'],
    ['squat', 'Squat'],
    ['jacks', 'Jumping jacks'],
    ['march', 'March'],
    ['arms', 'Arms up'],
    ['stand', 'Stand still']
  ];
  return `<div class="container">
      <section class="page-hero reveal accent-cool">
        <div class="kicker">Hackathon project · Gaussian avatars</div>
        <h1 class="h1">A 3-D avatar of me, from two phone videos</h1>
        <p class="lead">I filmed myself turning in place, then filmed a few seconds of waving. From those two clips, a model learned what I look like as about 100,000 tiny coloured blobs (3-D Gaussians) wrapped around a digital skeleton, and that skeleton can be posed any way you like. Try it below.</p>
        <div class="cta-row">
          <a class="btn primary" href="#play">Play with the avatar</a>
          <a class="btn" href="#how">How it works</a>
          <a class="btn" href="${REPO}" target="_blank" rel="noreferrer">Code on GitHub</a>
        </div>
      </section>

      <section id="play" class="card reveal avatar-play" data-avatar-viewer data-base="${prefix}assets/avatar/">
        <div class="avatar-stage">
          <canvas width="189" height="220" role="img" aria-label="The avatar, rendered by the model. Drag sideways to turn it."></canvas>
          <noscript><p>The playground needs JavaScript. The video further down shows the avatar in motion.</p></noscript>
        </div>
        <div class="avatar-controls">
          <div class="avatar-moves" role="group" aria-label="Choose a move">
            ${moves.map(([key, name]) => `<button type="button" class="filter-btn" data-move="${key}" aria-pressed="false">${name}</button>`).join('\n            ')}
          </div>
          <label class="avatar-turn">Turn
            <input type="range" min="0" max="8" step="1" value="4" data-avatar-turn aria-describedby="avatar-turn-hint" />
          </label>
          <p id="avatar-turn-hint" class="filter-hint">Drag the avatar or use the slider to turn it up to 60° either way.</p>
          <div class="cta-row">
            <button type="button" class="btn small" data-avatar-play aria-pressed="false">Play</button>
          </div>
          <p class="avatar-status" data-avatar-status aria-live="polite">Loading the avatar…</p>
        </div>
      </section>

      <section class="accent-mint">
        <h2 class="section-title reveal">What you are seeing</h2>
        <div class="grid two">
          <article class="card reveal">
            <h3>Moves I never filmed</h3>
            <p>I never waved both hands at once, squatted or did jumping jacks on camera. Each move is written as joint rotations for the skeleton, with smooth timing and follow-through (elbows trail shoulders, the trunk leans into a squat), and the feet stay planted on the floor. The learned appearance follows. That is the point of an avatar: it can do things the person was never recorded doing.</p>
          </article>
          <article class="card reveal">
            <h3>Every frame is a real render</h3>
            <p>Each frame here was rendered by the model on my GPU, at nine turn angles and sixteen moments per move. The page just plays them back, so it is fast on any phone. The colours and shading are the model's own; nothing is drawn by hand.</p>
          </article>
        </div>
        <figure class="card reveal avatar-video">
          <video controls muted loop playsinline preload="none" poster="${prefix}assets/avatar/avatar-copies-me-poster.webp" width="720" height="360">
            <source src="${prefix}assets/avatar/avatar-copies-me.mp4" type="video/mp4" />
          </video>
          <figcaption>Left: the second half of my motion clip, which the model never trained on. Right: the avatar copying it, posed from the motion tracked in that clip.</figcaption>
        </figure>
      </section>

      <section id="how" class="accent-amber">
        <h2 class="section-title reveal">How it works</h2>
        <div class="grid two">
          <article class="card reveal">
            <h3>1 · Two short videos</h3>
            <p>A slow turn in place (about 30 seconds) and a few seconds of free movement, on a phone, against a plain wall, with no markers or special suit. Plus four numbers: my height and my chest, waist and hip measurements.</p>
          </article>
          <article class="card reveal">
            <h3>2 · Tracking the body</h3>
            <p>Every frame is cut out automatically, and a standard body model (SMPL) is fitted to it using the silhouette and detected joints, nose, eyes and ears. The body's shape is fitted to views from all sides and to my measurements, and kept inside the outline of my clothes rather than inflated to fill it. Only frames used for training inform the shape, so the held-out half of the motion clip stays a fair test.</p>
          </article>
          <article class="card reveal">
            <h3>3 · Learning the appearance</h3>
            <p>Following Gaussian Shell Maps (Abdal et al., CVPR 2024), five thin shells around the body each carry a texture of 3-D Gaussians. Their colour, opacity, size and orientation are adjusted until renders match the turn and the first half of the motion clip, which takes about two minutes on one consumer GPU.</p>
          </article>
          <article class="card reveal">
            <h3>4 · Posing it</h3>
            <p>Because the Gaussians ride on shells bound to the skeleton, any new pose moves them along. Turning, waving and squatting are all just new skeleton poses.</p>
          </article>
        </div>
      </section>

      <section class="accent-cool">
        <h2 class="section-title reveal">Honest limits</h2>
        <div class="grid two">
          <article class="card reveal">
            <h3>Sharpest from the front</h3>
            <p>The first version tracked me well only while I faced the camera. Holding my position during the turn, fitting my shape from every side, and adding my real measurements cut the tracking error from about 42 to 9 pixels, so the sides and back are now learned too. They are still softer than the front.</p>
          </article>
          <article class="card reveal">
            <h3>Poses far from anything filmed</h3>
            <p>Training on part of the waving clip fixed most of the streaks on raised arms. Surfaces the camera never saw, like the undersides of my arms and my armpits, are now coloured from their well-seen neighbours instead of being left grey, but that colour is an educated guess, and deep poses like a squat are still blurrier. A longer capture with more varied movement is the next step.</p>
          </article>
        </div>
      </section>

      <section class="accent-mint">
        <h2 class="section-title reveal">Credits</h2>
        <article class="card reveal">
          <p>Built on <a href="${GSM}" target="_blank" rel="noreferrer">Gaussian Shell Maps</a> (Abdal et al., CVPR 2024) and 3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023), with the SMPL body model (Loper et al., 2015), torchvision's Mask R-CNN and Keypoint R-CNN (He et al., 2017) and PyTorch3D. The fitting and tracking layer that turns two phone videos into this avatar is ours, written with help from Claude Code.</p>
          <div class="cta-row">
            <a class="btn" href="${REPO}" target="_blank" rel="noreferrer">Code and benchmark results</a>
          </div>
        </article>
      </section>
    </div>
    <script src="${prefix}assets/js/avatar-viewer.js" defer></script>`;
}
