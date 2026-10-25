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
          <canvas width="247" height="240" role="img" aria-label="The avatar, rendered by the model. Drag sideways to turn it."></canvas>
          <noscript><p>The playground needs JavaScript. The video further down shows the avatar in motion.</p></noscript>
        </div>
        <div class="avatar-controls">
          <div class="avatar-moves" role="group" aria-label="Choose a move">
            ${moves.map(([key, name]) => `<button type="button" class="filter-btn" data-move="${key}" aria-pressed="false">${name}</button>`).join('\n            ')}
          </div>
          <label class="avatar-turn">Turn
            <input type="range" min="0" max="6" step="1" value="3" data-avatar-turn aria-describedby="avatar-turn-hint" />
          </label>
          <p id="avatar-turn-hint" class="filter-hint">Drag the avatar or use the slider to turn it up to 30° either way.</p>
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
            <p>I never waved both hands at once, squatted or did jumping jacks on camera. Each move is typed in as joint angles for the skeleton, and the learned appearance follows it. That is the point of an avatar: it can do things the person was never recorded doing.</p>
          </article>
          <article class="card reveal">
            <h3>Every frame is a real render</h3>
            <p>Each frame here was rendered by the model on my GPU, at seven turn angles and fourteen moments per move. The page just plays them back, so it is fast on any phone. The colours and shading are the model's own; nothing is drawn by hand.</p>
          </article>
        </div>
        <figure class="card reveal avatar-video">
          <video controls muted loop playsinline preload="none" poster="${prefix}assets/avatar/avatar-copies-me-poster.webp" width="720" height="360">
            <source src="${prefix}assets/avatar/avatar-copies-me.mp4" type="video/mp4" />
          </video>
          <figcaption>Left: a clip of me the model never trained on. Right: the avatar copying it, posed from the motion tracked in that clip.</figcaption>
        </figure>
      </section>

      <section id="how" class="accent-amber">
        <h2 class="section-title reveal">How it works</h2>
        <div class="grid two">
          <article class="card reveal">
            <h3>1 · Two short videos</h3>
            <p>A slow turn in place (about 30 seconds) for appearance, and a few seconds of free movement to copy. Phone camera, plain wall, no markers or special suit.</p>
          </article>
          <article class="card reveal">
            <h3>2 · Tracking the body</h3>
            <p>Every frame is cut out automatically, and a standard body model (SMPL) is fitted to it using the silhouette and 17 detected joints. The result is a skeleton pose for every frame.</p>
          </article>
          <article class="card reveal">
            <h3>3 · Learning the appearance</h3>
            <p>Following Gaussian Shell Maps (Abdal et al., CVPR 2024), five thin shells around the body each carry a texture of 3-D Gaussians. Their colour, opacity, size and orientation are adjusted until renders match the video, which takes about two minutes on one consumer GPU.</p>
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
            <h3>The front is sharp, the back is not</h3>
            <p>Tracking was accurate while I faced the camera and much worse side-on or from behind. So this avatar was trained only on the well-tracked front-facing frames, which is why turning is limited to 30°.</p>
          </article>
          <article class="card reveal">
            <h3>Raised arms smear</h3>
            <p>The undersides of my arms were barely visible in the turn, so arms raised far from where they were filmed show streaks. Better tracking and a longer capture are the next steps.</p>
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
