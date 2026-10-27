// Avatar playground: a Gaussian-splat avatar of me, built from two phone videos.
// The page plays frames the model rendered offline (assets/avatar/*.webp sprite
// sheets, rows = turn angle, columns = animation frame); assets/js/avatar-viewer.js
// only chooses which frame to draw.
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
          <a class="btn" href="#tools">The two-camera pipeline</a>
        </div>
      </section>

      <section id="play" class="card reveal avatar-play" data-avatar-viewer data-base="${prefix}assets/avatar/">
        <div class="avatar-stage">
          <canvas width="180" height="220" role="img" aria-label="The avatar, rendered by the model. Drag sideways to turn it."></canvas>
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
            <p>I never waved both hands at once, squatted or did jumping jacks on camera. Each move is written as joint rotations for the skeleton, with smooth timing and follow-through. Then the legs are solved frame by frame so that a planted foot stays flat and fixed on the floor while the body shifts its weight over it, and in jumping jacks the body dips, leaves the ground on a ballistic arc and lands to absorb it. The learned appearance follows. That is the point of an avatar: it can do things the person was never recorded doing.</p>
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
            <p>The first version tracked me well only while I faced the camera. Holding my position during the turn, fitting my shape from every side, and adding my real measurements cut the tracking error from about 40 pixels side-on to about 10 across the whole turn, so the sides and back are now learned too. Comparing my outline in a close crop around me, instead of across the whole frame, tightened the fit again, and the avatar now matches the held-out half of the motion clip noticeably better. A last per-frame touch-up of every pose, matched to my outline as the avatar itself draws it with the clothes included, added a little more. The sides and back are still softer than the front.</p>
          </article>
          <article class="card reveal">
            <h3>Poses far from anything filmed</h3>
            <p>Training on part of the waving clip fixed most of the streaks on raised arms. Surfaces the camera never saw, like the undersides of my arms and my armpits, are now coloured from their well-seen neighbours instead of being left grey, but that colour is an educated guess, and deep poses like a squat are still blurrier.</p>
          </article>
        </div>
      </section>

      <section class="accent-amber">
        <h2 class="section-title reveal">Next: a proper capture session</h2>
        <div class="grid two">
          <article class="card reveal">
            <h3>What is planned</h3>
            <p>Most of what still limits this avatar is the recording, not the model: one 720p phone with a guessed lens, loose clothes, and arms that were never raised during the turn. The next session uses two cameras calibrated with a checkerboard and synced with a clap, and a helper who films me holding still from every side at three heights, with my arms down and then raised. I will wear fitted clothes, record a library of the playground moves for real, and take a full set of body measurements.</p>
          </article>
          <article class="card reveal">
            <h3>A fair test</h3>
            <p>One take from that session will be set aside unopened and scored only once, after every choice about the model is final. Today's held-out score comes from later in the same clip the model learned from, and I used it to choose between versions, so it is a development number, not a sealed test.</p>
          </article>
        </div>
      </section>

      <section id="tools" class="accent-cool">
        <h2 class="section-title reveal">Built for the next recording</h2>
        <p class="lead reveal">The software for that session is already written. The core pieces were tested on synthetic data where the right answer is known exactly, against pass marks set before the code existed. Several failed at first; each failure and what changed because of it is written down. None of it has met a real recording yet.</p>
        <div class="grid two">
          <article class="card reveal">
            <h3>Measuring each lens</h3>
            <p>A short video of a printed checkerboard gives each camera's focal length, centre and lens distortion, replacing today's guessed field of view. The tests taught me a practical rule: the board has to visit all four corners of the frame, or the lens model there is a guess, so the tool now warns when a corner was missed.</p>
          </article>
          <article class="card reveal">
            <h3>Lining up two cameras with a clap</h3>
            <p>A sharp clap at the start and end of every take is found in both soundtracks, which aligns the two cameras to within a fraction of a millisecond in tests, corrects for one clock running slightly fast, and pairs frames by their own timestamps (phones do not record at a perfectly steady rate).</p>
          </article>
          <article class="card reveal">
            <h3>Where the second camera stands</h3>
            <p>Holding the board where both cameras see it gives the second camera's position and angle relative to the first; laying it on the floor gives the floor. On a simulated version of the planned setup, both come out within a twentieth of a degree and a few millimetres.</p>
          </article>
          <article class="card reveal">
            <h3>Tracking from two views at once</h3>
            <p>The body is fitted to both cameras' outlines and joints together. On a simulated turn in place, the average joint error fell from 29 to 66 millimetres with one camera to 5 to 7 millimetres with two, with front, side and back views all improving by a similar amount.</p>
          </article>
          <article class="card reveal">
            <h3>Thirty-six tape measurements</h3>
            <p>Lengths such as shoulder to elbow and girths such as calf and wrist are defined on the body model the way a tape sees them, so a measured body can be fitted to them directly. On simulated bodies they pin the shape to about 5 millimetres, against 11 with the four numbers I gave last time.</p>
          </article>
          <article class="card reveal">
            <h3>A fix that helped the avatar today</h3>
            <p>Building the two-camera tracker exposed a flaw in the one I already had: it compared my outline at a scale where its own rendering of the body was blurry and full of gaps. Comparing it in a close crop around me fixed that, and re-tracking the existing videos with the fix is what made the avatar above better on the held-out clip.</p>
          </article>
        </div>
        <article class="card reveal">
          <h3>How the session will run</h3>
          <p>Two cameras on tripods, about three metres away and seventy-five degrees apart, with focus, exposure and white balance locked and stabilisation off. First the checkerboard for each lens, then for both cameras together, then flat on the floor. Then a helper walks a slow circle around me at chest height, overhead and knee height while I hold still, arms out and then raised; I turn in place for both cameras; I perform each playground move for real; the helper films my face and hands up close; and one final take of mixed movement is set aside unopened as the sealed test. Every take starts and ends with a clap, and I take about thirty-six body measurements, arms and legs on both sides, twice each.</p>
        </article>
      </section>

      <section class="accent-mint">
        <h2 class="section-title reveal">Credits</h2>
        <article class="card reveal">
          <p>Built on <a href="${GSM}" target="_blank" rel="noreferrer">Gaussian Shell Maps</a> (Abdal et al., CVPR 2024) and 3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023), with the SMPL body model (Loper et al., 2015), torchvision's Mask R-CNN and Keypoint R-CNN (He et al., 2017) and PyTorch3D. The calibration, synchronisation and measurement tools use OpenCV, SciPy and FFmpeg. The fitting, tracking and evaluation layer that turns phone videos into this avatar is ours, written with help from Claude Code and OpenAI Codex; the code is kept in a private repository.</p>
        </article>
      </section>
    </div>
    <script src="${prefix}assets/js/avatar-viewer.js" defer></script>`;
}
