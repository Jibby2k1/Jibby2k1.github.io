// Recording guide for the avatar's next capture session (avatar/capture/index.html):
// one camera that moves between takes, plus a helper. The same protocol is kept as a
// Markdown working copy offline; this page is the shareable, printable copy.

const table = (caption, head, rows) => `<div class="guide-table-wrap" role="region" tabindex="0" aria-label="${caption.replace(/<[^>]+>/g, '')}">
          <table class="guide-table">
            <caption>${caption}</caption>
            <thead><tr>${head.map((h) => `<th scope="col">${h}</th>`).join('')}</tr></thead>
            <tbody>
${rows.map((r) => `              <tr>${r.map((c, i) => (i === 0 ? `<th scope="row">${c}</th>` : `<td data-label="${head[i]}">${c}</td>`)).join('')}</tr>`).join('\n')}
            </tbody>
          </table>
        </div>`;

const list = (items) => `<ul>${items.map((x) => `<li>${x}</li>`).join('')}</ul>`;

const take = (n, title, who, body) => `<article class="card reveal guide-take">
            <p class="guide-take-head"><span class="guide-take-num">Take ${n}</span><span class="guide-take-who">${who}</span></p>
            <h3>${title}</h3>
            ${body}
          </article>`;

export function capturePage({ prefix }) {
  const board = (size) => `${prefix}assets/avatar/checkerboard-${size}.pdf`;
  const settings = table('Camera settings: lock every one before take 0 and never change it mid-session', ['Setting', 'Value', 'Why'], [
    ['Camera app', 'A manual video app that can lock focus, exposure and white balance, set the shutter, fix the frame rate and turn stabilisation off (for example Blackmagic Camera, free on iPhone and Android). In a stock camera app, a long-press locks only focus and exposure.', 'The stock app cannot hold every setting below.'],
    ['Lens and zoom', 'The main 1× lens only. No zoom, never switched.', 'The calibration holds for one lens at one zoom.'],
    ['Orientation', 'Portrait, for every take.', 'At 2–2.5 m only portrait fits a whole standing body. One calibration per orientation.'],
    ['Resolution and rate', '4K at a fixed 30 fps (60 fps if the light allows). Any automatic low-light frame rate off.', 'More pixels on the board and on the body, at a steady rate.'],
    ['Stabilisation', 'Off, including electronic and “action” modes. Optical stabilisation usually cannot be disabled; that is fine.', 'Electronic stabilisation warps every frame differently.'],
    ['Focus, exposure, white balance', 'All three locked, with focus on the subject at about 2.5 m.', 'A focus change shifts the lens model; brightness or colour changes count as error.'],
    ['Shutter', '1/250 s or faster, if the light allows.', 'The camera moves during the orbits.'],
    ['HDR, night, beauty, portrait blur', 'Off.', 'These alter appearance frame by frame.']
  ]);
  const takes = table('The takes, in order (about 40 minutes)', ['#', 'Take', 'Camera', 'Length', 'Files'], [
    ['0', 'Calibration, start', 'Helper or tripod; the subject moves the board', '45–60 s', '<code>0</code>'],
    ['1', '<strong>Static orbit, A-pose</strong>', 'Helper, handheld', '3 passes × 30–40 s', '<code>1a</code> <code>1b</code> <code>1c</code>'],
    ['2', '<strong>Static orbit, arms up</strong>', 'Helper, handheld', '2 passes × 30 s', '<code>2a</code> <code>2b</code>'],
    ['3', 'Turn in place', 'Tripod, front', '2 slow turns arms out, 1 arms up', '<code>3</code>'],
    ['4', '<strong>Movement library</strong>', 'Tripod, front; then moved 60–90° to the side', 'About 20 s per move, per position', '<code>4-front</code> <code>4-side</code>'],
    ['5', '<strong>Sealed test</strong>', 'Tripod at a position no other take uses (e.g. 45°)', '60 s', '<code>sealed</code>'],
    ['6', 'Calibration, end', 'As in take 0', '45 s', '<code>6</code>'],
    ['7', 'Face and hands (optional, last)', 'Helper, close', '60 s, plus a 45 s board clip', '<code>7</code> <code>7-board</code>']
  ]);
  const measures = table('Body measurements: twice each, to the nearest 0.5 cm, arms and legs on both sides', ['Measurement', 'Landmark / how'], [
    ['Height', 'Barefoot, heels to a wall, a flat object on the head'],
    ['Weight', ''],
    ['Arm span', 'Arms straight out at shoulder height, fingertip to fingertip'],
    ['Sitting height', 'Seat to the top of the head, sitting upright'],
    ['Shoulder width', 'Bony point to bony point on top of each shoulder (acromion), in a straight line: a ruler, or a tape held straight behind the shoulders, not following the curve of the back'],
    ['Upper arm', 'Arm hanging straight and relaxed: acromion to the point of the elbow'],
    ['Forearm', 'Arm straight: the point of the elbow to the wrist bone on the little-finger side'],
    ['Hand length', 'Wrist crease to the middle fingertip'],
    ['Leg heights', 'Standing, measured straight up from the floor to: the bony bump on the outside of the upper thigh (it rolls under your fingers when you turn the foot in and out), the knee joint (the soft gap at the side of the knee), and the inner ankle bone'],
    ['Inseam', 'Crotch to the floor, standing'],
    ['Foot length and width', 'Heel to the longest toe; the widest point across the ball'],
    ['Head, neck, chest, waist, hip girths', 'Tape level and snug, not tight; the waist at its narrowest, the hips at their widest'],
    ['Upper arm, forearm, wrist girths', 'Widest point, widest point, at the wrist bone'],
    ['Mid-thigh, calf, ankle girths', 'Halfway between hip and knee; widest point; narrowest point'],
    ['Sole thickness', 'Only if shoes are worn in the takes']
  ]);
  const moves = ['wave with both hands', 'squats', 'jumping jacks', 'march in place', 'arms raised up and back down', 'arm circles', 'side bends', 'torso twists', 'a few steps forward and back'];

  return `<div class="container guide">
      <section class="page-hero reveal">
        <div class="kicker">Avatar · Recording guide</div>
        <h1 class="h1">How to film someone for a 3&#8209;D avatar, with one camera</h1>
        <p class="lead">This is the protocol for the next recording behind <a href="${prefix}avatar/index.html">my avatar</a>. It needs one phone, a helper, a printed checkerboard and a tape measure. The camera can move freely between takes, because what the software needs to know about it belongs to the lens, not to where it stands.</p>
        <dl class="guide-facts">
          <div><dt>Camera</dt><dd>1, moved between takes</dd></div>
          <div><dt>People</dt><dd>Subject + helper</dd></div>
          <div><dt>Takes</dt><dd>8, in 13 short files</dd></div>
          <div><dt>Time</dt><dd>About 40 min</dd></div>
        </dl>
        <div class="cta-row no-print">
          <a class="btn primary" href="#takes">Jump to the takes</a>
          <a class="btn" href="#measure">Measurements</a>
          <a class="btn" href="${board('a3')}">Checkerboard (A3 PDF)</a>
          <button type="button" class="btn" onclick="window.print()">Print this guide</button>
        </div>
      </section>

      <section id="before">
        <h2 class="section-title reveal">Before you start</h2>
        <div class="grid two">
          <article class="card reveal">
            <h3>Equipment</h3>
            ${list([
              '<strong>One phone or camera</strong>, on its main 1× lens.',
              '<strong>A tripod</strong>, or any stable mount, for the movement takes.',
              '<strong>A helper</strong> for the handheld orbits.',
              `<strong>A checkerboard</strong>: print the <a href="${board('a3')}">A3 board</a> (30 mm squares) or, if A3 is not possible, the <a href="${board('a4')}">A4 board</a> (22 mm squares) at 100 % / “Actual size”, never “Fit to page”. Glue it flat to stiff card or foam board, keep the white border, and measure one square to 0.5 mm.`,
              '<strong>Floor tape and a tape measure.</strong>'
            ])}
          </article>
          <article class="card reveal">
            <h3>The room</h3>
            ${list([
              '<strong>Even, diffuse light</strong> from several sides. Avoid a window whose light changes and hard shadows, and keep the helper’s shadow off the subject during the orbits.',
              '<strong>The whole room is background</strong>, because the orbit films every direction. No mirrors, TV screens or reflections, and nobody else in frame: the person detector must only ever see the subject. Walls that contrast with the clothes help.',
              '<strong>Floor marks</strong>: a tape cross where the subject stands, and two foot marks hip-width apart, so every take starts from the same spot.'
            ])}
          </article>
        </div>
        <article class="card reveal guide-block">
          ${settings}
          <p class="guide-note">Do not close the camera app or tap the screen between takes. If a lock is lost, re-lock it and write that down.</p>
          <p class="guide-note">Copy the original files off the phone by cable or file copy, not through a messaging app, which re-compresses the video. On an iPhone, first set Settings › Photos › Transfer to Mac or PC to “Keep Originals”.</p>
        </article>
      </section>

      <section id="takes">
        <h2 class="section-title reveal">The takes</h2>
        <article class="card reveal guide-block">
          ${takes}
          <p class="guide-note">Stop recording between passes and before moving the camera or the tripod, so each pass or position is its own file. Name the files as in the Files column.</p>
        </article>
        <div class="grid two guide-takes">
          ${take(0, 'Calibrate the lens', 'Helper or tripod · 45–60 s', list([
            'The camera stays still, on the tripod or in the helper’s hands, with the exact recording settings and locks. The subject holds the board about 2–2.5 m away and moves it.',
            'Move it slowly through the whole frame: the centre, <strong>all four corners</strong> and the edges. Spend most of the time on the corners.',
            'Tilt it by as much as 45° each way: left, right, up and down.',
            'Skip the corners and the lens model there is a guess. The calibration tool warns when one was missed.'
          ]))}
          ${take(1, 'Static orbit in an A-pose', 'Helper, handheld · 3 passes', list([
            '<strong>The subject</strong> stands on the foot marks, arms straight about 45° from the body, palms forward, chin level, eyes on a fixed point, and holds as still as is comfortable, breathing normally.',
            '<strong>The helper</strong> walks one full, slow circle per pass at a constant 2–2.5 m, keeping the whole body in frame and roughly centred, at the same camera height all the way round.',
            '<strong>Pass 1:</strong> camera at chest height, level. <strong>Pass 2:</strong> held high above the helper’s own head, tilted down, to see the top of the head, the shoulders and the tops of the feet. <strong>Pass 3:</strong> at knee height, tilted up, to see under the chin and arms and the inner legs.',
            'Stop recording after each pass. Rest, then return to the marks and the same pose.'
          ]) + `<p class="guide-note">Why a steady distance and height: the tracker treats an orbit like the subject turning in place and holds the body near where it stood in the first frame. Keeping the body centred, at a constant distance and height, keeps that true.</p>`)}
          ${take(2, 'Static orbit, arms up', 'Helper, handheld · 2 passes', list([
            'Arms raised overhead in a wide V, palms forward.',
            'One pass at chest height, one low and tilted up.',
            'This take is meant to remove the grey speckle under the arms.'
          ]))}
          ${take(3, 'Turn in place', 'Tripod, front · 3 slow turns', list([
            'The tripod close enough that the body fills most of the frame height, with about 20 cm to spare above the head and below the feet (about 2.5–3 m in portrait). Lens at waist-to-chest height, floor visible.',
            'Two slow turns on the cross with the arms out, then one with the arms raised in a wide V.'
          ]))}
          ${take(4, 'Movement library', 'Tripod, two positions · ~20 s per move', `<p>Same framing as take 3. Each move once slowly, then once at natural speed:</p>${list(moves)}<p>Then stop recording, move the tripod 60–90° around the subject, and do the library again as a new file. That is not a second camera, because the two views are not simultaneous, but it gives the model a second viewpoint of every move.</p>`)}
          ${take(5, 'The sealed test', 'Tripod, its own position · 60 s', list([
            'Mixed movement, including at least one move not in take 4: lunges, reaching to the floor, or a slow spin.',
            'Filmed from a tripod position no other take uses, for example 45°.',
            'Name the file <code>sealed</code>, do not open or share it, and keep it apart from the rest.',
            'It is scored once, after every choice about the model is final. That is what makes a later quality claim honest.'
          ]))}
          ${take(6, 'Calibrate again', 'As take 0 · 45 s', `<p>The same as take 0. If the two calibrations disagree, a setting slipped during the session.</p>`)}
          ${take(7, 'Face and hands (optional, last)', 'Helper, close · 60 s + 45 s', list([
            'A slow arc about 1 m from the head: a neutral expression, then a smile.',
            'Each hand held up, front and back, fingers spread, then closed.',
            'This distance needs refocusing, which slightly changes the lens model. Lock focus on the face, and before unlocking, film another board clip (<code>7-board</code>) at about 1 m.'
          ]))}
        </div>
      </section>

      <section id="wear">
        <h2 class="section-title reveal">What to wear</h2>
        <article class="card reveal">
          ${list([
            '<strong>Fitted, lightly textured clothes</strong>, such as a fitted long-sleeve top and leggings. Avoid all black, which hides both detail and shape, and loose fabric, which moves in ways a body model cannot represent.',
            '<strong>The same outfit, hair and shoes in every take.</strong> Barefoot, or the shoes the avatar should wear.',
            '<strong>Hair tied back</strong>, or the same way throughout.',
            '<strong>Nothing reflective</strong>: no watch, jewellery or glossy logos.'
          ])}
        </article>
      </section>

      <section id="measure">
        <h2 class="section-title reveal">Body measurements</h2>
        <p class="lead reveal">Lengths matter most: they fix where the joints are, which the fit otherwise has to infer. Stand relaxed, use a mirror or the helper to find the landmarks, and record both readings.</p>
        <article class="card reveal guide-block">
          ${measures}
          <p class="guide-note">Also take one front and one side photo on the tape cross, in the outfit.</p>
        </article>
      </section>

      <section id="handover">
        <h2 class="section-title reveal">Afterwards</h2>
        <div class="grid two">
          <article class="card reveal">
            <h3>What to hand over</h3>
            ${list([
              'The original video files, named as in the takes table.',
              'The board’s measured square size.',
              'A short <code>notes.txt</code>: the camera model, app and settings, the take order, where the tripod stood for takes 3–5, any lock that was lost, and the measurements.'
            ])}
            <p>All of it stays on one local machine. Personal video and measurements are never uploaded or committed anywhere.</p>
          </article>
          <article class="card reveal">
            <h3>What is ready, and what is not</h3>
            ${list([
              '<strong>Ready, proven on real footage:</strong> the single-camera pipeline that built the current avatar from a turn in place and a short movement clip, with a guessed lens.',
              '<strong>Built, not yet run on real footage:</strong> the checkerboard calibration and the step that prepares frames with the real lens. On simulated recordings the calibration finds the focal length to within 0.15 %, but it meets its accuracy target across the whole image only when the board reaches all four corners, and its strictest test, at the very corners, failed; that is recorded, not hidden. Take 0 is its first real test.',
              '<strong>Not built yet:</strong> a fit made for the orbit, with the body fixed and the camera moving, and the scoring of this session’s sealed take. Until the orbit fit exists, the orbit is tracked as a turn in place, which has not yet been tried on an orbit.',
              '<strong>No helper?</strong> A tripod with takes 0 and 3–6 still works: the earlier method, now with a real lens model, fitted clothes, the arms-up turn and full measurements.'
            ])}
          </article>
        </div>
        <p class="guide-back no-print"><a class="btn" href="${prefix}avatar/index.html">Back to the avatar</a></p>
      </section>
    </div>`;
}
