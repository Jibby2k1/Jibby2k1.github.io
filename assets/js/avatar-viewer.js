// Avatar playground: plays pre-rendered frames of the Gaussian-splat avatar.
// Each move is one sprite sheet: rows are turn angles, columns are animation
// frames. Every frame was rendered offline by the model; this script only
// picks which one to draw. No frameworks, no network calls beyond the sheets.
(function () {
  var root = document.querySelector('[data-avatar-viewer]');
  if (!root) return;
  var base = root.getAttribute('data-base');
  var canvas = root.querySelector('canvas');
  var ctx = canvas.getContext('2d');
  var turn = root.querySelector('[data-avatar-turn]');
  var playBtn = root.querySelector('[data-avatar-play]');
  var status = root.querySelector('[data-avatar-status]');
  var moveBtns = Array.prototype.slice.call(root.querySelectorAll('[data-move]'));
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var idx = null, sheets = {}, move = 'stand', frame = 0, row = 0, playing = false, last = 0, timer = null;
  var pending = null, pendingAt = 0, pendingCap = 0;
  var FPS = 9;            // default rate; a move may set its own in the index (moves[name].fps)
  // A new move starts when the current loop wraps; the cap (one loop plus one frame) only guards a stalled clock.

  function fps(name) { return (idx && idx.moves[name] && idx.moves[name].fps) || FPS; }

  function label(name) {
    var b = moveBtns.filter(function (x) { return x.getAttribute('data-move') === name; })[0];
    return b ? b.textContent.trim() : name;
  }

  function sheet(name) {
    if (sheets[name]) return sheets[name];
    var img = new Image();
    img.decoding = 'async';
    img.onload = draw;
    img.src = base + idx.moves[name].file;
    sheets[name] = img;
    return img;
  }

  function draw() {
    if (!idx) return;
    var img = sheet(move);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!img.complete || !img.naturalWidth) return;
    var n = idx.moves[move].frames;
    var f = n > 1 ? frame % n : 0;
    ctx.drawImage(img, f * idx.frame_w, row * idx.frame_h, idx.frame_w, idx.frame_h, 0, 0, idx.frame_w, idx.frame_h);
  }

  function say() {
    var deg = idx.yaws_deg[row];
    status.textContent = label(move) + ', turned ' + (deg === 0 ? 'to face you' : Math.abs(deg) + '° to the ' + (deg < 0 ? 'left' : 'right')) + (playing ? ', playing' : '');
  }

  function tick(t) {
    if (!playing) return;
    var period = 1000 / fps(move);
    if (t - last >= period) {
      // advance the clock by whole periods, not to t, so the mean rate is the move's own
      // fps rather than snapping to the display refresh (12 fps would play at 10 on 60 Hz)
      last = t - last > 2 * period ? t : last + period;
      frame += 1;
      var n = idx.moves[move].frames;
      if (pending && (frame % n === 0 || t - pendingAt >= pendingCap)) { var next = pending; pending = null; apply(next); return; }
      draw();
    }
    timer = requestAnimationFrame(tick);
  }

  function setPlaying(on) {
    if (!idx) return;
    if (!on && pending) { move = pending; pending = null; frame = 0; draw(); }   // pausing applies a queued move
    playing = on && idx.moves[move].frames > 1;
    playBtn.textContent = playing ? 'Pause' : 'Play';
    playBtn.setAttribute('aria-pressed', playing ? 'true' : 'false');
    if (timer) cancelAnimationFrame(timer);
    if (playing) timer = requestAnimationFrame(tick);
    say();
  }

  function setMove(name) {
    if (!idx) return;
    if (playing && name === move) { pending = null; say(); return; }   // re-clicking the playing move keeps it going
    moveBtns.forEach(function (b) {
      var on = b.getAttribute('data-move') === name;
      b.classList.toggle('active', on);
      b.setAttribute('aria-pressed', on ? 'true' : 'false');
    });
    if (playing && name !== move && name !== 'stand' && idx.moves[move].frames > 1) {
      pending = name; pendingAt = performance.now();   // finish the current loop first
      pendingCap = 1000 * (idx.moves[move].frames + 1) / fps(move);
      return;
    }
    pending = null;
    apply(name);
  }

  function apply(name) {
    move = name;
    frame = 0;
    moveBtns.forEach(function (b) {
      var on = b.getAttribute('data-move') === name;
      b.classList.toggle('active', on);
      b.setAttribute('aria-pressed', on ? 'true' : 'false');
    });
    draw();
    setPlaying(name !== 'stand' && !reduce ? true : playing);
  }

  function setRow(r) {
    if (!idx) return;
    row = Math.max(0, Math.min(idx.yaws_deg.length - 1, r));
    turn.value = String(row);
    draw();
    say();
  }

  // drag on the avatar to turn it
  var dragX = null, dragRow = 0;
  canvas.addEventListener('pointerdown', function (e) { dragX = e.clientX; dragRow = row; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener('pointermove', function (e) {
    if (dragX === null || !idx) return;
    var step = canvas.getBoundingClientRect().width / (idx.yaws_deg.length * 1.5);
    setRow(dragRow + Math.round((e.clientX - dragX) / step));
  });
  ['pointerup', 'pointercancel'].forEach(function (ev) { canvas.addEventListener(ev, function () { dragX = null; }); });
  turn.addEventListener('input', function () { setRow(parseInt(turn.value, 10)); });
  playBtn.addEventListener('click', function () { setPlaying(!playing); });
  moveBtns.forEach(function (b) { b.addEventListener('click', function () { setMove(b.getAttribute('data-move')); }); });

  fetch(base + 'avatar-sprites.json').then(function (r) { return r.json(); }).then(function (data) {
    idx = data;
    canvas.width = idx.frame_w;
    canvas.height = idx.frame_h;
    turn.max = String(idx.yaws_deg.length - 1);
    row = idx.yaws_deg.indexOf(0);
    turn.value = String(row);
    root.classList.add('ready');
    sheet('stand');
    setMove('wave');
  }).catch(function () {
    status.textContent = 'The avatar could not be loaded. The video below shows it in motion.';
  });
})();
