(() => {
  const prefersReduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function startBackground() {
    const canvas = document.getElementById('bg-canvas');
    if (!canvas) return;
    if (prefersReduced) return;

    const ctx = canvas.getContext('2d', { alpha: true });

    let w = 0, h = 0, dpr = 1;
    let particles = [];
    let raf = null;

    function resize() {
      dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
      w = canvas.clientWidth;
      h = canvas.clientHeight;
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      init();
    }

    function rand(a, b) { return a + Math.random() * (b - a); }

    function init() {
      const base = Math.floor((w * h) / 22000); // density heuristic
      const n = Math.max(55, Math.min(150, base));
      particles = [];
      for (let i = 0; i < n; i++) {
        particles.push({
          x: rand(0, w),
          y: rand(0, h),
          vx: rand(-0.35, 0.35),
          vy: rand(-0.35, 0.35),
          r: rand(1.0, 2.2),
          phase: rand(0, Math.PI * 2),
        });
      }
    }

    function step(t) {
      ctx.clearRect(0, 0, w, h);

      // subtle vignette
      const g = ctx.createRadialGradient(w * 0.5, h * 0.5, 0, w * 0.5, h * 0.5, Math.max(w, h) * 0.8);
      g.addColorStop(0, 'rgba(0,0,0,0)');
      g.addColorStop(1, 'rgba(0,0,0,0.45)');
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, w, h);

      const time = (t || 0) * 0.001;

      // update
      for (const p of particles) {
        // a gentle flow field (no external deps)
        const fx = Math.sin((p.y / 180) + time + p.phase) * 0.04;
        const fy = Math.cos((p.x / 220) - time + p.phase) * 0.04;

        p.vx = (p.vx + fx) * 0.994;
        p.vy = (p.vy + fy) * 0.994;

        p.x += p.vx;
        p.y += p.vy;

        if (p.x < -20) p.x = w + 20;
        if (p.x > w + 20) p.x = -20;
        if (p.y < -20) p.y = h + 20;
        if (p.y > h + 20) p.y = -20;
      }

      // draw connections
      for (let i = 0; i < particles.length; i++) {
        const a = particles[i];
        for (let j = i + 1; j < particles.length; j++) {
          const b = particles[j];
          const dx = a.x - b.x;
          const dy = a.y - b.y;
          const dist = Math.hypot(dx, dy);
          const max = 140;
          if (dist < max) {
            const alpha = (1 - dist / max) * 0.18;
            ctx.strokeStyle = `rgba(255,255,255,${alpha})`;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
          }
        }
      }

      // draw particles
      for (const p of particles) {
        ctx.fillStyle = 'rgba(255,255,255,0.22)';
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fill();
      }

      raf = requestAnimationFrame(step);
    }

    function onVis() {
      if (document.hidden) {
        if (raf) cancelAnimationFrame(raf);
        raf = null;
      } else {
        if (!raf) raf = requestAnimationFrame(step);
      }
    }

    window.addEventListener('resize', resize, { passive: true });
    document.addEventListener('visibilitychange', onVis);

    resize();
    raf = requestAnimationFrame(step);
  }

  window.SiteBackground = { startBackground };
})();
