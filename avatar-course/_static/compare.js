/* Before/after wipe slider. No dependencies.

   <div class="course-compare" data-before="standard track" data-after="hi-fi" data-start="50" style="--compare-max:640px">
   <img src="before.jpg" alt="..."><img src="after.jpg" alt="...">
   </div>

   The two children are two <img> or two <video> elements of the same pixel size. The first is shown left of the
   divider, the second right of it. Videos are muted, looped and kept in step; they play only while on screen, and
   not at all for readers who prefer reduced motion (a Play button is always there).
   Pointer: drag anywhere. Keyboard: focus the figure, then Left/Right (Shift for big steps), Home/End.
   The script may be included more than once on a page. */
(function () {
  "use strict";
  if (window.__courseCompareLoaded) { return; }
  window.__courseCompareLoaded = true;

  function setup(root) {
    if (root.classList.contains("is-ready")) { return; }
    var media = Array.prototype.filter.call(root.children, function (el) { return el.tagName === "IMG" || el.tagName === "VIDEO"; });
    if (media.length !== 2) { return; }
    var before = media[0], after = media[1];
    var isVideo = before.tagName === "VIDEO" && after.tagName === "VIDEO";
    var labelBefore = root.getAttribute("data-before") || "before";
    var labelAfter = root.getAttribute("data-after") || "after";
    before.classList.add("cc-media", "cc-before");
    after.classList.add("cc-media", "cc-after");
    root.insertBefore(after, before);                 // the second media sets the size; the first is clipped on top

    var range = document.createElement("input");
    range.type = "range"; range.min = "0"; range.max = "100"; range.step = "1";
    range.value = root.getAttribute("data-start") || "50";
    range.className = "cc-range";
    range.setAttribute("aria-label", "Divider between “" + labelBefore + "” (left) and “" + labelAfter + "” (right)");
    function apply() {
      root.style.setProperty("--cc-pos", range.value + "%");
      range.setAttribute("aria-valuetext", range.value + "% " + labelBefore + ", " + (100 - range.value) + "% " + labelAfter);
    }
    range.addEventListener("input", apply);
    range.addEventListener("keydown", function (e) {
      if (e.shiftKey && (e.key === "ArrowLeft" || e.key === "ArrowRight")) {
        range.value = String(Math.max(0, Math.min(100, Number(range.value) + (e.key === "ArrowLeft" ? -10 : 10))));
        apply(); e.preventDefault();
      }
    });
    root.appendChild(range);
    var line = document.createElement("div"); line.className = "cc-line"; root.appendChild(line);
    var knob = document.createElement("div"); knob.className = "cc-knob"; root.appendChild(knob);
    [["cc-label cc-label-before", labelBefore], ["cc-label cc-label-after", labelAfter]].forEach(function (p) {
      var s = document.createElement("span"); s.className = p[0]; s.textContent = p[1]; s.setAttribute("aria-hidden", "true"); root.appendChild(s);
    });
    apply();

    if (isVideo) {
      var vids = [before, after];
      vids.forEach(function (v) { v.muted = true; v.loop = true; v.playsInline = true; v.removeAttribute("controls"); v.preload = "auto"; });
      var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      var wanted = !reduce, visible = false;
      var btn = document.createElement("button");
      btn.type = "button"; btn.className = "cc-play";
      function refresh() {
        var on = wanted;
        btn.textContent = on ? "Pause" : "Play";
        btn.setAttribute("aria-label", on ? "Pause both videos" : "Play both videos");
        if (on && visible) { vids.forEach(function (v) { var p = v.play(); if (p && p.catch) { p.catch(function () {}); } }); }
        else { vids.forEach(function (v) { v.pause(); }); }
      }
      btn.addEventListener("click", function () { wanted = !wanted; refresh(); });
      root.appendChild(btn);
      // keep the second video on the first one's clock
      function sync() {
        if (!after.paused && Math.abs(after.currentTime - before.currentTime) > 0.06) { after.currentTime = before.currentTime; }
      }
      before.addEventListener("timeupdate", sync);
      before.addEventListener("seeked", function () { after.currentTime = before.currentTime; });
      if ("IntersectionObserver" in window) {
        new IntersectionObserver(function (entries) { visible = entries[0].isIntersecting; refresh(); }, { threshold: 0.25 }).observe(root);
      } else { visible = true; }
      refresh();
    }
    root.classList.add("is-ready");
  }

  function init() { Array.prototype.forEach.call(document.querySelectorAll(".course-compare"), setup); }
  if (document.readyState === "loading") { document.addEventListener("DOMContentLoaded", init); } else { init(); }
  window.courseCompareInit = init;
})();
