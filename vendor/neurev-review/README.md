# NeuRev Review (vendored)

`app.html.vendored` is a verbatim copy of `neurobench/tools/event_labeler/labeler.html` from
[NeuRev-Workbench](https://github.com/Jibby2k1/NeuRev-Workbench/tree/main/neurobench/tools/event_labeler).
`npm run build` publishes it unchanged at `neurev/review/app.html`; the landing page around it is
`scripts/neurev-review.mjs`.

To update: copy the new `labeler.html` over `app.html.vendored`, then `npm run build` and `npm run check`.

The app reads review packs on the reviewer's own computer and keeps answers in their browser. No recording,
pack, or result is ever stored in this repository or served by this site.
