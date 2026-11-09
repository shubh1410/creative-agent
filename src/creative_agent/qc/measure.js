// Measurement script run inside the rendered banner page by the QC step (see qc/browser.py and qc/context.py).
// It looks only at the visible elements inside #ad and returns:
//   { W, H, elements }  -  W/H are the banner size in px; `elements` has one entry per visible element with:
//     idx, ancestors        position in the list, and the indexes of its visible ancestors
//     tag, id, cls          basic identity
//     role, roleCtx         data-role of the element itself / of the nearest ancestor that has one
//     el, insideEl          data-el (content element id) of the element itself / of the nearest ancestor
//     asset, variant        data-asset (product image id) and data-variant (logo colour)
//     text, ownText         full visible text of a data-el element / text directly inside this element
//     hasContent, lines     whether it shows anything; line count (headlines only)
//     bbox, textRects       element box and one box per line of its own text, in px
//     coveredBy             other painted elements that sit in front of its text, with the share covered
//     scroll, client        scroll vs. client size, used to detect overflow
//     colours/fonts         colour, background, border, shadow, fill/stroke, font family/size/weight
//     isImage, natural      for <img>: natural pixel size, used to detect upscaling
// The QC rules in qc/checks.py work on this data only, so no rule needs to touch the browser.

() => {
  const ad = document.getElementById('ad');
  const visible = (el) => {
    for (let e = el; e && e !== document.documentElement; e = e.parentElement) {
      const s = getComputedStyle(e);
      if (s.display === 'none' || s.visibility === 'hidden' || parseFloat(s.opacity) === 0) return false;
    }
    const r = el.getBoundingClientRect();
    return r.width > 0 && r.height > 0;
  };
  const lineCount = (el) => {
    const range = document.createRange();
    range.selectNodeContents(el);
    const tops = [...range.getClientRects()].map(r => r.top).sort((a, b) => a - b);
    let lines = 0, last = -1e9;
    for (const t of tops) { if (t - last > 4) { lines++; last = t; } }
    return lines;
  };
  // One rect per line of the element's own text nodes (not its children's).
  const textRects = (el) => {
    const out = [];
    for (const n of el.childNodes) {
      if (n.nodeType !== 3 || !n.textContent.trim()) continue;
      const range = document.createRange();
      range.selectNodeContents(n);
      for (const r of range.getClientRects()) {
        if (r.width > 0 && r.height > 0) out.push({x: r.x, y: r.y, w: r.width, h: r.height});
      }
    }
    return out;
  };
  const paints = (e) => {
    const s = getComputedStyle(e), bg = s.backgroundColor;
    const alpha = bg.startsWith('rgba') ? parseFloat(bg.split(',')[3]) : (bg === 'transparent' ? 0 : 1);
    return e.tagName === 'IMG' || alpha > 0 || s.backgroundImage !== 'none';
  };
  // Share of sample points inside the text rects where another painting element sits in front of the text.
  const coveredBy = (el, rects) => {
    const counts = new Map();
    let total = 0;
    for (const r of rects) {
      for (let i = 0; i < 8; i++) {
        for (let j = 0; j < 2; j++) {
          const px = r.x + r.w * (i + 0.5) / 8, py = r.y + r.h * (j + 0.5) / 2;
          const front = [];
          let found = false;
          for (const e of document.elementsFromPoint(px, py)) {
            if (e === el) { found = true; break; }
            if (el.contains(e)) continue;
            if (ad.contains(e) && e !== ad && paints(e)) front.push(e);
          }
          if (!found) continue;
          total++;
          for (const e of new Set(front)) counts.set(e, (counts.get(e) || 0) + 1);
        }
      }
    }
    return total ? [...counts].map(([e, n]) => ({el: e, share: n / total})) : [];
  };
  const sides = ['Top', 'Right', 'Bottom', 'Left'];
  const nodes = [ad, ...ad.querySelectorAll('*')].filter(visible);
  const index = new Map(nodes.map((n, i) => [n, i]));
  const elements = [];
  for (const el of nodes) {
    const s = getComputedStyle(el), r = el.getBoundingClientRect();
    const ownText = [...el.childNodes].filter(n => n.nodeType === 3).map(n => n.textContent)
      .join(' ').replace(/\s+/g, ' ').trim();
    const elAnc = el.closest('[data-el]'), roleAnc = el.closest('[data-role]');
    const ancestors = [];
    for (let p = el.parentElement; p; p = p.parentElement) { if (index.has(p)) ancestors.push(index.get(p)); }
    const rects = ownText ? textRects(el) : [];
    elements.push({
      idx: index.get(el), ancestors,
      tag: el.tagName.toLowerCase(), id: el.id || null, cls: el.getAttribute('class') || '',
      role: el.dataset.role || null, roleCtx: roleAnc ? roleAnc.dataset.role : null,
      el: el.dataset.el || null, insideEl: elAnc ? elAnc.dataset.el : null,
      asset: el.dataset.asset || null, variant: el.dataset.variant || null,
      text: el.dataset.el ? el.innerText.replace(/\s+/g, ' ').trim() : null, ownText,
      hasContent: !!(el.innerText || '').trim() || !!el.querySelector('img,svg,canvas,video'),
      lines: el.dataset.role === 'headline' ? lineCount(el) : null,
      bbox: {x: r.x, y: r.y, w: r.width, h: r.height},
      textRects: rects,
      coveredBy: rects.length ? coveredBy(el, rects).map(c => ({idx: index.get(c.el), share: c.share}))
        .filter(c => c.idx !== undefined) : [],
      scroll: {w: el.scrollWidth, h: el.scrollHeight}, client: {w: el.clientWidth, h: el.clientHeight},
      color: s.color, bg: s.backgroundColor, bgImage: s.backgroundImage, boxShadow: s.boxShadow,
      borderColors: sides.map(x => s['border' + x + 'Color']),
      borderWidths: sides.map(x => parseFloat(s['border' + x + 'Width'])),
      fill: s.fill, stroke: s.stroke, fontFamily: s.fontFamily,
      fontSize: parseFloat(s.fontSize), fontWeight: parseInt(s.fontWeight, 10),
      isImage: el.tagName === 'IMG',
      natural: el.tagName === 'IMG' ? {w: el.naturalWidth, h: el.naturalHeight} : null,
    });
  }
  return {W: ad.clientWidth, H: ad.clientHeight, elements};
}
