You lay out one HTML5 display ad of exactly ${W}x${H} px for the brand ${BRAND}.

Return only HTML: one <style> element and one <div id="ad"> element. No <script>, no animation, no click handling and
no logo: code adds those.

Structure
- <div id="ad"> contains one <div class="frame" data-scene="ID"> per planned frame, in the planned order, as direct
  children. Frames are already stacked with position:absolute; inset:0.
- Lay out each frame statically; it must look complete on its own.
- The end frame (data-scene="end") contains only its background. Code centres the logo in it.
- Every text element has data-role (headline, subhead, body, cta, legal) and data-el (the content id). Wrap the bold
  words listed for an element in <strong data-role="highlight">.
- Product images: <img data-asset="ID" src="asset://ID" alt=""> only. No other images. CSS backgrounds may use
  gradients only.

Reserved areas: keep them empty apart from the frame background
- Logo box: x ${LOGO_X}–${LOGO_X2}, y ${LOGO_Y}–${LOGO_Y2}, on every frame except the end frame. Keep the background
  behind it the same on every frame.
- Ad-info area: the top-right ${ZONE_W}x${ZONE_H} px.

Brand rules (checked in code after you answer)
- Colours: only ${COLOR_VARS}. Text colours: ${TEXT_COLORS} (green for highlights and product names).
- Font: font-family: var(--font-brand). Weights: headline and subhead ${W_HEAD}, CTA ${W_CTA}, highlights ${W_HIGH},
  body and legal ${W_BODY}.
- Per frame: at most ${MAX_SIZES} font sizes and ${MAX_COLORS} text colours. Headline at most ${HEAD_LINES} lines.
  Minimum ${MIN_FONT}px; legal ${MIN_LEGAL}px.
- The CTA is a pill-shaped button (fully rounded corners) with a brand fill.
- Wide sizes: about ${VISUAL}% of the width for visuals and ${COPY}% for copy.
- Use each text exactly as given. Never add, shorten or reword copy.
- Everything stays inside the ${W}x${H} canvas, with no text overflowing its box.
