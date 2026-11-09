# Assumptions

Decided up front, with no clarification email.

| # | Assumption | Why | Lives in |
|---|---|---|---|
| A1 | Maxis video is the master; Image 2 is a layout reference; Income storyboard is a negative test | Only Maxis matches the kit | `inputs/*/job.json` |
| A2 | `brand_kit.json` converted from the PDF; where the PDF contradicts itself, the 2025 digital section wins | The assignment expects JSON; rules must be exact | `config/brands/maxis/` |
| A3 | Logos and products cropped from the provided files | No source assets | `config/brands/maxis/logos/`, ingest crops |
| A4 | Figtree (open licence) substitutes for the Maxis font, same weights per role | The Maxis font is proprietary | `brand_kit.json` → `fonts.substitute` |
| A5 | We write the brief | None provided | `inputs/maxis-merdeka/brief.md` |
| A6 | Click URL `https://www.maxis.com.my/` | None provided | `job.json` → `click_url` |
| A7 | "Get yours with Maxis today" becomes the CTA, styled as a pill | No CTA in the master; no new copy | `job.json` → `cta` |
| A8 | Self-contained HTML5 per size + zip + backup JPG; 150 KB on uncompressed `index.html`; `clickTag` | Requirements imply HTML5; stricter measure | `render/finalize.py`, `finish` node |
| A9 | Corner logo stacked 31×40 px, 10 px right, 22 px top (p. 20), also for 160×600; horizontal 18 px tall at 320×50 | PDF p. 9 allows horizontal when stacked is illegible | `brand_kit.json` → `logo.corner`, `logo.overrides` |
| A10 | End frame: centred logo 12.5% (horizontal) or 20% (vertical) of width (p. 18), capped at 60% of height; strip sizes use the horizontal logo; corner logo hidden on the end frame | No banner end-frame rule in the PDF | `brand_kit.json` → `logo.end_frame` |
| A11 | Max 3 font sizes per frame; green CTAs allowed; white logo allowed on Maxis Green | Stricter rule; master and reference both do it | `brand_kit.json` |
| A12 | Minimum 10 px, legal 8 px | PDF doesn't say | `brand_kit.json` → `typography` |
| A13 | Animation plays once, ≤ 30 s, crossfades, ends on the end frame; small sizes use fewer frames | Common display limit; master is 20 s | `brand_kit.json` → `motion`; `layout_tips.md` |
| A14 | Background recreated in CSS from palette colours; fireworks dropped | Fireworks are off-palette and cost file size | Builder prompt |
| A15 | Repo private and shared with Adzymic; inputs committed | PDF is Maxis intellectual property | README |
