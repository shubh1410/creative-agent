# Problems found in the inputs

How each one is noticed and handled. The "Logged as" column is the issue code in `result.json` (or the document where the decision lives).

| # | Problem | Handling | Logged as |
|---|---|---|---|
| 1 | Income storyboard doesn't match the Maxis kit | Job stops at `check_inputs`; no planner or builder call is made | `BRAND_MISMATCH` |
| 2 | Brand kit is a PDF; key rules exist only in slide images | Converted offline, reviewed, page per rule | `brand_kit_notes.md` |
| 3 | No logo files | Cropped from the video end frame and PDF page 9, recoloured to exact brand colours | `LOGO_FROM_CROP` |
| 4 | Maxis font is proprietary | Open-licensed substitute (Figtree) declared in the kit | `BRAND_FONT_SUBSTITUTED` |
| 5 | No brief or landing URL | Brief written from the ads; public Maxis URL | Assumptions A5, A6 |
| 6 | Master has no CTA | The closing line becomes the CTA; no new copy | `CTA_ADDED` |
| 7 | Text baked into frames | Products cropped from the text-free lower band; background recreated in CSS | Design |
| 8 | Stacked logo can't fit 320×50 | Horizontal logo, as PDF p. 9 allows | `LOGO_VARIANT_SWITCHED` |
| 9 | 160×600 and 320×50 aren't in the PDF's Google Display list | Display spec extended | `LOGO_SPEC_EXTRAPOLATED` |
| 10 | Master breaks the PDF's rules (corner logo with wordmark in a video, p. 19; end-frame logo ~27% of width vs 20%, p. 18) | Banners follow the kit, not the video | `MASTER_RULE_DEVIATION` |
| 11 | Colours sampled from video are off (#49E10C vs #40C706) | Snapped to the palette; never used as brand colours | `COLOR_SNAPPED` |
| 12 | Kit contradicts itself (CTA colour p. 25 vs pp. 38–43; 4 vs 3 font sizes pp. 23–24; white logo on green) | Resolved in `brand_kit_notes.md` | Assumptions A2, A11 |
| 13 | Static reference is a different offer | Used for layout only; its copy is never used | Design |
| 14 | 9:16 source, horizontal targets | New composition from the reference layout and the 60/40 grid (p. 39) | Design |
| 15 | No minimum legal text size in the PDF | 10 px, legal 8 px | Assumption A12 |

Input errors that stop a job (nothing is retried): `BRAND_MISMATCH`, `NO_CTA`, `CTA_NOT_FOUND`, `COPY_OVERRIDE_UNKNOWN_ID`, `LOGO_SPEC_IMPOSSIBLE`, `CLICK_URL_UNREACHABLE`, `INGEST_FAILED`. A click URL that blocks bots (HTTP 401/403) is a warning, not a failure.
