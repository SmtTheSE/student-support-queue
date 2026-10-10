# Presentation reference

Reviewed on October 10, 2026:

- [Anthropic homepage](https://www.anthropic.com/)
- [Claude product overview](https://claude.com/product/overview)

The Claude product page supplies the more restrained presentation used here. The design inspection covered rendered layouts and computed styles, not just the page text.

| Reference detail | Project treatment |
| --- | --- |
| Warm off-white page, dark near-black text | `#faf9f5` page and `#141413` text |
| Centered serif product headline, roughly 68px at desktop, weight 500, 1.1 line height | Source Serif 4 headline, 34–68px responsive scale, weight 500 |
| Sans-serif descriptions and navigation | Locally hosted Inter; 13px navigation and 15–20px introduction |
| Compact dark buttons, 8px corners, roughly 8px by 16px padding | Text-only buttons with 8px corners and 11px by 17px padding |
| Quiet borders and warm secondary controls | Thin neutral borders, beige secondary button, subtle hover changes |
| Clear reading hierarchy | Serif section titles and figures, sans-serif body text and chart labels |

Decorative arrow glyphs have been removed from all buttons, resource links, and the footer. The scoring results, evaluation limitations, downloads, and keyboard-operated capacity explorer are retained.

The original brand fonts, logos, illustrations, and site copy are not included. Inter and Source Serif 4 are open-licensed alternatives; their licenses are in `assets/fonts/`.

## Plain-language reading order

The page now follows five questions: What is the problem? How would a teacher use the tool? What did the saved-record test show? What must a school check next? Where are the project files?

The main comparison shows only this tool and the simpler school-record approach, using the same number of places. Labels and a complete sentence explain the counts without assuming statistics knowledge. Extra percentages, the test design, charts, and the command-line example are available in native expandable sections. Uncertainty, unverified source data, and the absence of measured grade improvements remain visible or plainly explained.

The presentation uses one stylesheet, a three-step workflow, consistent reading widths, and short paragraphs. Notebook and model artifacts retain the full technical analysis.

## One continuous reading column (October 10, 2026 revision)

Feedback: the page was split into too many small boxes, rules, and 11–13px notes. The page now has four sections in one 720px column: the problem and how it works, the results, what a school should check next, and the files. Spacing separates ideas instead of borders and cards. Body text is 18px (17px on phones) and nothing is below 14px.

The only boxed element is the results explorer. It replaces the three number tiles, context box, outside-list box, and test-note strip with a slider, two proportional bars (students who later did not pass, out of the same places), and one sentence. The caveats sit below it as plain paragraphs under "How to read this" and "What this does not show". The three optional detail panels remain, grouped under "Want more detail?".

## Student mascot

The hero pairs its plain-language introduction with an original Blender character: a student holding a notebook and waving. Chestnut hair, an oatmeal cardigan, a sage skirt, and a terracotta notebook follow the existing warm palette. The pale oval background is CSS so the transparent render sits naturally on the page.

The mascot appears beside the text on wider screens and below the main links on phones. It is a static WebP image with fixed intrinsic dimensions, descriptive alt text, and no extra JavaScript. The editable scene and generation script are in `art/mascot/`.
