# Iteration log

Every round = change → real-GPU render (`tests/visual.mjs`) → metrics. Targets: IoU ≥ 0.85, landmarks ≤ 0.03 × photo width,
ΔE2000 ≤ 10 at every sample, layout ≤ 0.01.

## Model fitting (offline preview, `tools/model/preview.py`, numpy sphere tracer through the shared camera)

| round | change | preview IoU | notes |
|---|---|---|---|
| m1 | first SDF: sphere head + torso + hump + belly | 0.891 | head bloated by large smooth-union radii; belly in front of the arm |
| m2 | smaller blends, belly pushed back | 0.935 | eyes buried under an over-sized socket mound |
| m3 | head pitch, eye depth, rim reduced | 0.930 | head read as a ball on a body, face under the hand too steep |
| m4 | separate jaw volume under the hand | 0.922 | hump became a visible lump on the right |
| m5 | hump → tapered neck capsule, head yaw ≠ body yaw (−27° / −14°) | 0.931 | belly patch now faces the camera like figure A |
| m6 | narrower neck, finger tips moved inside the contour, belly patch re-fit | 0.948 | hand still a fist |
| m7–m9 | smaller/lower palm, separated thumb, finger radii, left eye deeper | 0.948 | hand reads as four fingers + thumb |

Mesh pipeline problems found and fixed by the build's own assertions:
- genus-1 skin: a paper-thin bridge of skin between the left eye socket and the middle-finger groove → sockets and the
  arm groove are now removed as one blended volume (`k_wall`), so no wall thinner than ~k/2 can survive.
- the 0.011 air gap aliased at 0.0045 voxels → gap 0.013, voxel 0.004 (3.25 voxels across the gap).
- two 8-triangle numerical specks inside the solid → largest component kept, count reported.
- cage: arm/body node classification used the un-carved body → 396 false "bridging" tets; fixed (carved body). Now 0.
- cage fingers collapsed by a global coarse remesh pass → zone-exclusive passes, fingers 0.0135, hand 0.021.
- skin→cage binding by tet-centroid search missed large tets → search through the tets around the nearest cage nodes.
- 67 sliver tets (quality < 0.1) → smart-Laplacian improvement, 3 left.

## Render iterations (browser, WebGPU on Apple Metal)

| round | change | IoU | landmark worst | ΔE2000 max / mean | layout |
|---|---|---|---|---|---|
| iter1 | first light: AgX, key from the upper right-front | 0.946 | (mask bug) | 20.9 / 14.5 | 0.0007 |
| iter2 | ACES filmic, lower exposure, darker wall; ID-mask edge sliver ignored | 0.951 | 0.039 | 22.5 / 11.6 | 0.0007 |
| iter3 | matte skin (roughness 0.66, clearcoat 0.07), lighter hand, neutral wall, palm landmark re-fit | 0.951 | 0.032 | 16.9 / 8.3 | 0.0007 |
| iter4 | neutral-warm key, low front-left fill, yellower albedo | 0.951 | 0.032 | 16.0 / 7.0 | 0.0007 |
| iter5 | left eye deeper, thicker fingers, gaze down; key more from the right | 0.951 | 0.030 | 16.8 / 6.9 | 0.0007 |
| iter6 | overhead key (0.30, 0.90, 0.32) | 0.951 | 0.030 | 16.9 / 6.8 | 0.0007 |
| iter7 | key scan (0.45,0.88,0.08) / (0.55,0.82,0.16) | 0.951 | 0.030 | 12.2 / 6.9 | 0.0007 |
| iter8 | warm side bounce from the right, fill lowered | 0.951 | 0.030 | 12.1 / 5.7 | 0.0007 |
| iter9–10 | key scan around the top-right-back | 0.951 | 0.030 | 9.9 / 5.1 | 0.0007 |
| iter11 | key (0.46, 0.88, −0.09), belly albedo | 0.951 | 0.030 | 9.2 / 5.0 | 0.0007 |
| iter12–14 | bigger beads, quieter GTAO, skin speckle | 0.951 | 0.030 | 8.3 / 4.8 | 0.0007 |
| **final** | QA modes without vignette (tile metric) | **0.951** | **0.030** | **8.3 / 4.8** | **0.0007** |

Triptychs kept: `docs/screenshots/triptych-iter2.png`, `-iter4.png`, `-iter11.png`, `-final.png`.

Finding: the prompt assumed the key light comes from the upper **left**-front. The colour samples of figure A say otherwise
(head top and near shoulder brightest, forehead and lower right dim), so the key is above, slightly right and slightly behind.

## Physics iterations (`tests/physics.mjs` in Node, then `tests/e2e.mjs` in the browser)

| round | change | result |
|---|---|---|
| p1 | first parameters (soft compliance 1.6e-4) | "inverted" counted mid-solve: meaningless; contacts pushed nodes from behind |
| p2 | inversion counted on the shown state + end-of-step repair; sliver improvement in the build | 0 inverted in 6/8 zones |
| p3 | contacts: per-pair side flag, push cap, depth window | no false contacts; face drags penetrate 0 |
| p4 | grab speed limit, larger grab radius on the stiff arm, extra arm edge iterations | hand/forearm drags stable |
| p5 | material retune: soft 2e-4 / stiff 4e-7, damping 5.2 | head wobbles at ~4 Hz, 3 visible swings, rest in ~0.8–1.2 s |
| p6 | eyeballs push the grab target back (hand cannot be dragged through an eye) | no fight between grab and eye |
| final | | e2e 62/62 |


## v2 — user feedback (fingers, mouth, stretched-skin rendering, eyes, out-of-frame, balloon feel)

| item | problem | change | check |
|---|---|---|---|
| fingers | 4 fingers + thumb; Nailoong has four digits | 3 fingers + thumb, finger chains smoothed (`k_knuckle` 0.028) | skin genus 0; IoU 0.952 |
| mouth | none | smile line on a slight muzzle, 28° under the eyes (figure B); palm re-fitted (`palm_uv` 0.512) | palm landmark 0.014 |
| eyelids | ball in a dent | lid shell + elliptic-cone opening, upper lid lower | `docs/screenshots/model-face-front-sdf.png` |
| "mosaic" | tet-linear (C0) embedding showed tets when stretched | smooth meshless embedding (16 nodes, Wendland), float F targets | visual |
| "strings" | ridge wrinkle texture + jelly grab | wrinkles removed; membrane thinning shading | visual |
| eye pops out | ball hung on one tet | ball follows the smooth socket field (polar R + 3/4 stretch) | e2e eye checks |
| frame | everything clipped to the photo | full-cover canvas, rest-belongs-to-photo mask, picking with the same rule | new e2e check |
| feel | local springs (jelly) | weak restore, membrane edges, grab radius grows with the pull, group-only grab | physics OK, e2e 63/63 |
| inverted tets in extreme pulls | socket label applied to hand nodes near the eye; repair converged too slowly | socket = body nodes only; repair iterates on the collapsed set | 0 inverted in 5 extreme pulls |

Screenshots: `docs/screenshots/v2-pulls.png` (pulled out of the frame, head top, right cheek, hand); the mouth line shows next to the moved hand in `docs/screenshots/eye-vs-hand.png`.


## v3 — after the user's screen recordings (violent drags, original vs. this)

Frames extracted from the two recordings showed: this version tore into jagged sheets, cut straight along the photo border,
flattened the eyes and let them come out; the original stretches the whole skin into long smooth sheets with folds,
crumples when compressed and ripples back. The mouth was off-centre.

| item | change | check |
|---|---|---|
| physics | tet cage + embedding replaced by a Sheen-style GPU membrane on the character surface (34.5 k vertices; integrate / Jacobi / mouse / normals passes, 6 × 5 per frame) | `docs/screenshots/v3-violent.png`, `v3-recoil.png` |
| eyes | balls ride on the lid ring; sockets held by local rigid shape matching | violent drags: lid clearance ≥ 0.0033 |
| hand | rigid shape matching as a limb; arm grab radius 0.16 | pulling the hand away uncovers the face |
| hand / face | 68 collision spheres in the arm, rest-side push, collision-only sweep per sub-step | penetration ≤ 0.0125 (violent), ≤ 0.006 (face zones) |
| frame | outside the photo: fade in by displacement (no straight cut) | violent screenshots |
| mouth | laid out in the eye-pair face frame, cut after the hand imprint | `docs/screenshots/v3-mouth-hand-away.png` |
| WebGL2 | three flips render-target reads and screen y but not data textures / read-back: index from raw screen y, read-back rows flipped when measured | WebGL2 fallback drag + settle |
| tests | new violent-drag test; falloff shells at a fixed 0.12 scale; droplet check uses a 2-ring fan | e2e 63/63, visual PASS |

## v3.1 — "too soft, like a water balloon, moves too much" (third recording)

Shape recovery × 3, velocity retention 0.98 → 0.965, pull saturation 1.4 → 0.5 (body) and 0.5 → 0.22 (arm), the grab patch no
longer widens with the pull. Pulls now give a local rubber lobe while the head keeps its round volume; the hand moves a short
way as a limb. Flick: 3 decaying swings, at rest in 0.9 s. Tests adjusted: flick needs ≥ 2 swings (was 3, the firmer
material is meant to wobble less), the out-of-frame test drags further (pulls saturate). e2e 62/63: the 60 fps check failed
in a loaded session (the idle page ran at ~33 ms there as well). Screenshot: `docs/screenshots/v3.1-violent.png`.

## v3.2 — eyes look at the pointer

- Before: the gaze turned only 30 % of the way toward the pointer, was capped at 13° from the cover gaze, caught up at 5/s, and aimed at a point on the plane z = 1.4 (far from the face). The eyes barely moved.
- Now (`src/render/eyes.ts`, `src/interaction/grab.ts`): the look point is the pointer ray hit on the plane z = 0.8, just in front of the face. Each ball aims straight at it, clamped to a 45° cone around the socket normal (the cover gaze lies inside it). It turns there by constant-rate slerp at 16/s, and returns to the cover gaze when the pointer leaves the photo.
- New e2e check: at 8 pointer positions over the photo, the projected gaze direction of the front eye is compared with the direction to the pointer. Worst error 2.3°. 63/63 checks passed; median frame interval while dragging 16.67 ms.
- v3.2b: the eyes follow the pointer over the whole page, including the paper around the photo; they return to the cover gaze only on `pointerleave` of the document. e2e targets now include 5 points on the paper. Worst aim error 4.8°.
- Performance finding (not fixed): with the default solver (6 substeps × 5 iterations) the GPU frame cost sits right at the 16.7 ms budget. After about 18 s of continuous simulation, headless Chromium on this machine drops to a 50 ms frame interval and fluctuates between 16.7 and 50 ms afterwards. A blank tab stays at 16.7 ms, and substeps 2 / iterations 2 recovers 16.7 ms at once, so the cause is GPU throughput, not CPU (encode < 1 ms) or a leak. The e2e "60 fps while dragging" check is therefore order- and timing-dependent: 63/63 once, 61/63 and 62/63 in the two runs after this change.
