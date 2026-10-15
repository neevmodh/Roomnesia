# Real-world benchmark: SceneDiff

[SceneDiff](https://huggingface.co/datasets/yuqun/SceneDiff) (Wu et al., 2025, [arXiv:2512.16908](https://arxiv.org/abs/2512.16908), MIT licence) is a set of real phone-video pairs of the same scene, before and after objects were added, removed or moved, with a mask for every changed object in every frame.

**Protocol.**
- **Pairs:** indoor, room-scale pairs only (bedrooms, living rooms, offices, lounges, …). Kitchen head-camera clips and outdoor scenes are out of scope.
- **Validation split:** 13 pairs, used to develop and fix the system.
- **Test split:** every third indoor pair of the official test split, taken alphabetically (`data/scenediff/test_selection.txt`). Run once, after the code was frozen.
- **Run settings:** default; `--no-vlm`, because the noun phrases don't affect the scores.

**Scoring** (`scripts/eval_scenediff.py`):
- Each detected change is projected into the annotated frames (only where it is the visible surface).
- An annotated object counts as found when a detection of a compatible type overlaps its mask with IoU ≥ 0.2 in some keyframe.
- Moves count as found if either end is matched; "type correct" needs the detection itself to be a move.
- A detection that overlaps no annotated object is a false report.
- This is our own object-level protocol. The SceneDiff paper reports mask AP, which needs per-pixel confidence scores, so the numbers are not directly comparable.

### test split: 21 pairs, 56 annotated changes

**Recall 55%** (31/56) · **precision 65%** (20 false of 57 reports) · change type correct for 14 · by type: removed 7/17, added 6/15, moved 18/24

| object size (largest view) | recall |
|---|---|
| small (< 0.5% of the image: letters, pens, cups) | **0/3** (0%) |
| medium (0.5–3%: shoes, books, small boxes) | **5/24** (21%) |
| large (> 3%: chairs, bags, bins, boxes, stools) | **26/29** (90%) |

| pair | found | false reports | missed objects |
|---|---|---|---|
| bed_1_bed_2 | 2/4 | 1 | paint brush_1 (added), paperclip_1 (added) |
| bedroom_14_bedroom_15 | 1/2 | 0 | circuit_board_1 (removed) |
| bedroom_20_bedroom_21 | 0/2 | 1 | ipad_1 (moved), remote_1 (added) |
| bedroom_26_bedroom_27 | 1/3 | 1 | handkerchief_1 (added), phone_1 (added) |
| bedroom_34_bedroom_35 | 3/3 | 0 | – |
| bedroom_8_bedroom_9 | 1/2 | 0 | Screwdriver set_1 (removed) |
| game_room_1_game_room_2 | 0/6 | 1 | red_ball_1 (removed), black_8 (removed), white_ball_1 (removed), green_ball_1 (removed), blue_ball_1 (added), orange_ball_1 (added) |
| laundry_room_1_laundry_room_2 | 1/1 | 2 | – |
| living_room_11_living_room_12 | 2/3 | 0 | right_shoe (moved) |
| living_room_21_living_room_22 | 2/6 | 0 | snack_bag_1 (removed), bowl_1 (removed), tissue_box_1 (moved), cup_1 (added) |
| living_room_33_living_room_34 | 2/3 | 3 | Remote_1 (removed) |
| living_room_3_living_room_4 | 1/3 | 0 | cookies_1 (moved), remote_control_1 (moved) |
| living_room_49_living_room_50 | 0/2 | 4 | Cup_1 (added), Toy Dog_1 (moved) |
| livingroom_1_livingroom_2 | 2/2 | 0 | – |
| lounge_15_lounge_16 | 2/3 | 2 | gray_chair_1 (removed) |
| mailroom_3_mailroom_4 | 1/1 | 1 | – |
| office_1_office_2 | 2/2 | 3 | – |
| office_5_office_6 | 3/3 | 0 | – |
| room_entrance_1_room_entrance_2 | 2/2 | 0 | – |
| study_1_study_2 | 2/2 | 1 | – |
| workspace_1_workspace_2 | 1/1 | 0 | – |

### val split: 13 pairs, 41 annotated changes

**Recall 68%** (28/41) · **precision 58%** (18 false of 43 reports) · change type correct for 15 · by type: moved 15/19, removed 8/14, added 5/8

| object size (largest view) | recall |
|---|---|
| small (< 0.5% of the image: letters, pens, cups) | **0/4** (0%) |
| medium (0.5–3%: shoes, books, small boxes) | **12/18** (67%) |
| large (> 3%: chairs, bags, bins, boxes, stools) | **16/19** (84%) |

| pair | found | false reports | missed objects |
|---|---|---|---|
| bedroom_30_bedroom_31 | 1/1 | 5 | – |
| living_room_17_living_room_18 | 3/7 | 1 | stool_1 (removed), bread_1 (added), tissue_box_1 (added), controller_1 (moved) |
| living_room_35_living_room_36 | 3/4 | 0 | wooden_fruit_1 (moved) |
| living_room_47_living_room_48 | 1/2 | 0 | game controller_1 (removed) |
| mailroom_1_mailroom_2 | 3/3 | 4 | – |
| room_entrance_3_room_entrance_4 | 2/2 | 1 | – |
| room_entrance_9_room_entrance_10 | 4/4 | 0 | – |
| storage_1_storage_2 | 2/2 | 1 | – |
| laundry_room_3_laundry_room_4 | 2/3 | 1 | Phone box_1 (removed) |
| lounge_3_lounge_4 | 1/3 | 1 | brush_1 (removed), basket_1 (removed) |
| lounge_7_lounge_8 | 1/1 | 2 | – |
| bed_3_bed_4 | 0/2 | 2 | Marker_1 (moved), PaintBrush_1 (added) |
| living_room_15_living_room_16 | 5/7 | 0 | placemat_1 (removed), controller_1 (moved) |

**Reading the numbers.** Roomnesia reasons in 3D with ~2 cm voxels and requires several cameras to see through a spot before calling it a change. Furniture-scale objects (the obstacles a navigating robot cares about) are found reliably. Objects that cover under ~0.5% of the image (a letter, a marker) are below that resolution and are missed. Some "false reports" are real but unannotated changes (pillows and bedding moved, chairs nudged); they are counted as errors anyway.
