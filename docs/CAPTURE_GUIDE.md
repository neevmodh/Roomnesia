# Recording the two videos

Recording quality matters more than any setting in this tool. Five minutes of care here saves an hour of tuning.

## The room
- **Size:** a small area, about 3 × 3 to 5 × 5 m.
- **Texture helps:** books, posters, a rug and furniture with visible edges all help. Large blank white walls, mirrors and big windows give poor depth.
- **Lighting:** steady and the same for both videos. Avoid filming straight into a bright window.

## Video A (before)
- **Length:** 20–40 s, at 1080p or 4K. Portrait or landscape both work.
- **How to move:** walk a slow arc around the area with the phone at about chest height, tilted slightly down so the floor is in view. **Keep translating sideways.** Turning on the spot gives no depth.
- **Cover the spots that will change** from at least 3–4 different angles.

## Make the changes
- **How many:** 3–5 changes:
  - **move** something big, such as a chair, more than 50 cm, ideally turning it too;
  - **add** an object, such as a backpack or a box;
  - **remove** an object, such as a bin or a plant.
- **Size:** objects of about 20 cm or more work best. Glass or clear plastic is invisible to depth.
- **Leave everything else alone.** Keep people out of the shot.

## Video B (after)
- **Same speed and length.**
- **Take a different path or height** from video A. This tests that viewpoint changes aren't reported as changes. Still overlap at least ~60% of the area.
- **Look at every changed spot again**, including the empty places where things were removed. The system needs to see *through* those spots to confirm they're empty.

## Run
```bash
uv run roomnesia data/my_room/A.mov data/my_room/B.mov --out outputs/my_room --gif --camera-height 1.4
```
