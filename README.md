## Changes

- Original code didn't seem to support room type 7 and 8. Instead, I implemented room type 8 on my own, using room type
  9 parsing code for inspiration. It is understandable that room type 7 is not implemented, as its cornermap is empty,
  and points are not created. Room type 8, however, has entries in its cornermap, as well as is similar to room type 9.
  room type 9 edge map has blue channel, which marks floor, whereas room type 8 has only red channel, which corresponds
  to ceiling. I simply replaced the cornermap values and channels in corner and edge maps.
  - Instead parsing room type 7 as room type 1 and dropping unnecessary lines. Will dive-deeper and implement proper fix
    in the actual, non-prototype version of the app.