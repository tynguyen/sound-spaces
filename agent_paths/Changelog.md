# Major Changes Here

# Aug 23rd, 2021

## Added

- [x] Combine audio signals, pair by pair by adding

```
python scripts/add_two_audios.py
```

- [x] Find correlations between a sound source and other target sounds

```
python scripts/get_correlation_two_audios.py
```

- [x] Normalize recorded audio by 10. before saving to avoid clamping. Details: `soundpaces/simulator.py`
- [x] Save the transformation from agent's camera frame to the world in the metrics space

## Changed

- [x] Fixed mesh display bug
