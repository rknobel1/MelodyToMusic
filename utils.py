import pathlib
import glob
import tensorflow as tf
import pretty_midi
import numpy as np


def get_data(num_files=1282, bars=4, steps_per_beat=4, beats_per_bar=4, vel_bins=16, max_len=100, note_overlap=.1):
    data_dir = pathlib.Path('data/maestro-v2.0.0')
    if not data_dir.exists():
        tf.keras.utils.get_file(
            'maestro-v2.0.0-midi.zip',
            origin='https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip',
            extract=True,
            cache_dir='.', cache_subdir='data',
        )

    filenames = glob.glob(str(data_dir / '**/*.mid*'), recursive=True)
    print('Number of files:', len(filenames))

    all_segments = []
    for j, file in enumerate(filenames[:num_files], start=1):
        segs = midi_to_roll_segments(
            file,
            bars=bars,
            steps_per_beat=steps_per_beat,
            beats_per_bar=beats_per_bar,
            vel_bins=vel_bins
        )
        all_segments.extend(segs)

    if len(all_segments) == 0:
        return np.empty((0, bars * beats_per_bar * steps_per_beat, 128, 4), dtype=np.float32)

    return np.stack(all_segments, axis=0)  # [N, T, 128, 4]


# Build a pedal curve over quantized grid
def pedal_to_curve(pm, grid, inst_idx=0, threshold=64):
    pedal_events = extract_pedal(pm, inst_idx=inst_idx)
    T = len(grid)
    curve = np.zeros((T,), dtype=np.float32)
    if pedal_events.size == 0:
        return curve

    pedal_events = pedal_events[pedal_events[:, 0].argsort()]
    state = 0.0
    ei = 0

    for t in range(T):
        time = float(grid[t])
        while ei < len(pedal_events) and float(pedal_events[ei, 0]) <= time:
            state = 1.0 if float(pedal_events[ei, 1]) >= threshold else 0.0
            ei += 1
        curve[t] = state
    return curve


# Notes to roll
def notes_to_roll(notes, T, pitch_min=0, pitch_max=127, vel_bins=16):
    P = pitch_max - pitch_min + 1
    onset = np.zeros((T, P), dtype=np.float32)
    hold = np.zeros((T, P), dtype=np.float32)
    vel = np.zeros((T, P), dtype=np.float32)

    for pitch, s, d, v in notes:
        if s >= T:
            continue
        e = min(T, s + d)
        p = pitch - pitch_min
        onset[s, p] = 1.0
        hold[s:e, p] = 1.0
        vel_val = min(vel_bins-1, int(v * vel_bins / 128)) / (vel_bins-1)
        vel[s, p] = max(vel[s, p], vel_val)  # store on onset
    return np.stack([onset, hold, vel], axis=-1)  # [T, 128, 3]


# Convert midi to roll segments
def midi_to_roll_segments(midi_file, bars=4, steps_per_beat=4, beats_per_bar=4, inst_idx=0, vel_bins=16, overlap_pct=.5):
    pm = pretty_midi.PrettyMIDI(midi_file)
    grid = build_time_grid(pm, steps_per_beat=steps_per_beat)
    T_total = len(grid)

    notes = midi_to_notes_quantized(midi_file, steps_per_beat=steps_per_beat)
    if notes is None or len(notes) == 0:
        return []

    # Segment length in steps
    seg_len = bars * beats_per_bar * steps_per_beat
    if T_total < seg_len:
        return []

    # Make pedal curve on the same grid
    pedal_curve = pedal_to_curve(pm, grid, inst_idx=inst_idx)  # [T_total]

    segments = []

    # Slide in fixed time steps with small overlap
    hop = max(1, int(seg_len * (1 - overlap_pct)))

    for start in range(0, T_total - seg_len + 1, hop):
        end = start + seg_len

        # Filter notes whose start is in [start, end)
        mask = (notes[:, 1] >= start) & (notes[:, 1] < end)
        seg_notes = notes[mask].copy()
        if len(seg_notes) == 0:
            continue

        # Shift segment to local coordinates
        seg_notes[:, 1] -= start

        # Build roll [seg_len, 128, 3]
        roll = notes_to_roll(seg_notes, T=seg_len, vel_bins=vel_bins)  # onset/hold/vel

        # Add pedal as extra channel broadcast across pitches: [seg_len, 128, 1]
        pedal_seg = pedal_curve[start:end].reshape(seg_len, 1, 1)
        pedal_seg = np.repeat(pedal_seg, 128, axis=1)

        roll = np.concatenate([roll, pedal_seg], axis=-1)  # [T, 128, 4]
        segments.append(roll.astype(np.float32))

    return segments


# Extract pedal information from note sequence
def extract_pedal(pm, inst_idx=0):
    inst = pm.instruments[inst_idx]

    # CC64 sustain pedal messages
    pedal = [cc for cc in inst.control_changes if cc.number == 64]

    # pedal events: (time, value)
    return np.array([[cc.time, cc.value] for cc in pedal], dtype=np.float32)


# Quantize note durations
def build_time_grid(pm, steps_per_beat=4):
    beats = pm.get_beats()  # in seconds

    # Make sub-beat grid between beats
    grid = []
    for b0, b1 in zip(beats[:-1], beats[1:]):
        for k in range(steps_per_beat):
            grid.append(b0 + (b1-b0) * (k/steps_per_beat))
    grid.append(beats[-1])
    return np.array(grid, dtype=np.float32)


def time_to_step(t, grid):
    return int(np.argmin(np.abs(grid - t)))


def midi_to_notes_quantized(midi_file, steps_per_beat=4):
    pm = pretty_midi.PrettyMIDI(midi_file)
    inst = pm.instruments[0]

    grid = build_time_grid(pm, steps_per_beat=steps_per_beat)

    notes = []
    for n in inst.notes:
        s = time_to_step(n.start, grid)
        e = time_to_step(n.end, grid)
        d = max(1, e - s)
        v = n.velocity
        notes.append([n.pitch, s, d, v])

    # sort by step then pitch (stable ordering for chords)
    notes.sort(key=lambda x: (x[1], x[0]))
    return np.array(notes, dtype=np.int32)


# Piano roll to midi
def roll_to_midi(roll, out_path, steps_per_beat=4, tempo=120.0,
                 onset_th=0.2, hold_th=0.2, pedal_th=0.2,
                 min_steps=2, vel_floor=20):
    """
    roll: [T, 128, C] in [0,1]
    Channels: 0 onset, 1 hold, 2 vel, 3 pedal
    """
    T, P, C = roll.shape
    assert P == 128

    step_sec = (120.0 / tempo) / steps_per_beat

    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    inst = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program("Acoustic Grand Piano")
    )

    onset = roll[:, :, 0]
    hold = roll[:, :, 1]
    vel = roll[:, :, 2] if C >= 3 else np.ones((T, P), dtype=np.float32)
    pedal = roll[:, :, 3] if C >= 4 else None

    # Scan pitches, create notes
    for p in range(P):
        t = 0
        while t < T:
            if onset[t, p] > onset_th:
                start_t = t

                # velocity: scale to 1..127 with a floor
                v = int(np.clip(vel[t, p], 0.0, 1.0) * 127)
                v = max(vel_floor, v)

                # extend while hold is on
                t2 = t + 1
                while t2 < T and hold[t2, p] > hold_th:
                    t2 += 1

                # enforce minimum duration so notes are audible
                t2 = max(t2, start_t + min_steps)
                t2 = min(t2, T)

                start = start_t * step_sec
                end = t2 * step_sec

                inst.notes.append(pretty_midi.Note(
                    velocity=v,
                    pitch=p,
                    start=start,
                    end=end
                ))

                # move time forward
                t = t2
            else:
                t += 1

    # Pedal CC64
    if pedal is not None:
        prev = 0
        for t in range(T):
            cur = 1 if pedal[t, 0] > pedal_th else 0
            if cur != prev:
                inst.control_changes.append(pretty_midi.ControlChange(
                    number=64, value=(127 if cur else 0), time=t * step_sec
                ))
                prev = cur

    pm.instruments.append(inst)
    pm.write(out_path)
