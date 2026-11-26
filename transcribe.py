"""Transcribe an audio file into note names, MIDI, an ASCII guitar tab and a Guitar Pro 5 file.

basic-pitch detects every note in the recording (written to the MIDI file). Drone strings
are set aside, then the loudest, highest line is taken as the melody, quantised to an
eighth-note grid in the given time signature that follows the detected beats, moved into
one register, and laid out on a guitar in standard tuning.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import guitarpro
import librosa
import numpy as np
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Guitar Pro numbers strings from the high e (1) down to the low E (6).
STANDARD_TUNING = [64, 59, 55, 50, 45, 40]
STRING_LABELS = ["e", "B", "G", "D", "A", "E"]
MAX_FRET = 24
# Frets the fretting hand reaches without shifting position (index to little finger).
HAND_SPAN = 3
BARS_PER_TAB_LINE = 6
# Drone strings (dulcimer, banjo, bagpipes) sound on far more onsets than any tune note:
# over a third of them, against about a fifth for the most frequent note of a tune.
DRONE_MIN_SHARE = 0.3
GP_NOTE_VALUES = [1, 2, 4, 8, 16, 32]


@dataclass(frozen=True)
class Meter:
    """A time signature and the grid of steps that notes are quantised to.

    Compound meters (6/8, 9/8, 12/8) count a dotted quarter as the beat and split it into
    three eighths; simple meters (2/4, 3/4, 4/4) split their beat into two.
    """
    numerator: int
    denominator: int

    @classmethod
    def parse(cls, text: str) -> "Meter":
        numerator, denominator = (int(part) for part in text.split("/"))
        if numerator < 1 or denominator not in (2, 4, 8, 16):
            raise ValueError(f"unsupported time signature: {text}")
        return cls(numerator, denominator)

    @property
    def is_compound(self) -> bool:
        return self.denominator >= 8 and self.numerator % 3 == 0 and self.numerator > 3

    @property
    def steps_per_beat(self) -> int:
        return 3 if self.is_compound else 2

    @property
    def steps_per_bar(self) -> int:
        return self.numerator if self.is_compound else self.numerator * 2

    @property
    def beats_per_bar(self) -> int:
        return self.steps_per_bar // self.steps_per_beat

    @property
    def step_note_value(self) -> int:
        return self.denominator if self.is_compound else self.denominator * 2


@dataclass
class Note:
    step: int
    length: int
    pitch: int
    string: int = 0
    fret: int = 0


def note_name(pitch: int) -> str:
    return f"{NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


def is_playable(pitch: int) -> bool:
    return STANDARD_TUNING[-1] <= pitch <= STANDARD_TUNING[0] + MAX_FRET


def beat_grid(audio_path: Path) -> tuple[np.ndarray, float]:
    """Return beat times, extended back to time zero, and the tempo in BPM."""
    samples, sample_rate = librosa.load(audio_path)
    tempo, beat_frames = librosa.beat.beat_track(y=samples, sr=sample_rate)
    beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate)
    period = float(np.median(np.diff(beat_times)))
    lead_in = int(np.ceil(beat_times[0] / period))
    lead_times = beat_times[0] - period * np.arange(lead_in, 0, -1)
    return np.concatenate([lead_times, beat_times]), float(np.atleast_1d(tempo)[0])


def time_to_step(time: float, beat_times: np.ndarray, meter: Meter) -> int:
    beat_position = float(np.interp(time, beat_times, np.arange(len(beat_times))))
    return round(beat_position * meter.steps_per_beat)


def find_drones(note_events: list, beat_times: np.ndarray, meter: Meter) -> set[int]:
    """Return the pitches struck on at least DRONE_MIN_SHARE of all onset steps."""
    pitches_at_step: dict[int, set[int]] = defaultdict(set)
    for start, _, pitch, _, _ in note_events:
        pitches_at_step[time_to_step(start, beat_times, meter)].add(pitch)
    onsets_per_pitch = Counter(pitch for pitches in pitches_at_step.values() for pitch in pitches)
    return {pitch for pitch, onsets in onsets_per_pitch.items()
            if onsets / len(pitches_at_step) >= DRONE_MIN_SHARE}


def extract_melody(note_events: list, beat_times: np.ndarray, meter: Meter, drones: set[int],
                   min_confidence: float) -> list[Note]:
    """Keep the highest loud non-drone note at each grid step, cut off where the next one starts."""
    highest_at_step: dict[int, tuple[int, int]] = {}
    for start, end, pitch, amplitude, _ in note_events:
        if amplitude < min_confidence or pitch in drones or not is_playable(pitch):
            continue
        start_step, end_step = (time_to_step(t, beat_times, meter) for t in (start, end))
        if start_step not in highest_at_step or pitch > highest_at_step[start_step][0]:
            highest_at_step[start_step] = (pitch, max(end_step - start_step, 1))

    steps = sorted(highest_at_step)
    melody = []
    for step, next_step in zip(steps, steps[1:] + [None]):
        pitch, length = highest_at_step[step]
        if next_step is not None:
            length = min(length, next_step - step)
        melody.append(Note(step, length, pitch))
    return melody


def align_octaves(melody: list[Note], meter: Meter) -> None:
    """Shift notes by octaves so the tune sits in one register.

    Recordings often repeat the tune an octave higher (another instrument, an intro), so
    whole bars move first, which keeps the intervals within each bar intact. Single notes
    still more than an octave from the tune (a drone, an overtone) are then folded back.
    """
    typical_pitch = float(np.median([note.pitch for note in melody]))
    for bar in range(bar_count(melody, meter)):
        bar_notes = notes_in_bar(melody, bar, meter)
        if not bar_notes:
            continue
        wanted_octaves = (typical_pitch - np.mean([note.pitch for note in bar_notes])) / 12
        # Bars spanning several octaves (a drone under a high line) cannot move far.
        playable_octaves = [octaves for octaves in range(-4, 5)
                            if all(is_playable(note.pitch + 12 * octaves) for note in bar_notes)]
        octaves = min(playable_octaves, key=lambda candidate: abs(candidate - wanted_octaves))
        for note in bar_notes:
            note.pitch += 12 * octaves

    for note in melody:
        while note.pitch > typical_pitch + 12:
            note.pitch -= 12
        while note.pitch < typical_pitch - 12:
            note.pitch += 12


def assign_frets(melody: list[Note]) -> None:
    """Choose string and fret for each note, minimising how far the fretting hand moves."""
    def positions(pitch: int) -> list[tuple[int, int]]:
        return [(string, pitch - open_pitch) for string, open_pitch in enumerate(STANDARD_TUNING)
                if 0 <= pitch - open_pitch <= MAX_FRET]

    def move_cost(fret_from: int, fret_to: int) -> float:
        # Open strings leave the hand where it is, and frets within its span need no shift.
        return 0 if 0 in (fret_from, fret_to) else max(abs(fret_to - fret_from) - HAND_SPAN, 0)

    # Viterbi over candidate positions; the small fret term favours lower positions.
    costs = {position: 0.1 * position[1] for position in positions(melody[0].pitch)}
    back_pointers = []
    for note in melody[1:]:
        new_costs, pointers = {}, {}
        for position in positions(note.pitch):
            previous = min(costs, key=lambda p: costs[p] + move_cost(p[1], position[1]))
            new_costs[position] = costs[previous] + move_cost(previous[1], position[1]) + 0.1 * position[1]
            pointers[position] = previous
        costs = new_costs
        back_pointers.append(pointers)

    position = min(costs, key=costs.get)
    for note, pointers in zip(reversed(melody), reversed([None] + back_pointers)):
        note.string, note.fret = position
        if pointers:
            position = pointers[position]


def bar_count(melody: list[Note], meter: Meter) -> int:
    return melody[-1].step // meter.steps_per_bar + 1


def notes_in_bar(melody: list[Note], bar: int, meter: Meter) -> list[Note]:
    return [note for note in melody if note.step // meter.steps_per_bar == bar]


def note_names_text(melody: list[Note], meter: Meter) -> str:
    return "\n".join(
        f"Bar {bar + 1:3}: " + " ".join(NOTE_NAMES[note.pitch % 12] for note in notes_in_bar(melody, bar, meter))
        for bar in range(bar_count(melody, meter))
    ) + "\n"


def ascii_tab(melody: list[Note], meter: Meter) -> str:
    """One column per grid step, one tab line per BARS_PER_TAB_LINE bars."""
    steps_per_bar = meter.steps_per_bar
    columns = [["---"] * len(STANDARD_TUNING) for _ in range(bar_count(melody, meter) * steps_per_bar)]
    for note in melody:
        columns[note.step][note.string] = f"{note.fret:-<3}"

    blocks = []
    steps_per_line = steps_per_bar * BARS_PER_TAB_LINE
    for line_start in range(0, len(columns), steps_per_line):
        line_columns = columns[line_start:line_start + steps_per_line]
        bars = [line_columns[i:i + steps_per_bar] for i in range(0, len(line_columns), steps_per_bar)]
        first_bar = line_start // steps_per_bar + 1
        rows = [f"  bar {first_bar}"]
        for string, label in enumerate(STRING_LABELS):
            rows.append(f"{label}|" + "|".join("".join(column[string] for column in bar) for bar in bars) + "|")
        blocks.append("\n".join(rows))
    return "\n\n".join(blocks) + "\n"


def split_length(length: int, meter: Meter) -> list[tuple[int, bool]]:
    """Break a length in steps into (note value, dotted) pieces Guitar Pro can write."""
    # A note value spans step_note_value / value steps, half as much again when dotted.
    durations = sorted(
        ((meter.step_note_value * (3 if dotted else 2) // (2 * value), value, dotted)
         for value in GP_NOTE_VALUES for dotted in (False, True)
         if meter.step_note_value * (3 if dotted else 2) % (2 * value) == 0),
        reverse=True,
    )
    pieces = []
    while length:
        steps, value, dotted = next(d for d in durations if d[0] <= length)
        pieces.append((value, dotted))
        length -= steps
    return pieces


def add_beats(voice: guitarpro.models.Voice, length: int, note: Note | None, meter: Meter) -> None:
    for index, (value, dotted) in enumerate(split_length(length, meter)):
        beat = guitarpro.models.Beat(voice, duration=guitarpro.models.Duration(value=value, isDotted=dotted))
        if note is None:
            beat.status = guitarpro.models.BeatStatus.rest
        else:
            beat.status = guitarpro.models.BeatStatus.normal
            note_type = guitarpro.models.NoteType.tie if index else guitarpro.models.NoteType.normal
            beat.notes.append(guitarpro.models.Note(beat, value=note.fret, string=note.string + 1, type=note_type))
        voice.beats.append(beat)


def guitar_pro_song(melody: list[Note], title: str, tempo: float, meter: Meter) -> guitarpro.models.Song:
    # Guitar Pro counts tempo in quarter notes, the beat tracker in the meter's beat.
    quarters_per_beat = meter.steps_per_beat * 4 / meter.step_note_value
    song = guitarpro.models.Song(title=title, tempo=round(tempo * quarters_per_beat))
    song.measureHeaders.clear()
    track = song.tracks[0]
    track.name = "Melody"
    track.measures.clear()

    beams = ([meter.steps_per_beat] * meter.beats_per_bar + [0] * 4)[:4]
    for bar in range(bar_count(melody, meter)):
        song.newMeasure()
        header = song.measureHeaders[-1]
        header.number = bar + 1
        header.timeSignature = guitarpro.models.TimeSignature(
            numerator=meter.numerator, denominator=guitarpro.models.Duration(value=meter.denominator), beams=beams)
        header.start = guitarpro.models.Duration.quarterTime + bar * header.length

        voice = track.measures[-1].voices[0]
        bar_start, bar_end = bar * meter.steps_per_bar, (bar + 1) * meter.steps_per_bar
        cursor = bar_start
        for note in notes_in_bar(melody, bar, meter):
            if note.step > cursor:
                add_beats(voice, note.step - cursor, None, meter)
            length = min(note.length, bar_end - note.step)
            add_beats(voice, length, note, meter)
            cursor = note.step + length
        if cursor < bar_end:
            add_beats(voice, bar_end - cursor, None, meter)
    return song


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="audio file to transcribe (mp3, wav, ...)")
    parser.add_argument("--time", type=Meter.parse, default=Meter(4, 4),
                        help="time signature, e.g. 4/4, 3/4 or 6/8 (default: 4/4)")
    parser.add_argument("--min-confidence", type=float, default=0.5,
                        help="lowest basic-pitch note confidence (0-1) kept for the melody; lower it when "
                             "quiet tune notes go missing, raise it when accompaniment creeps in (default: 0.5)")
    parser.add_argument("--out", type=Path, default=Path("output"), help="folder for the results")
    args = parser.parse_args()
    if not 0 <= args.min_confidence <= 1:
        parser.error("--min-confidence must be between 0 and 1")

    out_dir = args.out / args.audio.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    _, midi, note_events = predict(args.audio, ICASSP_2022_MODEL_PATH)
    midi.write(str(out_dir / "all-notes.mid"))

    beat_times, tempo = beat_grid(args.audio)
    drones = find_drones(note_events, beat_times, args.time)
    if drones:
        print("Drone strings left out of the melody:", " ".join(note_name(pitch) for pitch in sorted(drones)))
    melody = extract_melody(note_events, beat_times, args.time, drones, args.min_confidence)
    align_octaves(melody, args.time)
    assign_frets(melody)

    (out_dir / "melody-notes.txt").write_text(note_names_text(melody, args.time))
    (out_dir / "melody-tab.txt").write_text(ascii_tab(melody, args.time))
    guitarpro.write(guitar_pro_song(melody, args.audio.stem, tempo, args.time), str(out_dir / "melody.gp5"))
    print(f"{len(melody)} melody notes, {bar_count(melody, args.time)} bars at {tempo:.0f} BPM -> {out_dir}")


if __name__ == "__main__":
    main()
