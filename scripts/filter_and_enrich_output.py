import argparse
import copy
import gzip
import json
from pathlib import Path


def load_json(path):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def resolve_g_projection_path(output_data, explicit_path=None):
    if explicit_path:
        return Path(explicit_path)

    location_code = output_data.get("location_code")
    if not location_code:
        raise ValueError("Cannot resolve G_projection path because location_code is missing.")

    return Path("location") / location_code / f"G_projection_{location_code}.json"


def resolve_output_path(output_path):
    path = Path(output_path)
    if path.is_absolute():
        return path
    if path.parent == Path("."):
        return Path("scripts") / path.name
    return path


def load_prior_map(prior_dimensions_path, prior_set=None):
    prior_data = load_json(prior_dimensions_path)

    if prior_set:
        if prior_set not in prior_data:
            raise ValueError(
                f"Unknown prior set '{prior_set}'. Available sets: {', '.join(sorted(prior_data))}"
            )
        return {k.lower(): v for k, v in prior_data[prior_set].items()}

    merged = {}
    for _, class_map in prior_data.items():
        for class_name, dims in class_map.items():
            merged.setdefault(class_name.lower(), dims)
    return merged


def build_track_stats(selected_ids):
    return {
        track_id: {
            "frames_present": 0,
            "missing_heading_count": 0,
            "missing_speed_count": 0,
        }
        for track_id in selected_ids
    }


def is_heading_missing(obj):
    return obj.get("heading") is None


def is_speed_missing(obj):
    if "speed_kmh" not in obj or obj.get("speed_kmh") is None:
        return True

    have_heading = obj.get("have_heading")
    if have_heading is False:
        return True

    return False


def compute_position_m(sat_coords, px_per_meter):
    if sat_coords is None or px_per_meter in (None, 0):
        return None
    if len(sat_coords) < 2:
        return None
    return [
        float(sat_coords[0]) / float(px_per_meter),
        float(sat_coords[1]) / float(px_per_meter),
    ]


def compute_velocity_mps(position_m, previous_state, frame_index, fps):
    if position_m is None or previous_state is None or fps in (None, 0):
        return None

    prev_frame_index = previous_state["frame_index"]
    prev_position_m = previous_state["position_m"]
    if prev_position_m is None:
        return None

    frame_delta = frame_index - prev_frame_index
    if frame_delta <= 0:
        return None

    dt = frame_delta / float(fps)
    if dt <= 0:
        return None

    return [
        (position_m[0] - prev_position_m[0]) / dt,
        (position_m[1] - prev_position_m[1]) / dt,
    ]


def enrich_object(obj, prior_map, px_per_meter, frame_index, fps, previous_state):
    enriched = copy.deepcopy(obj)
    class_name = str(enriched.get("class", "")).strip().lower()
    dims = prior_map.get(class_name)
    enriched["dimensions_m"] = copy.deepcopy(dims) if dims is not None else None
    enriched["position_m"] = compute_position_m(enriched.get("sat_coords"), px_per_meter)
    enriched["velocity_mps"] = compute_velocity_mps(
        enriched["position_m"],
        previous_state,
        frame_index,
        fps,
    )
    return enriched


def filter_and_enrich(data, selected_ids, px_per_meter, prior_map):
    output = copy.deepcopy(data)
    output.setdefault("meta", {})
    output["meta"]["px_per_meter"] = px_per_meter
    output["selected_tracked_ids"] = list(selected_ids)
    output["selected_track_stats"] = build_track_stats(selected_ids)
    fps = output.get("meta", {}).get("fps")
    previous_states = {}

    filtered_frames = []
    for frame in output.get("frames", []):
        frame_index = frame.get("frame_index")
        selected_objects = []
        for obj in frame.get("objects", []):
            track_id = obj.get("tracked_id")
            if track_id not in selected_ids:
                continue

            stats = output["selected_track_stats"][track_id]
            stats["frames_present"] += 1
            if is_heading_missing(obj):
                stats["missing_heading_count"] += 1
            if is_speed_missing(obj):
                stats["missing_speed_count"] += 1

            enriched = enrich_object(
                obj,
                prior_map,
                px_per_meter,
                frame_index,
                fps,
                previous_states.get(track_id),
            )
            previous_states[track_id] = {
                "frame_index": frame_index,
                "position_m": enriched.get("position_m"),
            }
            selected_objects.append(enriched)

        filtered_frames.append(
            {
                "frame_index": frame_index,
                "objects": selected_objects,
            }
        )

    output["frames"] = filtered_frames
    return output


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Filter TrafficLab replay output by tracked_id, enrich it with px_per_meter "
            "and vehicle dimensions, then report missing heading/speed counts."
        )
    )
    parser.add_argument("input_path", help="Path to input output.json or output.json.gz")
    parser.add_argument("output_path", help="Path to filtered output json/json.gz")
    parser.add_argument(
        "--ids",
        nargs="+",
        type=int,
        required=True,
        help="One or more tracked_id values to keep",
    )
    parser.add_argument(
        "--g-projection",
        dest="g_projection_path",
        help="Optional explicit path to G_projection_{location}.json",
    )
    parser.add_argument(
        "--prior-dimensions",
        default="prior_dimensions.json",
        help="Path to prior_dimensions.json",
    )
    parser.add_argument(
        "--prior-set",
        help="Optional prior set name inside prior_dimensions.json, e.g. measurements_visdrone",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    selected_ids = []
    for track_id in args.ids:
        if track_id not in selected_ids:
            selected_ids.append(track_id)

    input_data = load_json(args.input_path)
    g_projection_path = resolve_g_projection_path(input_data, args.g_projection_path)
    g_projection_data = load_json(g_projection_path)
    px_per_meter = g_projection_data["parallax"]["px_per_meter"]
    prior_map = load_prior_map(args.prior_dimensions, args.prior_set)
    output_path = resolve_output_path(args.output_path)

    filtered = filter_and_enrich(input_data, selected_ids, px_per_meter, prior_map)
    write_json(output_path, filtered)

    print(f"Saved filtered output to: {output_path}")
    print(f"Using px_per_meter: {px_per_meter}")
    print("Selected track stats:")
    for track_id in selected_ids:
        stats = filtered["selected_track_stats"][track_id]
        print(
            f"  tracked_id={track_id} "
            f"frames_present={stats['frames_present']} "
            f"missing_heading={stats['missing_heading_count']} "
            f"missing_speed={stats['missing_speed_count']}"
        )


if __name__ == "__main__":
    main()
