#!/usr/bin/env python3
"""
Merge room information from one set of DXF files to another.
Source: data/dxf/to_process/room_finished (Has ROOM layer)
Target: data/dxf/to_process/beam_finish_modified (Missing ROOM layer)
Output: data/dxf/to_process/beam_finish_modified (Overwrites or new folder)
"""

import os
import sys
from pathlib import Path

import ezdxf

from preprocess.dxf_extractor import DXFExtractor


def merge_rooms(room_dxf_dir, beam_dxf_dir, output_dir):
    """
    Extracts rooms from room_dxf_dir and adds them to drawings in beam_dxf_dir.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # List DXF files
    room_files = {
        f: os.path.join(room_dxf_dir, f) for f in os.listdir(room_dxf_dir) if f.lower().endswith(".dxf")
    }
    beam_files = {
        f: os.path.join(beam_dxf_dir, f) for f in os.listdir(beam_dxf_dir) if f.lower().endswith(".dxf")
    }

    # Find common files
    common_files = set(room_files.keys()) & set(beam_files.keys())
    missing_files = set(beam_files.keys()) - set(room_files.keys())

    print(f"🔍 Source (Room): {room_dxf_dir}")
    print(f"🔍 Target (Beam): {beam_dxf_dir}")
    print(f"📂 Output Dir:   {output_dir}")
    print(f"📋 Found {len(common_files)} matching DXF files.")

    if missing_files:
        print(f"⚠️ Warning: {len(missing_files)} files in target do not have corresponding room files.")

    extractor = DXFExtractor()

    processed_count = 0

    for filename in sorted(common_files):
        # paths
        src_path = room_files[filename]
        tgt_path = beam_files[filename]
        out_path = os.path.join(output_dir, filename)

        try:
            # 1. Extract room info from Source
            # We use the existing extractor logic to find ROOM layer polygons
            data = extractor.extract_from_file(src_path)
            rooms = data.get("rooms", [])

            # 2. Open Target DXF
            doc = ezdxf.readfile(tgt_path)
            msp = doc.modelspace()

            # Ensure "ROOM" layer exists in target
            if "ROOM" not in doc.layers:
                # Create ROOM layer with Cyan color (index 4)
                doc.layers.new(name="ROOM", dxfattribs={"color": 4})
            else:
                # modify existing layer color to Cyan
                room_layer = doc.layers.get("ROOM")
                room_layer.dxf.color = 4

            # 3. Add Rooms to Target
            added_rooms = 0
            for room in rooms:
                # room is a dict: {"Polygon": [{"X":..., "Y":...}, ...]}
                poly_pts = room.get("Polygon", [])
                if not poly_pts:
                    continue

                # Convert to (x, y) tuples
                points = [(pt["X"], pt["Y"]) for pt in poly_pts]

                # Create LWPolyline (Closed)
                msp.add_lwpolyline(points, close=True, dxfattribs={"layer": "ROOM"})
                added_rooms += 1

            # 4. Save
            doc.saveas(out_path)

            # print(f"  Processed {filename}: Merged {added_rooms} rooms.")
            processed_count += 1

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    print(f"✅ Successfully processed {processed_count} files into {output_dir}")


if __name__ == "__main__":
    # Determine absolute paths relative to the project root
    # Assuming this script is at <root>/preprocess/merge_rooms.py

    current_file = Path(__file__).resolve()
    # If run as script, parent is preprocess, parent.parent is root
    project_root = current_file.parent.parent

    src_dir = project_root / "data/dxf" / "to_process" / "room_finished"
    tgt_dir = project_root / "data/dxf" / "to_process" / "beam_finish_modified"

    # We output to a new folder to preserve the original inputs by default
    # But user asked to "draw them in the latter folder corresponding drawings".
    # To be safe, we create a new folder "beam_finish_modified_with_rooms"
    out_dir = project_root / "data/dxf" / "to_process" / "beam_finish_modified_with_rooms"

    if not src_dir.exists():
        print(f"Error: Source directory does not exist: {src_dir}")
        sys.exit(1)

    if not tgt_dir.exists():
        print(f"Error: Target directory does not exist: {tgt_dir}")
        sys.exit(1)

    merge_rooms(str(src_dir), str(tgt_dir), str(out_dir))
